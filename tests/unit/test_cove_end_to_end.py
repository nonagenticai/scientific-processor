"""End-to-end tests for CoVeService — full pipeline with mocked LLM and graph.

Exercises the complete verify_claim flow:
  generate questions -> find sources -> evaluate each question -> aggregate verdicts
"""

from __future__ import annotations

import os

os.environ["ENABLE_AUTH"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

from unittest.mock import AsyncMock

import httpx
import pytest

from clients.graph_adapter import GraphAdapterError
from models.verification import CoVeRequest
from services.cove_service import CoVeService

pytestmark = pytest.mark.unit


def _llm_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


@pytest.fixture
def inference_mock() -> AsyncMock:
    mock = AsyncMock()
    mock.chat_completion = AsyncMock()
    return mock


@pytest.fixture
def graph_mock() -> AsyncMock:
    mock = AsyncMock()
    mock.fulltext_search = AsyncMock()
    return mock


@pytest.fixture
def svc(inference_mock, graph_mock) -> CoVeService:
    return CoVeService(inference_client=inference_mock, graph_client=graph_mock)


class TestFullPipelineVerified:
    """End-to-end: majority SUPPORTS -> verified."""

    async def test_full_verified_pipeline(self, svc, inference_mock, graph_mock):
        """
        Pipeline flow:
        1. LLM generates 3 numbered questions
        2. Graph returns source nodes for each question
        3. LLM evaluates each question -> 2 SUPPORTS, 1 INCONCLUSIVE
        4. Result: verified (2/3 = 66% >= 60%), confidence = 2/3
        """
        # Call sequence: generate_questions, then 3x evaluate_question
        inference_mock.chat_completion.side_effect = [
            # generate_questions
            _llm_response("1. Is the drug effective?\n2. Were trials randomized?\n3. Is the sample size adequate?"),
            # evaluate_question for Q1
            _llm_response("SUPPORTS\nMultiple trials confirm efficacy."),
            # evaluate_question for Q2
            _llm_response("SUPPORTS\nTrials were double-blind randomized."),
            # evaluate_question for Q3
            _llm_response("INCONCLUSIVE\nSample size not reported in sources."),
        ]

        # Graph returns sources for each question search
        graph_mock.fulltext_search.return_value = {
            "results": [
                {"id": "study1", "name": "RCT 2024", "confidence": 0.95, "score": 0.9},
                {"id": "study2", "name": "Meta-Analysis 2023", "confidence": 0.88, "score": 0.85},
            ]
        }

        req = CoVeRequest(claim_text="Drug X cures disease Y", num_questions=3)
        result = await svc.verify_claim(req, "test-tenant")

        # Verify overall aggregation
        assert result.overall_verdict == "verified"
        assert result.supporting_evidence == 2
        assert result.refuting_evidence == 0
        assert result.confidence == pytest.approx(2 / 3, abs=0.01)

        # Verify questions were parsed correctly
        assert len(result.questions) == 3
        assert result.questions[0].verdict == "supports"
        assert result.questions[1].verdict == "supports"
        assert result.questions[2].verdict == "inconclusive"

        # Verify source_node_ids are populated from graph search
        for q in result.questions:
            assert q.source_node_ids == ["study1", "study2"]
            assert q.source_authority_score > 0

        # Verify timing
        assert result.computation_time_ms > 0
        assert result.claim_text == "Drug X cures disease Y"


class TestFullPipelineDisputed:
    """End-to-end: majority REFUTES -> disputed."""

    async def test_full_disputed_pipeline(self, svc, inference_mock, graph_mock):
        inference_mock.chat_completion.side_effect = [
            _llm_response("1. Is there causal evidence?\n2. Are results reproducible?"),
            _llm_response("REFUTES\nNo causal link established."),
            _llm_response("REFUTES\nResults not reproducible."),
        ]
        graph_mock.fulltext_search.return_value = {"results": [{"id": "src1", "name": "Review", "confidence": 0.7}]}

        req = CoVeRequest(claim_text="Supplement Z prevents cancer", num_questions=2)
        result = await svc.verify_claim(req, "tenant")

        assert result.overall_verdict == "disputed"
        assert result.refuting_evidence == 2
        assert result.supporting_evidence == 0
        assert result.confidence == pytest.approx(0.0)


class TestFullPipelineInconclusive:
    """End-to-end: even split -> inconclusive."""

    async def test_full_inconclusive_pipeline(self, svc, inference_mock, graph_mock):
        inference_mock.chat_completion.side_effect = [
            _llm_response("1. Q1\n2. Q2\n3. Q3\n4. Q4"),
            _llm_response("SUPPORTS\nOK."),
            _llm_response("REFUTES\nNot OK."),
            _llm_response("SUPPORTS\nPartially."),
            _llm_response("REFUTES\nContradicted."),
        ]
        graph_mock.fulltext_search.return_value = {"results": []}

        req = CoVeRequest(claim_text="Claim with mixed evidence", num_questions=4)
        result = await svc.verify_claim(req, "tenant")

        # 2 supports, 2 refutes -> not > and not >= 60% -> inconclusive
        assert result.overall_verdict == "inconclusive"
        assert result.supporting_evidence == 2
        assert result.refuting_evidence == 2


class TestFullPipelineNoSources:
    """End-to-end: graph returns no sources, LLM still evaluates."""

    async def test_no_sources_pipeline(self, svc, inference_mock, graph_mock):
        inference_mock.chat_completion.side_effect = [
            _llm_response("1. Is there evidence?\n2. Are experts in agreement?"),
            _llm_response("INCONCLUSIVE\nNo sources to evaluate."),
            _llm_response("INCONCLUSIVE\nNo data available."),
        ]
        graph_mock.fulltext_search.return_value = {"results": []}

        req = CoVeRequest(claim_text="Unverifiable claim", num_questions=2)
        result = await svc.verify_claim(req, "tenant")

        assert result.overall_verdict == "inconclusive"
        assert result.supporting_evidence == 0
        assert result.refuting_evidence == 0
        assert result.confidence == pytest.approx(0.0)

        # With no sources, authority score should be 0
        for q in result.questions:
            assert q.source_authority_score == pytest.approx(0.0)
            assert q.source_node_ids == []


class TestFullPipelineGraphError:
    """End-to-end: graph client errors are handled gracefully."""

    async def test_graph_error_graceful(self, svc, inference_mock, graph_mock):
        """When graph search fails, pipeline continues with empty sources."""
        inference_mock.chat_completion.side_effect = [
            _llm_response("1. Is X true?"),
            _llm_response("INCONCLUSIVE\nNo sources available."),
        ]
        graph_mock.fulltext_search.side_effect = GraphAdapterError("Neo4j connection refused")

        req = CoVeRequest(claim_text="Some claim", num_questions=1)
        result = await svc.verify_claim(req, "tenant")

        # Should still return a valid response (not crash)
        assert result.overall_verdict == "inconclusive"
        assert len(result.questions) == 1
        assert result.questions[0].source_node_ids == []


class TestFullPipelineLLMFallback:
    """End-to-end: LLM failure during question generation -> fallback question."""

    async def test_llm_failure_generates_fallback(self, svc, inference_mock, graph_mock):
        inference_mock.chat_completion.side_effect = [
            httpx.ConnectError(
                "LLM down"
            ),  # question generation fails — httpx.ConnectError is what chat_completion raises on network failure
            _llm_response("INCONCLUSIVE\nCan't verify."),  # evaluate fallback question
        ]
        graph_mock.fulltext_search.return_value = {"results": []}

        req = CoVeRequest(claim_text="Fallback claim", num_questions=3)
        result = await svc.verify_claim(req, "tenant")

        # Should use fallback question
        assert len(result.questions) == 1
        assert result.overall_verdict == "inconclusive"


class TestAuthorityScoreAggregation:
    """Verify source authority averaging in the full pipeline."""

    async def test_authority_averaged_across_sources(self, svc, inference_mock, graph_mock):
        inference_mock.chat_completion.side_effect = [
            _llm_response("1. Is this proven?"),
            _llm_response("SUPPORTS\nAll sources agree."),
        ]
        graph_mock.fulltext_search.return_value = {
            "results": [
                {"id": "a", "name": "A", "confidence": 0.9},
                {"id": "b", "name": "B", "confidence": 0.6},
                {"id": "c", "name": "C", "confidence": 0.3},
            ]
        }

        req = CoVeRequest(claim_text="Test authority", num_questions=1)
        result = await svc.verify_claim(req, "tenant")

        # Average of 0.9, 0.6, 0.3 = 0.6
        assert result.questions[0].source_authority_score == pytest.approx(0.6, abs=0.01)
        assert result.questions[0].source_node_ids == ["a", "b", "c"]
