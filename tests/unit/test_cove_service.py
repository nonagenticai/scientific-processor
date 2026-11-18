"""Unit tests for CoVeService — chain-of-verification with mocked LLM and graph."""

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
    """Build a minimal chat completion response dict."""
    return {"choices": [{"message": {"content": content}}]}


@pytest.fixture
def inference_mock() -> AsyncMock:
    mock = AsyncMock()
    mock.chat_completion = AsyncMock()
    return mock


@pytest.fixture
def graph_mock() -> AsyncMock:
    mock = AsyncMock()
    mock.fulltext_search = AsyncMock(return_value={"results": []})
    return mock


@pytest.fixture
def svc(inference_mock: AsyncMock, graph_mock: AsyncMock) -> CoVeService:
    return CoVeService(inference_client=inference_mock, graph_client=graph_mock)


class TestQuestionParsing:
    async def test_parses_numbered_questions(self, svc: CoVeService, inference_mock: AsyncMock) -> None:
        """Should parse numbered lines from LLM response."""
        inference_mock.chat_completion.return_value = _llm_response(
            "1. What is the dosage?\n2. What is the study size?\n3. Were controls used?"
        )
        questions = await svc._generate_questions("Drug X cures Y", 3, "tenant")
        assert len(questions) == 3
        assert "dosage" in questions[0].lower()

    async def test_fallback_on_empty_response(self, svc: CoVeService, inference_mock: AsyncMock) -> None:
        """Should return fallback question if LLM returns empty."""
        inference_mock.chat_completion.return_value = _llm_response("")
        questions = await svc._generate_questions("some claim", 3, "tenant")
        assert len(questions) == 1
        assert "accurate" in questions[0].lower()

    async def test_fallback_on_llm_error(self, svc: CoVeService, inference_mock: AsyncMock) -> None:
        """Should return fallback question if LLM call fails."""
        inference_mock.chat_completion.side_effect = httpx.ConnectError("LLM down")
        questions = await svc._generate_questions("some claim", 3, "tenant")
        assert len(questions) == 1

    async def test_caps_at_num_questions(self, svc: CoVeService, inference_mock: AsyncMock) -> None:
        """Should not return more questions than requested."""
        inference_mock.chat_completion.return_value = _llm_response("1. Q1\n2. Q2\n3. Q3\n4. Q4\n5. Q5")
        questions = await svc._generate_questions("claim", 3, "tenant")
        assert len(questions) == 3


class TestVerdictAggregation:
    async def test_verified_when_majority_supports(
        self, svc: CoVeService, inference_mock: AsyncMock, graph_mock: AsyncMock
    ) -> None:
        """Should produce 'verified' when >= 60% of questions support."""
        # Question generation returns 5 questions
        inference_mock.chat_completion.side_effect = [
            # generate_questions call
            _llm_response("1. Q1\n2. Q2\n3. Q3\n4. Q4\n5. Q5"),
            # 5 evaluate_question calls — 4 SUPPORTS, 1 INCONCLUSIVE
            _llm_response("SUPPORTS\nEvidence is clear."),
            _llm_response("SUPPORTS\nStudy confirms."),
            _llm_response("SUPPORTS\nData aligns."),
            _llm_response("SUPPORTS\nConsistent finding."),
            _llm_response("INCONCLUSIVE\nNot enough data."),
        ]
        req = CoVeRequest(claim_text="Drug X cures Y", num_questions=5)
        result = await svc.verify_claim(req, "tenant")
        assert result.overall_verdict == "verified"
        assert result.supporting_evidence == 4
        assert result.refuting_evidence == 0
        assert result.confidence == pytest.approx(0.8)

    async def test_disputed_when_refuting_majority(
        self, svc: CoVeService, inference_mock: AsyncMock, graph_mock: AsyncMock
    ) -> None:
        """Should produce 'disputed' when refuting > supporting."""
        inference_mock.chat_completion.side_effect = [
            _llm_response("1. Q1\n2. Q2\n3. Q3"),
            _llm_response("REFUTES\nContradicted."),
            _llm_response("REFUTES\nNot supported."),
            _llm_response("SUPPORTS\nPartially true."),
        ]
        req = CoVeRequest(claim_text="Claim A", num_questions=3)
        result = await svc.verify_claim(req, "tenant")
        assert result.overall_verdict == "disputed"
        assert result.refuting_evidence == 2

    async def test_inconclusive_when_split(
        self, svc: CoVeService, inference_mock: AsyncMock, graph_mock: AsyncMock
    ) -> None:
        """Should produce 'inconclusive' when evidence is split equally."""
        inference_mock.chat_completion.side_effect = [
            _llm_response("1. Q1\n2. Q2"),
            _llm_response("SUPPORTS\nOK."),
            _llm_response("REFUTES\nNot OK."),
        ]
        req = CoVeRequest(claim_text="Claim B", num_questions=2)
        result = await svc.verify_claim(req, "tenant")
        assert result.overall_verdict == "inconclusive"
        assert result.degraded is False

    async def test_degraded_flag_on_llm_outage(
        self, svc: CoVeService, inference_mock: AsyncMock, graph_mock: AsyncMock
    ) -> None:
        """A full LLM outage forces fallback question + inconclusive verdict — must be flagged degraded, not a silent 200."""
        inference_mock.chat_completion.side_effect = httpx.ConnectError("LLM down")
        req = CoVeRequest(claim_text="Claim C", num_questions=3)
        result = await svc.verify_claim(req, "tenant")
        assert result.degraded is True
        assert result.overall_verdict == "inconclusive"


class TestSourceSearch:
    async def test_find_sources_returns_list(self, svc: CoVeService, graph_mock: AsyncMock) -> None:
        """Should return results from graph search."""
        graph_mock.fulltext_search.return_value = {
            "results": [
                {"id": "n1", "name": "Study A", "confidence": 0.9, "score": 0.8},
                {"id": "n2", "name": "Study B", "confidence": 0.7, "score": 0.6},
            ]
        }
        sources = await svc._find_sources("What is X?", "tenant", 0.5)
        assert len(sources) == 2
        assert sources[0]["id"] == "n1"

    async def test_find_sources_handles_error(self, svc: CoVeService, graph_mock: AsyncMock) -> None:
        """Should return empty list on graph error."""
        graph_mock.fulltext_search.side_effect = GraphAdapterError("Neo4j down")
        sources = await svc._find_sources("question", "tenant", 0.5)
        assert sources == []

    async def test_find_sources_handles_raw_list(self, svc: CoVeService, graph_mock: AsyncMock) -> None:
        """Should handle when fulltext_search returns a plain list."""
        graph_mock.fulltext_search.return_value = [{"id": "n1", "name": "Study A", "confidence": 0.8}]
        sources = await svc._find_sources("question", "tenant", 0.5)
        assert len(sources) == 1


class TestEvaluateQuestion:
    async def test_supports_verdict(self, svc: CoVeService, inference_mock: AsyncMock) -> None:
        inference_mock.chat_completion.return_value = _llm_response("SUPPORTS\nEvidence supports the claim.")
        result = await svc._evaluate_question("claim", "question", [], "tenant")
        assert result.verdict == "supports"

    async def test_refutes_verdict(self, svc: CoVeService, inference_mock: AsyncMock) -> None:
        inference_mock.chat_completion.return_value = _llm_response("REFUTES\nEvidence contradicts the claim.")
        result = await svc._evaluate_question("claim", "question", [], "tenant")
        assert result.verdict == "refutes"

    async def test_inconclusive_on_llm_failure(self, svc: CoVeService, inference_mock: AsyncMock) -> None:
        inference_mock.chat_completion.side_effect = httpx.ConnectError("LLM error")
        result = await svc._evaluate_question("claim", "question", [], "tenant")
        assert result.verdict == "inconclusive"

    async def test_authority_score_averaged(self, svc: CoVeService, inference_mock: AsyncMock) -> None:
        inference_mock.chat_completion.return_value = _llm_response("SUPPORTS\nOK")
        sources = [
            {"id": "a", "name": "A", "confidence": 0.8},
            {"id": "b", "name": "B", "confidence": 0.6},
        ]
        result = await svc._evaluate_question("claim", "question", sources, "tenant")
        assert result.source_authority_score == pytest.approx(0.7, abs=0.01)
        assert result.source_node_ids == ["a", "b"]

    async def test_source_content_included_in_prompt(self, svc: CoVeService, inference_mock: AsyncMock) -> None:
        """Source CONTENT (graph `text`) must appear in the evaluation prompt, not just the title."""
        inference_mock.chat_completion.return_value = _llm_response("REFUTES\nContent contradicts.")
        refuting_content = "The trial found no significant effect; the drug did NOT cure the disease."
        sources = [{"id": "n1", "name": "Trial A", "confidence": 0.9, "text": refuting_content}]
        await svc._evaluate_question("Drug X cures Y", "Does the trial support it?", sources, "tenant")

        prompt = inference_mock.chat_completion.call_args.kwargs["messages"][0]["content"]
        assert refuting_content in prompt
        # And the discovery `content` key is honored too.
        inference_mock.chat_completion.reset_mock()
        inference_mock.chat_completion.return_value = _llm_response("SUPPORTS\nOK")
        abstract = "This meta-analysis confirms a strong positive effect across cohorts."
        await svc._evaluate_question(
            "claim", "q", [{"id": "d1", "name": "Paper", "confidence": 0.7, "content": abstract}], "tenant"
        )
        prompt2 = inference_mock.chat_completion.call_args.kwargs["messages"][0]["content"]
        assert abstract in prompt2

    async def test_source_without_content_graceful_fallback(self, svc: CoVeService, inference_mock: AsyncMock) -> None:
        """A source with no text/content/description still yields a valid title-only prompt."""
        inference_mock.chat_completion.return_value = _llm_response("INCONCLUSIVE\nNo content.")
        sources = [{"id": "n1", "name": "Bare Source", "confidence": 0.8}]
        result = await svc._evaluate_question("claim", "question", sources, "tenant")

        prompt = inference_mock.chat_completion.call_args.kwargs["messages"][0]["content"]
        assert "Bare Source" in prompt
        assert "confidence=0.8" in prompt
        assert result.verdict == "inconclusive"
