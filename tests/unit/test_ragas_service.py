"""Unit tests for RagasService — faithfulness, relevancy, precision with mocked backends."""

from __future__ import annotations

import os

os.environ["ENABLE_AUTH"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

from unittest.mock import AsyncMock

import httpx
import pytest

from models.verification import RagasRequest
from services.ragas_service import RagasService

pytestmark = pytest.mark.unit


def _llm_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


@pytest.fixture
def inference_mock() -> AsyncMock:
    mock = AsyncMock()
    mock.chat_completion = AsyncMock()
    return mock


@pytest.fixture
def embedding_mock() -> AsyncMock:
    mock = AsyncMock()
    mock.get_embeddings = AsyncMock()
    return mock


@pytest.fixture
def svc(inference_mock: AsyncMock, embedding_mock: AsyncMock) -> RagasService:
    return RagasService(inference_client=inference_mock, embedding_client=embedding_mock)


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert RagasService._cosine_similarity(a, b) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert RagasService._cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert RagasService._cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert RagasService._cosine_similarity(a, b) == 0.0

    def test_dimension_mismatch_returns_zero(self) -> None:
        """A dim mismatch (model/dim changed) is non-comparable -> 0.0, never a truncated/raised value."""
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0]
        assert RagasService._cosine_similarity(a, b) == 0.0


class TestParseScore:
    def test_valid_score(self) -> None:
        resp = _llm_response("0.85")
        assert RagasService._parse_score(resp) == pytest.approx(0.85)

    def test_score_with_text(self) -> None:
        resp = _llm_response("0.72 — this is quite faithful")
        assert RagasService._parse_score(resp) == pytest.approx(0.72)

    def test_clamps_above_one(self) -> None:
        resp = _llm_response("1.5")
        assert RagasService._parse_score(resp) == 1.0

    def test_clamps_below_zero(self) -> None:
        resp = _llm_response("-0.3")
        assert RagasService._parse_score(resp) == 0.0

    def test_unparseable_prose_returns_none(self) -> None:
        """A reachable judge that answers in prose must yield None, never a fabricated 0.5."""
        resp = _llm_response("I cannot rate this")
        assert RagasService._parse_score(resp) is None

    def test_empty_content_returns_none(self) -> None:
        resp = _llm_response("")
        assert RagasService._parse_score(resp) is None

    def test_empty_choices_returns_none(self) -> None:
        """A malformed envelope must degrade, not raise IndexError out of evaluate()."""
        assert RagasService._parse_score({"choices": []}) is None

    def test_missing_choices_returns_none(self) -> None:
        assert RagasService._parse_score({}) is None

    def test_non_finite_returns_none(self) -> None:
        """'nan'/'inf' parse as floats but clamp to a plausible 1.0 — that is fabrication too."""
        assert RagasService._parse_score(_llm_response("nan")) is None
        assert RagasService._parse_score(_llm_response("inf")) is None


class TestFaithfulness:
    async def test_returns_llm_score(self, svc: RagasService, inference_mock: AsyncMock) -> None:
        inference_mock.chat_completion.return_value = _llm_response("0.9")
        score = await svc._faithfulness("Answer text", ["Context 1", "Context 2"], "tenant")
        assert score == pytest.approx(0.9)

    async def test_none_on_error(self, svc: RagasService, inference_mock: AsyncMock) -> None:
        """Backend outage must yield None (degraded sentinel), never a fabricated 0.5."""
        inference_mock.chat_completion.side_effect = httpx.ConnectError("LLM down")
        score = await svc._faithfulness("Answer", ["Ctx"], "tenant")
        assert score is None

    async def test_none_when_judge_reachable_but_unparseable(
        self, svc: RagasService, inference_mock: AsyncMock
    ) -> None:
        """Judge answered, but not with a score. Propagate None — do not invent 0.5."""
        inference_mock.chat_completion.return_value = _llm_response("I cannot rate this")
        score = await svc._faithfulness("Answer", ["Ctx"], "tenant")
        assert score is None


class TestAnswerRelevancy:
    async def test_high_similarity(self, svc: RagasService, embedding_mock: AsyncMock) -> None:
        vec = [1.0, 0.0, 0.0]
        embedding_mock.get_embeddings.return_value = {"embeddings": [vec, vec]}
        score = await svc._answer_relevancy("Q?", "A.", "tenant")
        assert score == pytest.approx(1.0)

    async def test_none_on_error(self, svc: RagasService, embedding_mock: AsyncMock) -> None:
        embedding_mock.get_embeddings.side_effect = httpx.ConnectError("Backend down")
        score = await svc._answer_relevancy("Q?", "A.", "tenant")
        assert score is None

    async def test_handles_raw_list_response(self, svc: RagasService, embedding_mock: AsyncMock) -> None:
        """Some backends return a plain list instead of dict with 'embeddings' key."""
        vec_q = [1.0, 0.0]
        vec_a = [0.0, 1.0]
        embedding_mock.get_embeddings.return_value = [vec_q, vec_a]
        score = await svc._answer_relevancy("Q?", "A.", "tenant")
        assert score == pytest.approx(0.0)


class TestContextPrecision:
    async def test_single_highly_relevant_context(self, svc: RagasService, embedding_mock: AsyncMock) -> None:
        vec = [1.0, 0.0, 0.0]
        embedding_mock.get_embeddings.return_value = {"embeddings": [vec, vec]}
        score = await svc._context_precision("Q?", ["Relevant context"], "tenant")
        assert score == pytest.approx(1.0)

    async def test_multiple_contexts_rank_weighted(self, svc: RagasService, embedding_mock: AsyncMock) -> None:
        q = [1.0, 0.0]
        c1 = [0.9, 0.1]  # high similarity
        c2 = [0.1, 0.9]  # low similarity
        embedding_mock.get_embeddings.return_value = {"embeddings": [q, c1, c2]}
        score = await svc._context_precision("Q?", ["ctx1", "ctx2"], "tenant")
        # First context gets weight 1.0, second gets 0.5
        # Score should be biased toward first (high similarity) context
        assert score is not None
        assert 0.3 < score < 1.0

    async def test_none_on_error(self, svc: RagasService, embedding_mock: AsyncMock) -> None:
        embedding_mock.get_embeddings.side_effect = httpx.ConnectError("down")
        score = await svc._context_precision("Q?", ["ctx"], "tenant")
        assert score is None


class TestEvaluateEnd2End:
    async def test_full_evaluation(
        self, svc: RagasService, inference_mock: AsyncMock, embedding_mock: AsyncMock
    ) -> None:
        """Full evaluate() with mocked backends."""
        inference_mock.chat_completion.return_value = _llm_response("0.8")
        vec = [1.0, 0.0, 0.0]
        embedding_mock.get_embeddings.return_value = {"embeddings": [vec, vec]}

        req = RagasRequest(
            question="What causes X?",
            answer="X is caused by Y.",
            contexts=["Y has been shown to cause X in multiple studies."],
        )
        result = await svc.evaluate(req, "tenant")

        # Healthy path: every metric is a real score (never None) and not degraded.
        assert result.faithfulness is not None
        assert result.answer_relevancy is not None
        assert result.context_precision is not None
        assert result.factual_accuracy is not None
        assert 0.0 <= result.faithfulness <= 1.0
        assert 0.0 <= result.answer_relevancy <= 1.0
        assert 0.0 <= result.context_precision <= 1.0
        assert 0.0 <= result.factual_accuracy <= 1.0
        assert result.computation_time_ms >= 0
        # factual_accuracy = (faithfulness + context_precision) / 2
        assert result.factual_accuracy == pytest.approx((result.faithfulness + result.context_precision) / 2, abs=0.01)
        assert result.degraded is False

    async def test_unparseable_judge_reply_yields_degraded_not_fabricated_scores(
        self, svc: RagasService, inference_mock: AsyncMock, embedding_mock: AsyncMock
    ) -> None:
        """Judge REACHABLE but answering prose is indistinguishable from a real 0.5 unless we degrade."""
        inference_mock.chat_completion.return_value = _llm_response("I cannot rate this")
        vec = [1.0, 0.0, 0.0]
        embedding_mock.get_embeddings.return_value = {"embeddings": [vec, vec]}

        req = RagasRequest(
            question="What causes X?",
            answer="X is caused by Y.",
            contexts=["Y causes X."],
        )
        result = await svc.evaluate(req, "tenant")

        assert result.degraded is True
        assert result.faithfulness is None
        # factual_accuracy depends on faithfulness -> must not be half-fabricated either.
        assert result.factual_accuracy is None
        # The embedding-backed metrics were healthy and must still be real scores.
        assert result.answer_relevancy is not None
        assert result.context_precision is not None

    async def test_outage_yields_degraded_not_fabricated_scores(
        self, svc: RagasService, inference_mock: AsyncMock, embedding_mock: AsyncMock
    ) -> None:
        """A TOTAL backend outage must NOT yield 0.5 'moderately faithful' — it must be degraded + null metrics."""
        inference_mock.chat_completion.side_effect = httpx.ConnectError("LLM down")
        embedding_mock.get_embeddings.side_effect = httpx.ConnectError("Embeddings down")

        req = RagasRequest(
            question="What causes X?",
            answer="X is caused by Y.",
            contexts=["Y causes X."],
        )
        result = await svc.evaluate(req, "tenant")

        assert result.degraded is True
        assert result.faithfulness is None
        assert result.answer_relevancy is None
        assert result.context_precision is None
        assert result.factual_accuracy is None
        # Crucially: no fabricated 0.5 anywhere.
        assert 0.5 not in (
            result.faithfulness,
            result.answer_relevancy,
            result.context_precision,
            result.factual_accuracy,
        )
