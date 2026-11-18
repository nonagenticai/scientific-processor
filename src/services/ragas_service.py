"""RAGAS-inspired faithfulness and accuracy scoring service."""

from __future__ import annotations

import asyncio
import math
import time

import httpx
import numpy as np

from clients.inference_router import InferenceRouterClient
from clients.pytorch_backend import PyTorchBackendClient
from log_config import get_logger
from models.verification import RagasRequest, RagasResponse

logger = get_logger(__name__)


class RagasService:
    """RAGAS-inspired metrics for faithfulness and accuracy."""

    def __init__(
        self,
        inference_client: InferenceRouterClient,
        embedding_client: PyTorchBackendClient,
    ) -> None:
        self.inference = inference_client
        self.embeddings = embedding_client

    async def evaluate(self, request: RagasRequest, tenant_id: str) -> RagasResponse:
        start = time.monotonic()

        faithfulness, relevancy, precision = await asyncio.gather(
            self._faithfulness(request.answer, request.contexts, tenant_id),
            self._answer_relevancy(request.question, request.answer, tenant_id),
            self._context_precision(request.question, request.contexts, tenant_id),
        )

        # A metric is None when its backend was unreachable / returned garbage.
        # Do NOT fabricate a 0.5 "moderately faithful" score on an outage —
        # surface None + a degraded flag so callers can tell them apart.
        if faithfulness is not None and precision is not None:
            factual_accuracy: float | None = (faithfulness + precision) / 2.0
        else:
            factual_accuracy = None

        degraded = None in (faithfulness, relevancy, precision)
        elapsed = (time.monotonic() - start) * 1000

        return RagasResponse(
            faithfulness=self._round_or_none(faithfulness),
            answer_relevancy=self._round_or_none(relevancy),
            context_precision=self._round_or_none(precision),
            factual_accuracy=self._round_or_none(factual_accuracy),
            computation_time_ms=round(elapsed, 2),
            degraded=degraded,
        )

    async def _faithfulness(self, answer: str, contexts: list[str], tenant_id: str) -> float | None:
        """Check if answer claims are grounded in contexts. LLM-as-judge.

        Returns None when the judge was unreachable OR was reachable but did not
        answer with a parseable score — both are `degraded`, not a real score.
        """
        context_text = "\n---\n".join(contexts[:5])
        prompt = (
            "Evaluate the faithfulness of the following answer based on the given contexts.\n"
            "Faithfulness measures whether every claim in the answer can be inferred from the contexts.\n\n"
            f"Contexts:\n{context_text}\n\n"
            f"Answer: {answer}\n\n"
            "Rate faithfulness from 0.0 (no claims supported) to 1.0 (all claims supported).\n"
            "Respond with ONLY a number between 0.0 and 1.0."
        )

        try:
            response = await self.inference.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                tenant_id=tenant_id,
                schema_id="sp-ragas-faithfulness",
            )
            return self._parse_score(response)
        except httpx.HTTPError, ValueError, KeyError:
            logger.warning("LLM call failed for faithfulness evaluation", exc_info=True)
            return None

    async def _answer_relevancy(self, question: str, answer: str, tenant_id: str) -> float | None:
        """Semantic similarity between question and answer via embeddings.

        Returns None (not a fabricated 0.5) when the embedding backend is
        unreachable or the response is malformed — callers see `degraded`.
        """
        try:
            result = await self.embeddings.get_embeddings([question, answer], tenant_id=tenant_id)
            embeddings = result.get("embeddings", result) if isinstance(result, dict) else result
            if isinstance(embeddings, list) and len(embeddings) >= 2:
                return self._cosine_similarity(embeddings[0], embeddings[1])
        except httpx.HTTPError, ValueError, KeyError, TypeError:
            logger.warning("Embedding failed for answer relevancy", exc_info=True)
            return None
        return None

    async def _context_precision(self, question: str, contexts: list[str], tenant_id: str) -> float | None:
        """Rank-weighted relevance of contexts to the question.

        Returns None on backend outage / malformed response (degraded); a real
        0.0 only when there genuinely are no context scores to weigh.
        """
        try:
            all_texts = [question, *contexts[:10]]
            result = await self.embeddings.get_embeddings(all_texts, tenant_id=tenant_id)
            embeddings = result.get("embeddings", result) if isinstance(result, dict) else result
            if not isinstance(embeddings, list) or len(embeddings) < 2:
                return None

            q_emb = embeddings[0]
            scores = [self._cosine_similarity(q_emb, c_emb) for c_emb in embeddings[1:]]
            if not scores:
                return 0.0

            # Rank-weighted: higher weight for earlier contexts
            weights = [1.0 / (i + 1) for i in range(len(scores))]
            weighted_sum = sum(s * w for s, w in zip(scores, weights, strict=True))
            return min(1.0, weighted_sum / sum(weights))
        except httpx.HTTPError, ValueError, KeyError, TypeError:
            logger.warning("Embedding failed for context precision", exc_info=True)
            return None

    @staticmethod
    def _round_or_none(value: float | None) -> float | None:
        return round(value, 4) if value is not None else None

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        # Defensive dimension guard (mirrors media_discovery.cosine_similarity):
        # a length mismatch means the model/dim changed and the vectors aren't
        # comparable — return 0.0 rather than letting numpy raise or truncating.
        if len(a) != len(b):
            return 0.0
        a_arr = np.array(a, dtype=np.float64)
        b_arr = np.array(b, dtype=np.float64)
        dot = float(np.dot(a_arr, b_arr))
        norm = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
        return dot / norm if norm > 0 else 0.0

    @staticmethod
    def _parse_score(response: dict) -> float | None:
        """Extract the judge's 0.0-1.0 score, or None when it did not give one.

        A REACHABLE judge that answers in prose ("I cannot rate this"), returns an
        empty/malformed envelope, or emits nan/inf has not scored anything. Returning
        a fabricated 0.5 there made "moderately faithful" indistinguishable from "the
        judge never answered" — the caller saw a plausible number with degraded=False.
        None propagates through `_faithfulness` into the `degraded` flag instead.
        """
        try:
            content = response["choices"][0]["message"]["content"]
            score = float(content.strip().split()[0])
        except AttributeError, IndexError, KeyError, TypeError, ValueError:
            logger.warning("LLM judge returned no parseable score — reporting degraded, not a fabricated 0.5")
            return None

        if not math.isfinite(score):
            logger.warning("LLM judge returned a non-finite score — reporting degraded")
            return None
        return max(0.0, min(1.0, score))
