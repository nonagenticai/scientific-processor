"""Chain-of-Verification service — generate fact-checking questions, verify against graph + external sources."""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

import httpx

from clients.graph_adapter import GraphAdapterClient, GraphAdapterError
from clients.inference_router import InferenceRouterClient
from log_config import get_logger
from models.verification import CoVeRequest, CoVeResponse, VerificationQuestion

if TYPE_CHECKING:
    from clients.discovery_engine import DiscoveryEngineClient

logger = get_logger(__name__)

# Max chars of source content shown to the LLM per source in the evidence block.
# No shared truncation helper exists in this repo — inline [:N] like everywhere else.
_SOURCE_CONTENT_CHARS = 1500


class CoVeService:
    """Chain-of-Verification: generate fact-checking questions, verify against sources."""

    def __init__(
        self,
        inference_client: InferenceRouterClient,
        graph_client: GraphAdapterClient,
        discovery_client: DiscoveryEngineClient | None = None,
    ) -> None:
        self.inference = inference_client
        self.graph = graph_client
        self.discovery = discovery_client

    async def verify_claim(self, request: CoVeRequest, tenant_id: str) -> CoVeResponse:
        start = time.monotonic()

        # Collects a marker whenever an LLM call falls back (outage). A populated
        # list => the verdict rests on fallback questions / forced "inconclusive"
        # rather than a real chain-of-verification -> surface `degraded`.
        degraded: list[bool] = []

        # Step 1: Generate verification questions via LLM
        questions = await self._generate_questions(
            request.claim_text, request.num_questions, tenant_id, degraded=degraded
        )

        # Step 2: For each question, find sources and evaluate (in parallel)
        async def _verify_one(q: str) -> VerificationQuestion:
            sources = await self._find_sources(q, tenant_id, request.source_authority_threshold)
            return await self._evaluate_question(request.claim_text, q, sources, tenant_id, degraded=degraded)

        verified_questions = list(await asyncio.gather(*[_verify_one(q) for q in questions]))

        # Step 3: Aggregate verdicts
        supporting = sum(1 for q in verified_questions if q.verdict == "supports")
        refuting = sum(1 for q in verified_questions if q.verdict == "refutes")

        if supporting > refuting and supporting >= len(verified_questions) * 0.6:
            overall = "verified"
        elif refuting > supporting:
            overall = "disputed"
        else:
            overall = "inconclusive"

        confidence = supporting / len(verified_questions) if verified_questions else 0.0
        elapsed = (time.monotonic() - start) * 1000

        return CoVeResponse(
            claim_text=request.claim_text,
            questions=verified_questions,
            overall_verdict=overall,
            confidence=confidence,
            supporting_evidence=supporting,
            refuting_evidence=refuting,
            computation_time_ms=round(elapsed, 2),
            degraded=bool(degraded),
        )

    async def _generate_questions(
        self, claim: str, num_questions: int, tenant_id: str, *, degraded: list[bool] | None = None
    ) -> list[str]:
        """Use LLM to generate fact-checking questions."""
        prompt = (
            f"Generate {num_questions} specific, verifiable questions that would help determine "
            f"if the following claim is true or false. Each question should test a different aspect "
            f"of the claim.\n\n"
            f"Claim: {claim}\n\n"
            f"Return ONLY the questions, one per line, numbered 1-{num_questions}."
        )

        try:
            response = await self.inference.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                tenant_id=tenant_id,
                schema_id="sp-cove-questions",
            )
        except httpx.HTTPError, json.JSONDecodeError:
            logger.warning("LLM call failed for question generation", exc_info=True)
            if degraded is not None:
                degraded.append(True)
            return [f"Is the following claim accurate: {claim}?"]

        content = (response.get("choices") or [{}])[0].get("message", {}).get("content", "")
        questions: list[str] = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                q = line.lstrip("0123456789.)-: ")
                if q:
                    questions.append(q)
        return questions[:num_questions] if questions else [f"Is the following claim accurate: {claim}?"]

    async def _find_sources(self, question: str, tenant_id: str, authority_threshold: float) -> list[dict]:
        """Search graph + discovery engine for relevant sources."""
        # 1. Search knowledge graph (existing)
        graph_sources = await self._search_graph(question, tenant_id)

        # 2. Search Discovery Engine (graceful degradation)
        external_sources = await self._search_discovery_engine(question, tenant_id)

        # 3. Merge
        return graph_sources + external_sources

    async def _search_graph(self, question: str, tenant_id: str) -> list[dict]:
        """Search graph for relevant source nodes via fulltext index.

        Uses the structured /nodes/fulltext-search endpoint (raw Cypher is gated
        off in prod) — same {id,name,labels,confidence,score} rows as before.
        """
        try:
            result = await self.graph.fulltext_search(question, min_score=0.5, limit=10, tenant_id=tenant_id)
            if isinstance(result, list):
                return result
            return result.get("results", [])
        except GraphAdapterError, KeyError, TypeError:
            logger.warning("Graph search failed for CoVe question", exc_info=True)
            return []

    async def _search_discovery_engine(self, question: str, tenant_id: str) -> list[dict]:
        """Search Discovery Engine for external academic sources."""
        if self.discovery is None:
            return []

        try:
            result = await self.discovery.search(query=question, limit=5, tenant_id=tenant_id)
            # Normalize to same format as graph sources
            sources = []
            for item in result.get("results", []):
                sources.append(
                    {
                        "id": item.get("doi") or item.get("id", ""),
                        "name": item.get("title", "Unknown"),
                        "labels": ["ExternalPaper"],
                        "confidence": item.get("confidence", 0.5),
                        "content": item.get("abstract"),
                        "source": "discovery_engine",
                    }
                )
            return sources
        except httpx.HTTPError, KeyError, TypeError:
            logger.warning("Discovery Engine search failed, continuing with graph-only", exc_info=True)
            return []

    async def _evaluate_question(
        self,
        claim: str,
        question: str,
        sources: list[dict],
        tenant_id: str,
        *,
        degraded: list[bool] | None = None,
    ) -> VerificationQuestion:
        """Use LLM to evaluate a verification question against sources."""
        lines: list[str] = []
        for s in sources[:5]:
            name = s.get("name", "Unknown")
            confidence = s.get("confidence", "N/A")
            # Graph sources carry `text` (+ `description`); discovery sources carry `content`.
            raw_content = s.get("text") or s.get("content") or s.get("description")
            if raw_content:
                content = str(raw_content).strip()[:_SOURCE_CONTENT_CHARS]
                lines.append(f"- {name} (confidence={confidence}):\n  {content}")
            else:
                # Graceful fallback: no content available (e.g. before graph_adapter's
                # fulltext PR ships `text`) — keep the title-only line so nothing breaks.
                lines.append(f"- {name}: confidence={confidence}")
        source_text = "\n".join(lines) or "No sources found."

        prompt = (
            "Given the following claim and verification question, evaluate whether the "
            "sources and their content support or refute the claim. Judge the claim "
            "against the CONTENT of each source, not merely its title.\n\n"
            f"Claim: {claim}\n"
            f"Question: {question}\n"
            f"Available sources:\n{source_text}\n\n"
            "Respond with EXACTLY one of: SUPPORTS, REFUTES, or INCONCLUSIVE\n"
            "Then on a new line, provide a brief explanation."
        )

        try:
            response = await self.inference.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                tenant_id=tenant_id,
                schema_id="sp-cove-evaluate",
            )
        except httpx.HTTPError, json.JSONDecodeError:
            logger.warning("LLM call failed for question evaluation", exc_info=True)
            if degraded is not None:
                degraded.append(True)
            return VerificationQuestion(question=question, verdict="inconclusive")

        content = (response.get("choices") or [{}])[0].get("message", {}).get("content", "")
        first_line = content.strip().split("\n")[0].upper() if content.strip() else ""

        if "SUPPORTS" in first_line:
            verdict = "supports"
        elif "REFUTES" in first_line:
            verdict = "refutes"
        else:
            verdict = "inconclusive"

        avg_authority = sum(s.get("confidence", 0) or 0 for s in sources) / max(len(sources), 1)

        return VerificationQuestion(
            question=question,
            answer=content,
            source_node_ids=[str(s.get("id", "")) for s in sources[:5]],
            source_authority_score=round(avg_authority, 4),
            verdict=verdict,
        )
