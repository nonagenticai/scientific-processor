"""Autonomous verification pipeline — orchestrates integrity, CoVe, and formal verification."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import networkx
from botocore.exceptions import ClientError

from clients.graph_adapter import GraphAdapterError
from log_config import get_logger
from models.pipeline import EntityVerificationResult, PipelineStatusResponse
from models.verification import CoVeRequest

if TYPE_CHECKING:
    from clients.graph_adapter import GraphAdapterClient
    from clients.minio_client import MinIOClient
    from services.cove_service import CoVeService
    from services.formal_service import FormalService
    from services.history_service import HistoryService
    from services.integrity_service import IntegrityService

logger = get_logger(__name__)


@dataclass
class PipelineState:
    """Mutable pipeline counters shared across PipelineService instances.

    ``get_pipeline_service`` is a per-request FastAPI dependency, so each request
    builds a fresh PipelineService. Keeping the counters on the instance made
    ``/status`` always report zeros. Passing one process-wide PipelineState into
    every request-scoped service makes status reflect real work. ``running_batches``
    is a counter (not a bool) so overlapping batches — e.g. a webhook and a
    scheduled batch — report ``running`` correctly and never wedge it True.
    """

    running_batches: int = 0
    processed_total: int = 0
    failed_total: int = 0
    last_run_at: str | None = None
    last_run_duration: float = 0.0


class PipelineService:
    """Orchestrates the full verification pipeline for graph entities."""

    def __init__(
        self,
        integrity_service: IntegrityService,
        cove_service: CoVeService,
        formal_service: FormalService,
        graph_client: GraphAdapterClient,
        minio_client: MinIOClient | None,
        settings: Any,
        history_service: HistoryService | None = None,
        state: PipelineState | None = None,
    ) -> None:
        self.integrity_service = integrity_service
        self.cove_service = cove_service
        self.formal_service = formal_service
        self.graph_client = graph_client
        self.minio_client = minio_client
        self.settings = settings
        self.history_service = history_service

        # Pipeline counters. Defaults to a fresh per-instance state; the request
        # factory injects one process-wide PipelineState so /status survives the
        # discarded per-request service instance.
        self._state = state if state is not None else PipelineState()

    async def verify_entity(
        self,
        entity_id: str,
        tenant_id: str,
        write_back: bool = False,
        run_cove: bool = False,
        run_formal: bool = False,
        max_hops: int = 2,
    ) -> EntityVerificationResult:
        """Run full verification pipeline for a single entity."""
        start = time.monotonic()
        try:
            # Step 1: Compute integrity score
            integrity = await self.integrity_service.compute_integrity(
                entity_id=entity_id,
                tenant_id=tenant_id,
                write_back=write_back,
            )

            score = integrity.integrity_score
            components = [c.model_dump() if hasattr(c, "model_dump") else c for c in integrity.components]

            # Carry the degradation diagnosis across the pipeline boundary.
            # `degraded` is derived from the score this result actually holds
            # rather than copied from the IntegrityResponse flag, so the two can
            # never disagree — integrity_service sets the score to None exactly
            # when a component could not be measured (`degraded = bool(
            # degraded_components)`), so this is the same invariant.
            degraded = score is None
            degraded_components: list[str] = []
            degraded_reason: str | None = None
            if degraded:
                degraded_components = [str(c) for c in (getattr(integrity, "degraded_components", None) or [])]
                reason = getattr(integrity, "degraded_reason", None)
                degraded_reason = reason if isinstance(reason, str) else None

            # Carried unconditionally — a renormalized score is NOT degraded, so it
            # would otherwise be invisible to the caller. This is the field that
            # distinguishes "scored, with nothing to cite" from "could not measure".
            not_applicable = [str(c) for c in (getattr(integrity, "not_applicable_components", None) or [])]

            result = EntityVerificationResult(
                entity_id=entity_id,
                integrity_score=score,
                components=components,
                degraded=degraded,
                degraded_components=degraded_components,
                degraded_reason=degraded_reason,
                not_applicable_components=not_applicable,
            )

            # Steps 2/3: escalate when the score is below threshold, forced, or
            # UNKNOWN. A degraded integrity score (score is None) must not be read
            # as "fine" — an unmeasurable entity is exactly the one worth a second
            # opinion, so fail towards more verification, not less.
            cove_threshold = self.settings.pipeline_auto_cove_threshold
            if run_cove or score is None or score < cove_threshold:
                result.cove_verdict = await self._run_cove(entity_id, tenant_id)

            formal_threshold = self.settings.pipeline_auto_formal_threshold
            if run_formal or score is None or score < formal_threshold:
                result.formal_result = await self._run_formal(entity_id, tenant_id)

            elapsed = (time.monotonic() - start) * 1000
            result.computation_time_ms = round(elapsed, 2)

            # Step 4: Record in history
            if self.history_service and score is not None:
                try:
                    await self.history_service.record(
                        entity_id=entity_id,
                        tenant_id=tenant_id,
                        integrity_score=score,
                        components={c.get("name", ""): c.get("score", 0) for c in components if isinstance(c, dict)},
                        trigger="pipeline",
                        cove_verdict=result.cove_verdict,
                        formal_result=result.formal_result,
                    )
                except ClientError, OSError:
                    logger.warning("Failed to record history", entity_id=entity_id, exc_info=True)

            return result

        except (GraphAdapterError, httpx.HTTPError, KeyError, ValueError, networkx.NetworkXError) as exc:
            elapsed = (time.monotonic() - start) * 1000
            logger.warning("Pipeline verification failed", entity_id=entity_id, error=str(exc))
            return EntityVerificationResult(
                entity_id=entity_id,
                error=str(exc),
                computation_time_ms=round(elapsed, 2),
            )

    async def verify_batch(
        self,
        entity_ids: list[str],
        tenant_id: str,
        write_back: bool = False,
        run_cove: bool = False,
        run_formal: bool = False,
    ) -> list[EntityVerificationResult]:
        """Verify a batch of entities with concurrency control."""
        self._state.running_batches += 1
        batch_start = time.monotonic()
        concurrency = getattr(self.settings, "pipeline_concurrency", 10)
        semaphore = asyncio.Semaphore(concurrency)

        async def _limited(eid: str) -> EntityVerificationResult:
            async with semaphore:
                return await self.verify_entity(
                    eid,
                    tenant_id,
                    write_back=write_back,
                    run_cove=run_cove,
                    run_formal=run_formal,
                )

        try:
            results = await asyncio.gather(*[_limited(eid) for eid in entity_ids])
            result_list = list(results)

            # Update counters. These read-modify-writes have no interleaving await
            # between them, so they are safe under concurrent batches on one loop.
            self._state.processed_total += len(result_list)
            self._state.failed_total += sum(1 for r in result_list if r.error)
            self._state.last_run_at = datetime.now(UTC).isoformat()
            self._state.last_run_duration = time.monotonic() - batch_start

            return result_list
        finally:
            self._state.running_batches -= 1

    async def find_unscored_entities(self, tenant_id: str, limit: int = 50) -> list[str]:
        """Query graph for entities missing integrity scores.

        Uses the structured /nodes/missing-property endpoint (raw Cypher is gated
        off in prod; this also drops the old string-interpolated LIMIT).
        """
        try:
            result = await self.graph_client.find_nodes_missing_property(
                "integrity_score", limit=limit, tenant_id=tenant_id
            )
            rows = result.get("results", [])
            return [str(r["id"]) for r in rows if r.get("id")]
        except GraphAdapterError, httpx.HTTPError, KeyError:
            logger.warning("Failed to query unscored entities", exc_info=True)
            return []

    def get_status(self) -> PipelineStatusResponse:
        """Return current pipeline status."""
        throughput = 0.0
        if self._state.last_run_duration > 0:
            throughput = round(self._state.processed_total / (self._state.last_run_duration / 60), 2)

        return PipelineStatusResponse(
            running=self._state.running_batches > 0,
            queue_depth=0,
            processed_total=self._state.processed_total,
            failed_total=self._state.failed_total,
            last_run_at=self._state.last_run_at,
            throughput_per_minute=throughput,
        )

    async def _run_cove(self, entity_id: str, tenant_id: str) -> str | None:
        """Run CoVe verification for an entity, returning the verdict."""
        try:
            node = await self.graph_client.get_node(entity_id, tenant_id)
            claim_text = node.get("properties", {}).get("name") or node.get("name", entity_id)

            cove_req = CoVeRequest(claim_text=claim_text)
            cove_result = await self.cove_service.verify_claim(cove_req, tenant_id)
            return cove_result.overall_verdict
        except GraphAdapterError, httpx.HTTPError, ValueError, KeyError:
            logger.warning("CoVe verification failed in pipeline", entity_id=entity_id, exc_info=True)
            return None

    async def _run_formal(self, entity_id: str, tenant_id: str) -> dict | None:
        """Run formal verification for an entity."""
        try:
            node = await self.graph_client.get_node(entity_id, tenant_id)
            claim_text = node.get("properties", {}).get("name") or node.get("name", entity_id)

            translate_result = await self.formal_service.translate(claim_text, "z3", tenant_id)
            verify_result = await self.formal_service.verify(
                translate_result.formal_expression, translate_result.target_language
            )
            return (
                verify_result.model_dump()
                if hasattr(verify_result, "model_dump")
                else {"is_valid": verify_result.is_valid}
            )
        except GraphAdapterError, httpx.HTTPError, ValueError, KeyError:
            logger.warning("Formal verification failed in pipeline", entity_id=entity_id, exc_info=True)
            return None
