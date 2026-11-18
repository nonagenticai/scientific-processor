"""Temporal trust tracking — append-only JSONL history stored in MinIO."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError

from log_config import get_logger

if TYPE_CHECKING:
    from clients.minio_client import MinIOClient

logger = get_logger(__name__)


class HistoryService:
    """Records and queries verification history for graph entities.

    Each record is written to its own immutable object under
    ``history/{tenant}/{entity}/{ts}-{uuid}.jsonl`` rather than appended to a
    single per-entity file. MinIO/S3 has no atomic append, so the old
    download-append-upload path lost writes when two verifications for the same
    entity raced. Per-object writes are race-free by construction; the read paths
    aggregate the objects back together.
    """

    def __init__(self, minio_client: MinIOClient | None) -> None:
        self.minio_client = minio_client

    def _entity_prefix(self, tenant_id: str, entity_id: str) -> str:
        return f"history/{tenant_id}/{entity_id}/"

    async def record(
        self,
        entity_id: str,
        tenant_id: str,
        integrity_score: float,
        components: dict[str, Any],
        trigger: str,
        cove_verdict: str | None = None,
        formal_result: dict[str, Any] | None = None,
    ) -> None:
        """Write a verification record as its own immutable object (race-free)."""
        if self.minio_client is None:
            return

        now = datetime.now(UTC)
        entry = {
            "timestamp": now.isoformat(),
            "integrity_score": integrity_score,
            "components": components,
            "trigger": trigger,
            "cove_verdict": cove_verdict,
            "formal_result": formal_result,
        }
        # Unique key per write → no read-modify-write, no lost updates under
        # concurrent verifies of the same entity.
        key = f"{self._entity_prefix(tenant_id, entity_id)}{now.strftime('%Y%m%dT%H%M%S%f')}-{uuid.uuid4().hex}.jsonl"
        data = (json.dumps(entry) + "\n").encode()
        await self.minio_client.upload(key=key, data=data, content_type="application/x-ndjson")

    async def _load_entries(self, keys: list[str]) -> list[dict[str, Any]]:
        """Download and parse all record objects, sorted chronologically."""
        client = self.minio_client
        if client is None:
            return []
        entries: list[dict[str, Any]] = []
        for key in keys:
            try:
                data = await client.download(key)
            except ClientError:
                continue
            for line in data.decode().strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed history record", key=key)
        entries.sort(key=lambda e: e.get("timestamp", ""))
        return entries

    async def get_history(self, entity_id: str, tenant_id: str) -> list[dict[str, Any]]:
        """Retrieve verification history for an entity."""
        if self.minio_client is None:
            return []

        keys = await self.minio_client.list_objects(prefix=self._entity_prefix(tenant_id, entity_id))
        return await self._load_entries(keys)

    async def get_anomalies(self, tenant_id: str, threshold_delta: float = 0.2) -> list[dict[str, Any]]:
        """Find entities whose score dropped by more than threshold_delta."""
        if self.minio_client is None:
            return []

        prefix = f"history/{tenant_id}/"
        keys = await self.minio_client.list_objects(prefix=prefix)

        # Group record objects by entity id (first path segment after the prefix).
        keys_by_entity: dict[str, list[str]] = {}
        for key in keys:
            entity_id = key.removeprefix(prefix).split("/", 1)[0]
            if entity_id:
                keys_by_entity.setdefault(entity_id, []).append(key)

        anomalies: list[dict[str, Any]] = []
        for entity_id, entity_keys in keys_by_entity.items():
            entries = await self._load_entries(entity_keys)
            if len(entries) < 2:
                continue

            try:
                prev_score = entries[-2]["integrity_score"]
                curr_score = entries[-1]["integrity_score"]
            except KeyError:
                logger.warning("History record missing integrity_score", entity_id=entity_id)
                continue

            delta = curr_score - prev_score
            if delta < -threshold_delta:
                anomalies.append(
                    {
                        "entity_id": entity_id,
                        "previous_score": prev_score,
                        "current_score": curr_score,
                        "delta": round(delta, 4),
                        "timestamp": entries[-1].get("timestamp"),
                    }
                )

        return anomalies
