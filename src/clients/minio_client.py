"""Async MinIO/S3-compatible client using aioboto3."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any

import aioboto3
from botocore.exceptions import ClientError

from log_config import get_logger

logger = get_logger(__name__)


class MinIOClient:
    """S3 client over aioboto3, holding ONE client for the life of the instance.

    Building a botocore client is *synchronous* work — it loads and parses the S3
    service model JSON off disk and constructs the signer/endpoint stack — and it
    happens inside ``Session.client(...).__aenter__``, i.e. on the event loop
    thread with no ``await`` to yield at. Measured on this service's own venv:
    ~172 ms for the first client in a process, ~2-3 ms for each one after (the
    model parse is cached, the client build is not). Opening a fresh client per
    operation therefore froze the whole worker — every concurrent request, /health
    included — on every single upload/download/delete.

    One lazily-created client, built on first use and kept open, pays that cost
    once per process instead of once per call.
    """

    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str) -> None:
        self.endpoint = endpoint if endpoint.startswith("http") else f"http://{endpoint}"
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket
        self._session = aioboto3.Session()
        # Built lazily on first use so constructing MinIOClient outside a running
        # event loop stays cheap and side-effect free.
        self._stack: AsyncExitStack | None = None
        self._s3: Any | None = None
        self._client_lock = asyncio.Lock()

    def _client_kwargs(self) -> dict[str, Any]:
        return {
            "service_name": "s3",
            "endpoint_url": self.endpoint,
            "aws_access_key_id": self.access_key,
            "aws_secret_access_key": self.secret_key,
        }

    async def _client(self) -> Any:
        """Return the shared S3 client, opening it on first use."""
        if self._s3 is None:
            async with self._client_lock:
                # Re-check: a concurrent caller may have opened it while we waited.
                if self._s3 is None:
                    stack = AsyncExitStack()
                    self._s3 = await stack.enter_async_context(self._session.client(**self._client_kwargs()))
                    self._stack = stack
        return self._s3

    async def aclose(self) -> None:
        """Close the shared client. Safe to call repeatedly, and safe never to call.

        The close happens WHILE THE LOCK IS HELD, not after releasing it. Releasing
        first opens a window in which ``_s3`` is already ``None`` but the old stack is
        still unwinding: a concurrent operation sees ``None``, takes the free lock and
        builds a SECOND client — which is open at the moment ``aclose()`` returns, and
        which ``dependencies.close_clients()`` then orphans by dropping the singleton.
        Measured on the released-early version with a build that actually awaits:
        2 clients built, 1 still open when ``aclose()`` returned. Holding the lock: 0
        open at return.

        Residual, deliberately NOT closed here: an operation that reaches ``_client()``
        AFTER this returns still builds a fresh client that nothing will close. That is
        not a property of the lock — it is the lifespan calling ``close_clients()``
        without first draining ``subgraphx_service._background_tasks``, whose
        fire-and-forget uploads are exactly the racing callers. Making ``aclose()``
        terminal instead would fail those uploads and lose finished job results, so the
        fix belongs in shutdown ordering, not here.
        """
        async with self._client_lock:
            stack, self._stack, self._s3 = self._stack, None, None
            if stack is not None:
                await stack.aclose()

    async def upload(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        s3 = await self._client()
        await s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        logger.info("Uploaded to MinIO", key=key, size=len(data))

    async def download(self, key: str) -> bytes:
        s3 = await self._client()
        resp = await s3.get_object(Bucket=self.bucket, Key=key)
        body: bytes = await resp["Body"].read()
        return body

    async def presigned_url(self, key: str, expires: int = 3600) -> str:
        s3 = await self._client()
        url: str = await s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires,
        )
        return url

    async def delete(self, key: str) -> None:
        s3 = await self._client()
        await s3.delete_object(Bucket=self.bucket, Key=key)

    async def list_objects(self, prefix: str = "") -> list[str]:
        keys: list[str] = []
        s3 = await self._client()
        paginator = s3.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    async def ensure_bucket(self) -> None:
        s3 = await self._client()
        try:
            await s3.head_bucket(Bucket=self.bucket)
        except ClientError:
            await s3.create_bucket(Bucket=self.bucket)
            logger.info("Created MinIO bucket", bucket=self.bucket)
