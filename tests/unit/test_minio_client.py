"""Unit tests for MinIOClient with mocked aioboto3."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from clients.minio_client import MinIOClient

pytestmark = pytest.mark.unit


@pytest.fixture
def minio():
    return MinIOClient(
        endpoint="http://minio:9000",
        access_key="testkey",
        secret_key="testsecret",
        bucket="test-bucket",
    )


def _mock_s3_client():
    """Create a mock S3 client context manager.

    NB its ``__aenter__``/``__aexit__`` are ``AsyncMock``s: awaiting a coroutine that
    never itself awaits does NOT yield to the event loop, so this mock can never
    interleave with a concurrent task. Fine for shape assertions; useless for anything
    about concurrency — use :func:`_awaiting_s3_client` there.
    """
    mock_s3 = AsyncMock()
    mock_s3.__aenter__ = AsyncMock(return_value=mock_s3)
    mock_s3.__aexit__ = AsyncMock(return_value=False)
    return mock_s3


def _awaiting_s3_client(live: list | None = None, **_kwargs):
    """A mock S3 client whose ``__aenter__``/``__aexit__`` actually SUSPEND.

    This is what the real ``aioboto3`` build does — ``Session.client(...)`` returns an
    async context manager whose ``__aenter__`` is a genuine coroutine. Without a
    suspension point, ``asyncio.gather`` runs each ``upload`` to completion before
    starting the next one, so no test can ever observe a concurrency bug.

    ``live`` (if given) tracks which clients are currently open: appended on enter,
    removed on exit.
    """
    mock_s3 = AsyncMock()

    async def _aenter(*_a):
        await asyncio.sleep(0)
        if live is not None:
            live.append(mock_s3)
        return mock_s3

    async def _aexit(*_a):
        # Tearing down a connection pool suspends too.
        await asyncio.sleep(0.02)
        if live is not None and mock_s3 in live:
            live.remove(mock_s3)
        return False

    mock_s3.__aenter__ = _aenter
    mock_s3.__aexit__ = _aexit
    return mock_s3


class TestMinIOClient:
    def test_init_adds_http_prefix(self):
        c = MinIOClient(endpoint="minio:9000", access_key="a", secret_key="s", bucket="b")
        assert c.endpoint == "http://minio:9000"

    def test_init_preserves_http(self):
        c = MinIOClient(endpoint="http://minio:9000", access_key="a", secret_key="s", bucket="b")
        assert c.endpoint == "http://minio:9000"

    def test_init_preserves_https(self):
        c = MinIOClient(endpoint="https://minio:9000", access_key="a", secret_key="s", bucket="b")
        assert c.endpoint == "https://minio:9000"

    async def test_upload(self, minio):
        mock_s3 = _mock_s3_client()
        with patch.object(minio._session, "client", return_value=mock_s3):
            await minio.upload("key/file.json", b'{"data": 1}', "application/json")
        mock_s3.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="key/file.json",
            Body=b'{"data": 1}',
            ContentType="application/json",
        )

    async def test_download(self, minio):
        mock_s3 = _mock_s3_client()
        body_mock = AsyncMock()
        body_mock.read.return_value = b"file-contents"
        mock_s3.get_object.return_value = {"Body": body_mock}
        with patch.object(minio._session, "client", return_value=mock_s3):
            data = await minio.download("key/file.json")
        assert data == b"file-contents"

    async def test_presigned_url(self, minio):
        mock_s3 = _mock_s3_client()
        mock_s3.generate_presigned_url.return_value = "http://minio:9000/presigned"
        with patch.object(minio._session, "client", return_value=mock_s3):
            url = await minio.presigned_url("key/file.json", expires=7200)
        assert url == "http://minio:9000/presigned"
        mock_s3.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": "test-bucket", "Key": "key/file.json"},
            ExpiresIn=7200,
        )

    async def test_delete(self, minio):
        mock_s3 = _mock_s3_client()
        with patch.object(minio._session, "client", return_value=mock_s3):
            await minio.delete("key/file.json")
        mock_s3.delete_object.assert_called_once_with(Bucket="test-bucket", Key="key/file.json")

    async def test_list_objects(self, minio):
        mock_s3 = _mock_s3_client()

        async def mock_pages(**kwargs):  # noqa: N803
            yield {"Contents": [{"Key": "prefix/a.json"}, {"Key": "prefix/b.json"}]}
            yield {"Contents": [{"Key": "prefix/c.json"}]}

        mock_paginator = MagicMock()
        mock_paginator.paginate = mock_pages
        # get_paginator is sync on the mock — returns a plain object
        mock_s3.get_paginator = MagicMock(return_value=mock_paginator)

        with patch.object(minio._session, "client", return_value=mock_s3):
            keys = await minio.list_objects("prefix/")
        assert keys == ["prefix/a.json", "prefix/b.json", "prefix/c.json"]

    async def test_list_objects_empty(self, minio):
        mock_s3 = _mock_s3_client()

        async def mock_pages(**kwargs):  # noqa: N803
            yield {"Contents": []}

        mock_paginator = MagicMock()
        mock_paginator.paginate = mock_pages
        mock_s3.get_paginator = MagicMock(return_value=mock_paginator)

        with patch.object(minio._session, "client", return_value=mock_s3):
            keys = await minio.list_objects("empty/")
        assert keys == []

    async def test_ensure_bucket_exists(self, minio):
        mock_s3 = _mock_s3_client()
        with patch.object(minio._session, "client", return_value=mock_s3):
            await minio.ensure_bucket()
        mock_s3.head_bucket.assert_called_once_with(Bucket="test-bucket")
        mock_s3.create_bucket.assert_not_called()

    async def test_ensure_bucket_creates_if_missing(self, minio):
        mock_s3 = _mock_s3_client()
        mock_s3.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "Not found"}}, "HeadBucket"
        )
        with patch.object(minio._session, "client", return_value=mock_s3):
            await minio.ensure_bucket()
        mock_s3.create_bucket.assert_called_once_with(Bucket="test-bucket")


class TestClientReuse:
    """One S3 client (and one connection pool) per instance, not one per operation.

    Building a botocore client is synchronous work — it loads/parses the S3 service
    model and builds the signer stack — and it happens inside ``__aenter__``, on the
    event loop thread with nothing to yield at (~172 ms for the first client in a
    process, ~2-3 ms thereafter). Opening one per call put that blocking work on the
    hot path of every upload/download/delete.
    """

    async def test_client_is_built_once_across_operations(self, minio):
        mock_s3 = _mock_s3_client()
        with patch.object(minio._session, "client", return_value=mock_s3) as factory:
            await minio.upload("a.json", b"1")
            await minio.upload("b.json", b"2")
            await minio.delete("a.json")
        assert factory.call_count == 1, (
            f"built {factory.call_count} clients for 3 operations — each build is blocking work on the event loop"
        )
        assert mock_s3.__aenter__.await_count == 1

    async def test_concurrent_first_use_builds_exactly_one_client(self, minio):
        """THE test for the lock — and it used to prove nothing.

        Deleting the ``asyncio.Lock`` from ``_client()`` entirely left all 16 minio
        tests green, this one included, because ``_mock_s3_client()`` builds with
        ``AsyncMock.__aenter__``: awaiting a coroutine that never itself awaits does
        not yield to the loop, so ``gather`` could never interleave inside the
        critical section. The real build DOES suspend (``Session.client(...)``
        returns an async context manager whose ``__aenter__`` is a genuine
        coroutine), so production had 20 concurrent uploads building 20 clients —
        20 connection pools, 19 of them orphaned — while the suite stayed green.

        ``_awaiting_s3_client`` restores the suspension, which is what makes the
        lock (and its inner double-check) load-bearing here.
        """
        with patch.object(minio._session, "client", side_effect=_awaiting_s3_client) as factory:
            await asyncio.gather(*(minio.upload(f"{i}.json", b"x") for i in range(20)))
        assert factory.call_count == 1, (
            f"20 concurrent uploads built {factory.call_count} clients — "
            "the lock or its inner double-check is not doing its job"
        )

    async def test_aclose_releases_the_client_and_a_second_call_is_a_no_op(self, minio):
        """The old version of this test did not test its own name.

        It asserted only ``__aexit__.await_count == 1``, which ``AsyncExitStack``
        guarantees on its own — a second ``stack.aclose()`` unwinds an already-empty
        stack and touches nothing. So an ``aclose()`` that forgot to release
        ``self._stack`` passed. Assert the state contract instead, and make a second
        touch of the closed stack a hard failure rather than an invisible no-op.
        """
        mock_s3 = _mock_s3_client()
        with patch.object(minio._session, "client", return_value=mock_s3):
            await minio.delete("a.json")
            assert minio._s3 is not None
            assert minio._stack is not None
            stack = minio._stack

            await minio.aclose()

            # Released, not merely closed: nothing can hand the dead client out again.
            assert minio._s3 is None
            assert minio._stack is None
            assert mock_s3.__aexit__.await_count == 1

            with patch.object(stack, "aclose", side_effect=AssertionError("closed stack touched twice")) as spy:
                await minio.aclose()
                assert spy.call_count == 0
        assert mock_s3.__aexit__.await_count == 1

    async def test_no_client_is_left_open_when_aclose_returns(self, minio):
        """Releasing the lock before ``await stack.aclose()`` resurrects the client.

        The shipped version did::

            async with self._client_lock:
                stack, self._stack, self._s3 = self._stack, None, None
            if stack is not None:
                await stack.aclose()      # <- lock already released

        In that window ``_s3`` is ``None`` and the lock is free, so a concurrent
        operation builds a SECOND client — open at the instant ``aclose()`` returns,
        and immediately orphaned by ``dependencies.close_clients()`` dropping the
        singleton. SubgraphX ``_spawn``s fire-and-forget uploads that the lifespan
        never awaits, so this races every SIGTERM.

        Reproduced: 2 clients built, 1 still open when ``aclose()`` returned.
        """
        live: list[object] = []
        with patch.object(minio._session, "client", side_effect=lambda **kw: _awaiting_s3_client(live)):
            await minio.upload("warm.json", b"1")
            assert len(live) == 1

            inflight = asyncio.create_task(minio.upload("inflight.json", b"2"))
            # Deliberately NO yield here: aclose() runs to its first suspension point
            # (inside stack.aclose()) before the task gets scheduled, which is exactly
            # the window — the task reaches _client() with _s3 already None.
            await minio.aclose()
            open_at_return = list(live)

            await inflight
        assert open_at_return == [], (
            f"{len(open_at_return)} client(s) still open when aclose() returned — "
            "the close must happen while the lock is held"
        )

    async def test_aclose_without_any_operation_is_safe(self, minio):
        await minio.aclose()

    async def test_client_reopens_after_aclose(self, minio):
        mock_s3 = _mock_s3_client()
        with patch.object(minio._session, "client", return_value=mock_s3) as factory:
            await minio.delete("a.json")
            await minio.aclose()
            await minio.delete("b.json")
        assert factory.call_count == 2


class TestShutdownReleasesTheSingleton:
    """``dependencies.close_clients()`` must actually release the MinIO client.

    Deleting the ``await _minio_client.aclose()`` line from ``close_clients()`` left
    the whole 805-test suite green: nothing covered it. Now that ``MinIOClient`` holds
    a long-lived S3 client and its connection pool for the life of the process, that
    line is the only thing that closes it at shutdown.
    """

    async def test_close_clients_acloses_and_drops_the_minio_singleton(self):
        import dependencies

        previous = dependencies._minio_client
        minio = MinIOClient(endpoint="http://minio:9000", access_key="a", secret_key="s", bucket="b")
        mock_s3 = _mock_s3_client()
        try:
            with patch.object(minio._session, "client", return_value=mock_s3):
                await minio.upload("a.json", b"1")
                assert minio._s3 is not None

                dependencies._minio_client = minio
                await dependencies.close_clients()

            assert mock_s3.__aexit__.await_count == 1, "close_clients() did not close the S3 client"
            assert minio._s3 is None
            assert dependencies._minio_client is None
        finally:
            dependencies._minio_client = previous

    async def test_close_clients_is_safe_with_no_minio_client(self):
        import dependencies

        previous = dependencies._minio_client
        dependencies._minio_client = None
        try:
            await dependencies.close_clients()
            assert dependencies._minio_client is None
        finally:
            dependencies._minio_client = previous
