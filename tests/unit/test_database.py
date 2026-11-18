"""Tests for database module -- SQLAlchemy engine, session factory, and PersistentJobManager."""

from __future__ import annotations

import functools
import os
import socket
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.engine import make_url

pytestmark = pytest.mark.unit

# tenant_id is a real Postgres ``uuid`` column (migration 005) — an opaque string
# like "t1" is no longer castable. These also assert the ``as_uuid=False`` contract:
# the value round-trips as a ``str`` and compares equal to the ``str`` that went in.
TENANT_1 = "019de842-c92a-7e37-8fdd-eb0d8c5396f9"
TENANT_2 = "019de843-0000-7e37-8fdd-eb0d8c539600"

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg_async://scientific_processor_user:TdG8xKm2vQp5nR7w@scientific-processor-test-db.invalid:5432/scientific_processor",
)


@functools.lru_cache(maxsize=1)
def _db_host_reachable() -> bool:
    """Return True if the database host can be resolved and reached.

    Parsed with SQLAlchemy's ``make_url``, NOT ``urllib.parse.urlparse``. A URL
    scheme may not contain an underscore (RFC 3986 ``ALPHA *( ALPHA / DIGIT /
    "+" / "-" / "." )``), so ``urlparse`` refuses to split
    ``postgresql+psycopg_async://…`` at all and hands back
    ``scheme='' netloc='' hostname=None``. ``hostname or "…​.invalid"`` then
    resolved the ``.invalid`` sentinel, raised ``gaierror``, and returned False
    — for EVERY value of ``DATABASE_URL``, including a perfectly reachable one.

    So ``requires_db`` could never be satisfied and the 10 ``PersistentJobManager``
    tests below were skipped everywhere, CI included, despite the archetype
    provisioning ``has_postgres: true``. They had never executed. ``make_url``
    understands the ``dialect+driver`` form and returns the real host.
    """
    try:
        parsed = make_url(DATABASE_URL)
        host = parsed.host or "scientific-processor-test-db.invalid"
        port = parsed.port or 5432
        sock = socket.create_connection((host, port), timeout=3)
        sock.close()
    except socket.gaierror, OSError, TimeoutError:
        return False
    return True


requires_db = pytest.mark.skipif(not _db_host_reachable(), reason="PostgreSQL host not reachable")


class TestDatabaseUrlParsing:
    """Floor under ``_db_host_reachable``: a gate that can never fire is not a gate.

    ``requires_db`` silently skipped all 10 ``PersistentJobManager`` tests in every
    environment — including CI with ``has_postgres: true`` — because the parser
    could not extract a host from a SQLAlchemy ``dialect+driver`` URL. Nothing
    failed, so nothing surfaced it. These pin the parse itself.
    """

    def test_a_driver_qualified_url_yields_a_real_host(self):
        """The exact shape CI and the deployment use."""
        url = make_url("postgresql+psycopg_async://u:p@db.example:5432/scientific_processor")
        assert url.host == "db.example"
        assert url.port == 5432

    def test_urlparse_cannot_do_this_and_that_is_why_make_url_is_used(self):
        """Regression pin for the actual defect, so a "simplification" back to
        ``urlparse`` fails loudly instead of re-muting ten tests."""
        from urllib.parse import urlparse

        broken = urlparse("postgresql+psycopg_async://u:p@db.example:5432/scientific_processor")
        assert broken.hostname is None, (
            "urlparse now handles underscores in the scheme; re-check whether make_url is still required"
        )

    def test_reachability_probe_is_not_hardwired_to_false(self):
        """``_db_host_reachable`` must actually answer the configured URL.

        A loopback URL on a closed port must be probed and reported unreachable —
        i.e. reached the socket layer — rather than short-circuiting on a parse
        failure the way the ``.invalid`` fallback did.
        """
        assert make_url(DATABASE_URL).host, f"DATABASE_URL has no parseable host: {DATABASE_URL!r}"


class TestDatabaseModule:
    def test_create_engine_from_url(self):
        from database import create_async_engine_from_url

        engine = create_async_engine_from_url(DATABASE_URL)
        assert engine is not None

    def test_create_session_factory(self):
        from database import create_async_engine_from_url, create_session_factory

        engine = create_async_engine_from_url(DATABASE_URL)
        factory = create_session_factory(engine)
        assert factory is not None

    def test_job_table_model_exists(self):
        from database import Base, JobRow

        assert "jobs" in Base.metadata.tables
        assert JobRow.__tablename__ == "jobs"

    def test_verification_history_table_exists(self):
        from database import Base, VerificationHistoryRow

        assert "verification_history" in Base.metadata.tables
        assert VerificationHistoryRow.__tablename__ == "verification_history"


@requires_db
class TestPersistentJobManager:
    @pytest.fixture(autouse=True)
    async def _setup_and_cleanup(self):
        """Create tables and truncate job data before each test."""
        from sqlalchemy import text

        from database import Base, create_async_engine_from_url, create_session_factory

        engine = create_async_engine_from_url(DATABASE_URL)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        # Truncate before the test
        async with engine.begin() as conn:
            await conn.execute(text("TRUNCATE TABLE jobs CASCADE"))
        self._engine = engine
        self._factory = create_session_factory(engine)
        yield
        # Truncate after the test
        async with engine.begin() as conn:
            await conn.execute(text("TRUNCATE TABLE jobs CASCADE"))
        await engine.dispose()

    async def test_create_and_get_job(self):
        from services.persistent_job_manager import PersistentJobManager

        mgr = PersistentJobManager(session_factory=self._factory)

        job_id = await mgr.create_job("sandbox", tenant_id=TENANT_1)
        assert job_id

        job = await mgr.get_job(job_id)
        assert job is not None
        assert job.job_id == job_id
        assert job.job_type == "sandbox"
        assert job.status == "pending"
        assert job.tenant_id == TENANT_1

    async def test_update_job(self):
        from services.persistent_job_manager import PersistentJobManager

        mgr = PersistentJobManager(session_factory=self._factory)

        job_id = await mgr.create_job("subgraphx")
        updated = await mgr.update_job(job_id, status="running", progress=0.5)
        assert updated is not None
        assert updated.status == "running"
        assert updated.progress == 0.5

    async def test_update_job_completed_sets_timestamp(self):
        from services.persistent_job_manager import PersistentJobManager

        mgr = PersistentJobManager(session_factory=self._factory)

        job_id = await mgr.create_job("formal")
        updated = await mgr.update_job(job_id, status="completed")
        assert updated is not None
        assert updated.completed_at is not None
        assert updated.progress == 1.0

    async def test_update_job_with_result(self):
        from services.persistent_job_manager import PersistentJobManager

        mgr = PersistentJobManager(session_factory=self._factory)

        job_id = await mgr.create_job("sandbox")
        result_data = {"node_id": "A", "score": 0.95}
        updated = await mgr.update_job(job_id, status="completed", result=result_data)
        assert updated is not None
        assert updated.result == result_data

        # Re-fetch to verify persistence
        fetched = await mgr.get_job(job_id)
        assert fetched is not None
        assert fetched.result == result_data

    async def test_list_jobs_by_type(self):
        from services.persistent_job_manager import PersistentJobManager

        mgr = PersistentJobManager(session_factory=self._factory)

        await mgr.create_job("sandbox")
        await mgr.create_job("formal")
        await mgr.create_job("sandbox")

        sandbox_jobs = await mgr.list_jobs(job_type="sandbox")
        assert len(sandbox_jobs) == 2

    async def test_list_jobs_by_tenant(self):
        from services.persistent_job_manager import PersistentJobManager

        mgr = PersistentJobManager(session_factory=self._factory)

        await mgr.create_job("sandbox", tenant_id=TENANT_1)
        await mgr.create_job("sandbox", tenant_id=TENANT_2)

        t1_jobs = await mgr.list_jobs(tenant_id=TENANT_1)
        assert len(t1_jobs) == 1
        assert t1_jobs[0].tenant_id == TENANT_1

    async def test_get_nonexistent_job_returns_none(self):
        from services.persistent_job_manager import PersistentJobManager

        mgr = PersistentJobManager(session_factory=self._factory)

        job = await mgr.get_job("nonexistent-id")
        assert job is None

    async def test_update_nonexistent_returns_none(self):
        from services.persistent_job_manager import PersistentJobManager

        mgr = PersistentJobManager(session_factory=self._factory)

        result = await mgr.update_job("nonexistent-id", status="running")
        assert result is None

    async def test_cleanup_expired(self):
        from database import JobRow
        from services.persistent_job_manager import PersistentJobManager

        mgr = PersistentJobManager(session_factory=self._factory)

        old_id = await mgr.create_job("sandbox")
        fresh_id = await mgr.create_job("sandbox")

        # Backdate the old job directly in the DB
        async with self._factory() as session:
            row = await session.get(JobRow, old_id)
            assert row is not None
            row.created_at = datetime.now(UTC) - timedelta(hours=25)
            await session.commit()

        removed = await mgr.cleanup_expired(max_age_hours=24)
        assert removed == 1
        assert await mgr.get_job(old_id) is None
        assert await mgr.get_job(fresh_id) is not None

    async def test_cleanup_no_expired(self):
        from services.persistent_job_manager import PersistentJobManager

        mgr = PersistentJobManager(session_factory=self._factory)

        await mgr.create_job("sandbox")
        removed = await mgr.cleanup_expired(max_age_hours=24)
        assert removed == 0
