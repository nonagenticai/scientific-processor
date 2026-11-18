"""Enable Postgres RLS on the tenant tables (HIPAA G4 backstop).

The provable tenant-isolation backstop beneath the application-layer
``WHERE tenant_id = …`` predicates: even if a predicate were dropped, Postgres
returns zero foreign-tenant rows and rejects foreign-tenant writes.

Ships INERT. While the service connects as the table owner the owner bypasses
RLS (ENABLE-not-FORCE) → no behaviour change. The policy binds only at the
atomic cutover: the runtime switches to a separate non-owner restricted role
**and** ``TENANT_RLS_MODE=enforce`` together (a reversible per-service gitops
flip).

Tables (both carry ``tenant_id`` ``varchar`` at THIS point in history →
text-to-text predicate, no cast):
  * ``jobs.tenant_id`` is NULLABLE → "global-or-own" (NULL rows are shared job
    metadata, readable by any tenant; writes still GUC-scoped);
  * ``verification_history.tenant_id`` is NOT NULL → strict equality.

⚠️ WHY THE DDL IS NOW WRITTEN OUT LITERALLY (changed 2026-08-28).

This revision used to render its DDL from
``kag_common.database.rls.upgrade_statements(Base.metadata)`` at runtime, for an
anti-drift reason. That coupling is a bug once the model changes: ``005`` flips
``tenant_id`` to a real ``uuid`` column, and ``rls._predicate`` keys its ``::uuid``
cast off ``isinstance(col.type, String)`` — so a metadata-rendered ``003`` would
start emitting ``tenant_id = NULLIF(...)::uuid`` against a column that is still
``character varying`` at revision 003, and ``alembic upgrade head`` from an empty
database would abort with ``operator does not exist: character varying = uuid``.

A migration is immutable history: it must describe the schema as it was **at this
revision**, not as the current model happens to look. These statements are
byte-identical to what the helper rendered for the varchar columns, so already-
migrated databases (prod) are unchanged. The live/`create_all` source of truth is
still the shared helper — ``tests/unit/test_rls.py`` pins it against the model.

Revision ID: 003
Revises: 002
Create Date: 2026-06-21
"""

from collections.abc import Sequence

from alembic import op

revision: str = "003"
down_revision: str = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Frozen: the GUC is transaction-scoped and always TEXT. ``current_setting(…, true)``
# is missing-ok; ``NULLIF(…, '')`` folds the empty string PgBouncer leaves behind back
# to NULL so the predicate is simply false (0 rows), never a cast error.
_GUC = "NULLIF(current_setting('app.current_tenant', true), '')"
_PREDICATE = f"tenant_id = {_GUC}"  # varchar column → text-to-text, no cast.

_TABLES = ("jobs", "verification_history")
_ALL_POLICY_NAMES = (
    "tenant_isolation",
    "tenant_isolation_read",
    "tenant_isolation_ins",
    "tenant_isolation_upd",
    "tenant_isolation_del",
)

_UPGRADE: tuple[str, ...] = (
    # jobs: NULLABLE tenant_id → global-or-own (NULL rows shared, writes scoped).
    "ALTER TABLE jobs ENABLE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_isolation_read ON jobs",
    "DROP POLICY IF EXISTS tenant_isolation_ins ON jobs",
    "DROP POLICY IF EXISTS tenant_isolation_upd ON jobs",
    "DROP POLICY IF EXISTS tenant_isolation_del ON jobs",
    f"CREATE POLICY tenant_isolation_read ON jobs FOR SELECT USING (tenant_id IS NULL OR {_PREDICATE})",
    f"CREATE POLICY tenant_isolation_ins ON jobs FOR INSERT WITH CHECK ({_PREDICATE})",
    f"CREATE POLICY tenant_isolation_upd ON jobs FOR UPDATE USING ({_PREDICATE}) WITH CHECK ({_PREDICATE})",
    f"CREATE POLICY tenant_isolation_del ON jobs FOR DELETE USING ({_PREDICATE})",
    # verification_history: NOT NULL tenant_id → strict equality.
    "ALTER TABLE verification_history ENABLE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_isolation ON verification_history",
    f"CREATE POLICY tenant_isolation ON verification_history USING ({_PREDICATE}) WITH CHECK ({_PREDICATE})",
)


def upgrade() -> None:
    for stmt in _UPGRADE:
        op.execute(stmt)


def downgrade() -> None:
    for table in _TABLES:
        for policy in _ALL_POLICY_NAMES:
            op.execute(f"DROP POLICY IF EXISTS {policy} ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
