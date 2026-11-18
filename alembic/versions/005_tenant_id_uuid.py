"""Fleet ruling: ``tenant_id`` is a real ``uuid``, never text.

Both tenant-scoped tables carried ``tenant_id`` as ``character varying(100)``.
Measured live on the production database 2026-08-27: ``jobs`` and
``verification_history`` are **both empty (0 rows)**, so the cast is trivially
safe here — there is no name→uuid backfill to do (unlike the w2v content_service
sibling, whose ``n8n-dev`` rows had to be mapped first).

The guard in step 2 is therefore not load-bearing for *this* data — it is
load-bearing for every database created from scratch afterwards (CI, dev, a
re-provisioned prod). It stays, and it stays fail-loud: a filtered
``ALTER … WHERE tenant_id ~ …`` would quietly leave or drop unconvertible rows,
and silent data loss on a tenant key is worse than a failed migration.

Order is load-bearing:

1. **Drop the RLS policies first.** Postgres refuses ``ALTER TABLE … ALTER COLUMN
   … TYPE`` on a column referenced by a policy predicate (``cannot alter type of
   a column used in a policy definition``). ``003`` compares ``tenant_id`` to the
   ``app.current_tenant`` GUC as **text**; leaving the policies in place aborts
   the migration outright.
2. **Guard, fail loud** on any non-NULL value that is not a uuid.
3. **Cast** to ``uuid``. Nullability is preserved by ``ALTER COLUMN TYPE``;
   ``idx_jobs_tenant_status`` and ``idx_vh_entity`` are rebuilt automatically.
4. **Re-create the policies with ``::uuid`` on the GUC.** The GUC is always text;
   against a ``uuid`` column the old text predicate raises ``operator does not
   exist: uuid = text``. Getting this wrong returns ZERO rows (not an error) from
   the pod once ``TENANT_RLS_MODE=enforce``, so it happens in the SAME migration.

⚠️ **THE TWO TABLES HAVE DIFFERENT POLICY SHAPES AND THAT DIFFERENCE IS THE POINT.**
``kag_common.database.rls`` renders a *strict* single ``tenant_isolation`` policy
for a NOT NULL ``tenant_id`` and a *four*-policy "global-or-own" set
(``_read``/``_ins``/``_upd``/``_del``) for a NULLABLE one:

  * ``verification_history.tenant_id`` is NOT NULL → one strict policy.
  * ``jobs.tenant_id`` is NULLABLE → four policies, and the SELECT policy carries
    the ``tenant_id IS NULL OR …`` clause that makes NULL-tenant rows shared job
    metadata readable by **every** tenant. Flatten that to the strict shape and
    those rows silently vanish for everyone.

The drop list covers all five policy names on both tables so it is exhaustive
regardless of which shape was applied; the re-create reproduces each table's own
shape verbatim.

Written out literally rather than rendered from
``rls.upgrade_statements(Base.metadata)`` on purpose — a migration is immutable
history and must not start emitting DDL for tables that do not exist yet at this
point in the chain. It matches what the helper renders for a non-``String``
column type (``_predicate`` keys the cast off ``isinstance(col.type, String)``;
``PgUUID`` is not a ``String``), so migration and ``create_all`` still agree —
``tests/unit/test_rls.py`` pins that.

No PL/pgSQL function or view references ``tenant_id`` (verified live on the
production database 2026-08-27), so there is no function/view step.

Downgrade restores ``character varying(100)`` and the text predicate, including
the 4-vs-1 policy split.

Revision ID: 005
Revises: 004
Create Date: 2026-08-28
"""

from collections.abc import Sequence

from alembic import op

revision: str = "005"
down_revision: str = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("jobs", "verification_history")

# Every policy name kag_common may have created (strict + the nullable
# global-or-own variants), so the drop is exhaustive on both tables.
_ALL_POLICY_NAMES = (
    "tenant_isolation",
    "tenant_isolation_read",
    "tenant_isolation_ins",
    "tenant_isolation_upd",
    "tenant_isolation_del",
)

# The GUC is transaction-scoped and always TEXT. NULLIF folds the empty string
# PgBouncer leaves behind back to NULL → predicate false (0 rows), never ''::uuid.
_GUC = "NULLIF(current_setting('app.current_tenant', true), '')"
_PREDICATE_UUID = f"tenant_id = {_GUC}::uuid"
_PREDICATE_TEXT = f"tenant_id = {_GUC}"

_UUID_RE = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"


def _drop_policies() -> None:
    for table in _TABLES:
        for policy in _ALL_POLICY_NAMES:
            op.execute(f"DROP POLICY IF EXISTS {policy} ON {table}")


def _create_policies(predicate: str) -> None:
    """Re-create each table's OWN shape. ENABLE, never FORCE (owner must bypass)."""
    # jobs: NULLABLE tenant_id → global-or-own. The IS NULL clause on SELECT is
    # what keeps shared, tenant-less job metadata readable by every tenant.
    op.execute("ALTER TABLE jobs ENABLE ROW LEVEL SECURITY")
    op.execute(f"CREATE POLICY tenant_isolation_read ON jobs FOR SELECT USING (tenant_id IS NULL OR {predicate})")
    op.execute(f"CREATE POLICY tenant_isolation_ins ON jobs FOR INSERT WITH CHECK ({predicate})")
    op.execute(f"CREATE POLICY tenant_isolation_upd ON jobs FOR UPDATE USING ({predicate}) WITH CHECK ({predicate})")
    op.execute(f"CREATE POLICY tenant_isolation_del ON jobs FOR DELETE USING ({predicate})")

    # verification_history: NOT NULL tenant_id → strict equality, no global clause.
    op.execute("ALTER TABLE verification_history ENABLE ROW LEVEL SECURITY")
    op.execute(f"CREATE POLICY tenant_isolation ON verification_history USING ({predicate}) WITH CHECK ({predicate})")


def _guard_non_uuid() -> None:
    for table in _TABLES:
        op.execute(
            f"""
            DO $$
            DECLARE
                bad_count bigint;
                sample text;
            BEGIN
                SELECT count(*), min(tenant_id)
                  INTO bad_count, sample
                  FROM {table}
                 WHERE tenant_id IS NOT NULL
                   AND tenant_id !~ '{_UUID_RE}';
                IF bad_count > 0 THEN
                    RAISE EXCEPTION
                        '{table}.tenant_id: % row(s) are not a uuid and cannot be cast '
                        '(first offender: %). Triage these by hand - map each tenant NAME '
                        'to its uuid in licenses.tenants and backfill before re-running. '
                        'Refusing to drop or skip rows.',
                        bad_count, sample;
                END IF;
            END $$;
            """  # noqa: S608
        )


def upgrade() -> None:
    # 1. Policies reference tenant_id → they block the type change.
    _drop_policies()
    # 2. Fail loud on anything non-uuid. No silent skip, no row loss.
    _guard_non_uuid()
    # 3. The cast. Preserves nullability; rebuilds the tenant_id indexes.
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN tenant_id TYPE uuid USING tenant_id::uuid")
    # 4. Re-create RLS with the GUC cast to uuid, or the pod reads 0 rows.
    _create_policies(_PREDICATE_UUID)


def downgrade() -> None:
    _drop_policies()
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN tenant_id TYPE character varying(100) USING tenant_id::text")
    # Back to the text-to-text predicate that matches a varchar column,
    # restoring the same 4-vs-1 split.
    _create_policies(_PREDICATE_TEXT)
