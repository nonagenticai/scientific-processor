"""Grant the runtime (non-owner) role DML on the tenant tables.

WHY THIS EXISTS — a live outage, measured 2026-08-25.

``003_rls`` shipped RLS "inert", and its docstring states the cutover condition
exactly: the policy binds only when *"the runtime switches to a separate
non-owner restricted role **and** ``TENANT_RLS_MODE=enforce`` together"*. Both
halves of that cutover shipped in gitops — the runtime container's
``DATABASE_URL`` is fed from the secret key ``RUNTIME_DATABASE_URL``
(``scientific_processor_app``, non-owner) while the migrate initContainer keeps
``DATABASE_URL`` (``scientific_processor_user``, the owner), and
``TENANT_RLS_MODE=enforce`` is set on the deployment.

The GRANT that has to accompany that switch was never written. A non-owner role
with no table privileges cannot read its own service's tables, so the moment the
database was re-provisioned (2026-08-25 04:35, dropping the grants the owner had
implicitly relied on) the service was one query away from failing. Measured on
worker4 immediately before this migration:

    SET ROLE scientific_processor_app; SELECT count(*) FROM public.jobs;
    ERROR:  permission denied for table jobs

    SELECT grantee FROM information_schema.role_table_grants
     WHERE table_schema='public';
    -> scientific_processor_user only. pg_default_acl: 0 rows.

⚠️  WHY GRANTING IS SAFE HERE, AND WHY THAT NEEDED CHECKING FIRST. Handing a
non-owner DML on an RLS-enabled table normally trades a loud failure for a
silent one: the policies are fail-closed on a GUC
(``NULLIF(current_setting('app.current_tenant', true), '')``), so a role that
never binds it reads **zero rows** instead of raising. That would be strictly
worse than the permission error. It is safe here because the other end is
already wired: ``dependencies.set_db_session_factory`` calls
``register_tenant_guc(engine, …)`` on the single session factory, binding
``app.current_tenant`` on every transaction, and ``TENANT_RLS_MODE`` is
``enforce`` on the live deployment. Owner-run migrations are unaffected either
way (RLS is ENABLE, not FORCE, so the owner bypasses it).

``alembic_version`` is deliberately NOT granted: the runtime must never be able
to mutate migration state.

Revision ID: 004
Revises: 003
Create Date: 2026-08-25
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from database import Base

revision: str = "004"
down_revision: str = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The runtime role, as carried by the RUNTIME_DATABASE_URL secret key.
APP_ROLE = "scientific_processor_app"

_DML = "SELECT, INSERT, UPDATE, DELETE"


# Derived from the model metadata rather than hand-listed, for the same
# anti-drift reason 003 renders its DDL from kag_common.database.rls: a table
# added later must not silently miss its grant. The ALTER DEFAULT PRIVILEGES
# below is the belt to this braces — it covers tables that do not exist yet.
def _tables() -> list[str]:
    return sorted(Base.metadata.tables)


def _role_exists() -> bool:
    """Is the runtime role present on THIS database?

    Decided in Python, deliberately, rather than in a SQL ``DO`` block. The first version of
    this migration wrapped the grants in ``IF EXISTS (SELECT 1 FROM pg_roles …) … ELSE RAISE
    NOTICE``, on the reasonable-sounding theory that a skipped production run would be visible
    in the migration log. **It would not have been.** psycopg surfaces a Postgres NOTICE
    through ``logger.info`` on ``sqlalchemy.dialects.postgresql``, and ``alembic.ini`` sets
    ``[logger_root] level = WARNING`` with no entry for that qualname — so the notice is
    swallowed. In the exact scenario this migration exists to fix, it would have reported
    success, granted nothing, and left the outage in place: a green run in the wrong state,
    which is the failure mode the grants themselves were written to end.

    A Python ``print`` goes to stdout unconditionally and lands in the initContainer log no
    matter how logging is configured.
    """
    return bool(op.get_bind().execute(sa.text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": APP_ROLE}).scalar())


def _assert_granted() -> None:
    """Fail the migration if the grants did not actually land.

    The grants are the entire point of this revision, so 'ran without raising' is not good
    enough evidence that they exist — that is the same unchecked assumption that let the
    original outage through. Read them back.
    """
    missing = [
        t
        for t in _tables()
        if not op.get_bind()
        .execute(
            sa.text(
                "SELECT 1 FROM information_schema.role_table_grants "
                "WHERE table_schema='public' AND table_name=:t "
                "AND grantee=:r AND privilege_type='SELECT'"
            ),
            {"t": t, "r": APP_ROLE},
        )
        .scalar()
    ]
    if missing:
        raise RuntimeError(
            f"GRANT verification FAILED: {APP_ROLE} still lacks SELECT on {missing}. "
            "The runtime connects as this role and would get 'permission denied'."
        )


def upgrade() -> None:
    if not _role_exists():
        # Loud on stdout, never a silent pass. The role is provisioned out of band: it exists
        # on worker4 and does NOT exist in CI's per-job ephemeral postgres:18, so skipping is
        # correct there — but a skip in PRODUCTION is an outage waiting, and must be readable.
        print(
            f"::warning::SKIPPED all grants — role {APP_ROLE} does not exist on this database. "
            "Expected in CI. NOT expected in production: the runtime connects as this role and "
            "will get 'permission denied' on every query."
        )
        return

    op.execute(f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}")
    for t in _tables():
        op.execute(f"GRANT {_DML} ON public.{t} TO {APP_ROLE}")
    # Future tables created by THIS role (the migration runner is the owner) get the same
    # grant automatically. Without this the outage recurs the next time a table is added.
    #
    # ⚠️ KNOWN LIMIT, stated rather than implied: default ACLs key on the CREATING role. If a
    # future re-provision creates the tables as a SUPERUSER and then `ALTER TABLE … OWNER TO`
    # the owner, these defaults do not apply and the grants above are what save it — which is
    # why both halves are here, and why this migration runs on every deploy rather than once.
    op.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT {_DML} ON TABLES TO {APP_ROLE}")
    op.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO {APP_ROLE}")
    _assert_granted()
    print(f"granted {_DML} on {_tables()} to {APP_ROLE}; verified by read-back")


def downgrade() -> None:
    if not _role_exists():
        print(f"::warning::SKIPPED revoke — role {APP_ROLE} does not exist on this database.")
        return
    op.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE {_DML} ON TABLES FROM {APP_ROLE}")
    op.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE USAGE, SELECT ON SEQUENCES FROM {APP_ROLE}")
    for t in _tables():
        op.execute(f"REVOKE {_DML} ON public.{t} FROM {APP_ROLE}")
    # Revoke on ALL tables, not just those in metadata: a table created AFTER this migration
    # received its grant through the default-privileges rule above and would otherwise keep it,
    # leaving downgrade() a non-inverse of upgrade().
    op.execute(f"REVOKE {_DML} ON ALL TABLES IN SCHEMA public FROM {APP_ROLE}")
