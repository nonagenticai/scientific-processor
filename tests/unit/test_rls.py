"""RLS coverage guard for scientific_processor (HIPAA G4).

Asserts the shared ``kag_common.database.rls`` helper places RLS on both tenant
tables — ``jobs`` (global-or-own, nullable tenant_id) and ``verification_history``
(strict, NOT NULL) — and nothing else. A new ``tenant_id`` table added without an
explicit RLS/exclusion decision fails this test.
"""

from __future__ import annotations

import pytest
from kag_common.database import rls

from database import Base

pytestmark = pytest.mark.unit

_GUC = "NULLIF(current_setting('app.current_tenant', true), '')"

# Tables carrying tenant_id but intentionally kept OUT of RLS. None today.
_EXCLUDE: set[str] = set()


def test_both_tenant_tables_are_under_rls() -> None:
    assert rls.tenant_table_names(Base.metadata, exclude=_EXCLUDE) == [
        "jobs",
        "verification_history",
    ]


def test_excluded_tables_actually_carry_tenant_id() -> None:
    all_tenant = set(rls.tenant_table_names(Base.metadata))
    assert _EXCLUDE <= all_tenant


def test_jobs_gets_a_global_or_own_policy() -> None:
    # jobs.tenant_id is NULLABLE → global-or-own: NULL rows are shared, writes scoped.
    stmts = rls.upgrade_statements(Base.metadata, exclude={"verification_history"})
    joined = " ".join(stmts)
    pred = f"tenant_id = {_GUC}::uuid"
    assert "ALTER TABLE jobs ENABLE ROW LEVEL SECURITY" in stmts
    assert f"FOR SELECT USING (tenant_id IS NULL OR {pred})" in joined
    assert f"FOR INSERT WITH CHECK ({pred})" in joined
    assert "::uuid" in joined  # uuid tenant_id → the GUC (always text) must be cast


def test_verification_history_gets_a_strict_policy() -> None:
    # verification_history.tenant_id is NOT NULL → strict equality, no global clause.
    stmts = rls.upgrade_statements(Base.metadata, exclude={"jobs"})
    joined = " ".join(stmts)
    pred = f"tenant_id = {_GUC}::uuid"
    assert "ALTER TABLE verification_history ENABLE ROW LEVEL SECURITY" in stmts
    assert f"CREATE POLICY tenant_isolation ON verification_history USING ({pred}) WITH CHECK ({pred})" in joined
    assert "IS NULL" not in joined


def test_downgrade_is_reversible() -> None:
    stmts = rls.downgrade_statements(Base.metadata, exclude=_EXCLUDE)
    joined = " ".join(stmts)
    for table in ("jobs", "verification_history"):
        assert f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY" in joined


def test_tenant_id_is_a_real_uuid_column_with_as_uuid_false() -> None:
    """Pin the column type AND the ``as_uuid=False`` half of it.

    Both halves are load-bearing and neither is caught by a type checker:

    * a ``String`` column would silently re-render the RLS predicate WITHOUT the
      ``::uuid`` cast (``rls._predicate`` keys the cast off ``isinstance(col.type,
      String)``), which against a ``uuid`` column raises at DDL time;
    * ``as_uuid=True`` would hand Python ``uuid.UUID`` objects, and
      ``uuid.UUID(t) == t`` is ``False`` for every ``str`` t — silently breaking
      ``PersistentJobManager.get_job``'s ``row.tenant_id != tenant_id`` guard (404s
      the owner on their own job) and ``JobStatus.tenant_id: str | None``.
    """
    import uuid as _uuid

    from sqlalchemy import String as SAString
    from sqlalchemy.dialects.postgresql import UUID as PgUUID  # noqa: N811

    for table in ("jobs", "verification_history"):
        col = Base.metadata.tables[table].columns["tenant_id"]
        assert isinstance(col.type, PgUUID), f"{table}.tenant_id must be a Postgres uuid column"
        assert not isinstance(col.type, SAString), f"{table}.tenant_id must not be a String"
        assert col.type.as_uuid is False, (
            f"{table}.tenant_id must use as_uuid=False — the Python value has to stay a str"
        )

    # The reason, executable: this is the comparison that would silently go False.
    t = "019de842-c92a-7e37-8fdd-eb0d8c5396f9"
    assert _uuid.UUID(t) != t


def test_nullability_split_drives_the_two_policy_shapes() -> None:
    """jobs is NULLABLE (4-policy global-or-own), verification_history is NOT (1 strict)."""
    assert Base.metadata.tables["jobs"].columns["tenant_id"].nullable is True
    assert Base.metadata.tables["verification_history"].columns["tenant_id"].nullable is False


def _renders_from_metadata(path) -> bool:  # noqa: ANN001
    """True if the migration at ``path`` builds its DDL from live model metadata.

    Decided on the AST rather than by grepping the source, so a docstring that
    *explains* the hazard (both migrations have one) does not read as the hazard.
    """
    import ast

    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("kag_common"):
            return True
        if isinstance(node, ast.Attribute) and node.attr == "metadata":
            return True
        if isinstance(node, ast.Name) and node.id in {"upgrade_statements", "downgrade_statements"}:
            return True
        if isinstance(node, ast.Attribute) and node.attr in {
            "upgrade_statements",
            "downgrade_statements",
        }:
            return True
    return False


def test_migration_003_is_frozen_to_the_varchar_shape() -> None:
    """``003`` must contain LITERAL DDL, never render it from live ``Base.metadata``.

    Mutation-proved 2026-08-28 against a fresh empty Postgres 18: with ``003``
    restored to its metadata-rendering form, ``alembic upgrade head`` from zero
    dies at 002 -> 003 with ``operator does not exist: character varying = uuid``.
    An already-migrated database (prod) shows nothing at all, so this ships green.

    Cause: ``rls._predicate`` picks the ``::uuid`` cast off the CURRENT model type
    (``isinstance(col.type, String)``), and ``upgrade_statements`` picks the
    strict-vs-global POLICY SHAPE off the current model's nullability. A migration
    that renders at runtime therefore describes today's model, not the schema as
    it stood at its own revision. ``005`` flipped the columns to ``uuid``, so ``003``
    had to be pinned to the varchar predicate it was actually written for.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "alembic" / "versions" / "003_rls.py"
    # Checked on the AST, not the raw text: both docstrings legitimately *name*
    # ``Base.metadata`` / ``upgrade_statements`` while explaining why they are gone.
    assert _renders_from_metadata(path) is False, (
        "003 must not render DDL from live model metadata — that is the bug above."
    )

    spec = importlib.util.spec_from_file_location("_m003", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frozen: list[str] = list(module._UPGRADE)
    joined = " ".join(frozen)

    # varchar at revision 003 → text-to-text, NO cast.
    assert "::uuid" not in joined
    # ...and the 4-vs-1 split as it stood then.
    assert sum(1 for s in frozen if s.startswith("CREATE POLICY") and " ON jobs " in s) == 4
    assert sum(1 for s in frozen if s.startswith("CREATE POLICY") and " ON verification_history " in s) == 1
    assert f"FOR SELECT USING (tenant_id IS NULL OR tenant_id = {_GUC})" in joined


def test_migration_005_pins_its_predicate_literally_too() -> None:
    """Same rule for the new revision: literals, and the two shapes kept apart.

    Asserted on the imported module's constants rather than on raw source text, so
    the check survives reformatting and cannot be satisfied by a comment.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "alembic" / "versions" / "005_tenant_id_uuid.py"
    assert _renders_from_metadata(path) is False

    spec = importlib.util.spec_from_file_location("_m005", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # uuid column → the always-text GUC must be cast; downgrade goes back to text.
    assert module._PREDICATE_UUID == f"tenant_id = {_GUC}::uuid"
    assert module._PREDICATE_TEXT == f"tenant_id = {_GUC}"

    # Capture the DDL _create_policies actually emits, and pin the 4-vs-1 split.
    emitted: list[str] = []
    original = module.op.execute
    module.op.execute = emitted.append  # type: ignore[assignment]
    try:
        module._create_policies(module._PREDICATE_UUID)
    finally:
        module.op.execute = original  # type: ignore[assignment]

    creates = [stmt for stmt in emitted if stmt.startswith("CREATE POLICY")]
    assert sum(1 for stmt in creates if " ON jobs " in stmt) == 4, (
        "jobs.tenant_id is NULLABLE → it must keep the 4-policy global-or-own shape"
    )
    assert sum(1 for stmt in creates if " ON verification_history " in stmt) == 1, (
        "verification_history.tenant_id is NOT NULL → one strict policy, no global clause"
    )
    joined = " ".join(creates)
    # The global-read clause is the whole point of the nullable shape: without it,
    # every tenant_id IS NULL row silently disappears for every tenant.
    assert (
        f"CREATE POLICY tenant_isolation_read ON jobs FOR SELECT "
        f"USING (tenant_id IS NULL OR {module._PREDICATE_UUID})" in joined
    )
    # ...and it must NOT leak onto the strict table.
    strict = next(stmt for stmt in creates if " ON verification_history " in stmt)
    assert "IS NULL" not in strict
