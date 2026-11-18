"""Tests for Z3 SMT solver verifier."""

from __future__ import annotations

import pytest

from services.z3_verifier import Z3Verifier

pytestmark = pytest.mark.unit


@pytest.fixture
def verifier():
    return Z3Verifier()


class TestZ3Verifier:
    async def test_tautology_is_valid(self, verifier):
        """Or(p, Not(p)) is always true → is_valid=True."""
        result = await verifier.verify("Or(Bool('p'), Not(Bool('p')))")
        assert result["is_valid"] is True
        assert result["counterexample"] is None
        assert result["proof"] is not None
        assert result["error"] is None

    async def test_modus_ponens_valid(self, verifier):
        """Implies(And(p, Implies(p, q)), q) is a tautology."""
        result = await verifier.verify("Implies(And(Bool('p'), Implies(Bool('p'), Bool('q'))), Bool('q'))")
        assert result["is_valid"] is True
        assert result["error"] is None

    async def test_non_tautology_invalid(self, verifier):
        """And(p, q) is not always true → is_valid=False with counterexample."""
        result = await verifier.verify("And(Bool('p'), Bool('q'))")
        assert result["is_valid"] is False
        assert result["counterexample"] is not None
        assert result["error"] is None

    async def test_contradiction_invalid(self, verifier):
        """And(p, Not(p)) is never true → is_valid=False."""
        result = await verifier.verify("And(Bool('p'), Not(Bool('p')))")
        assert result["is_valid"] is False
        assert result["counterexample"] is not None

    async def test_integer_expression_valid(self, verifier):
        """ForAll x: x + 0 == x."""
        expr = "ForAll([Int('x')], Int('x') + 0 == Int('x'))"
        result = await verifier.verify(expr)
        assert result["is_valid"] is True

    async def test_integer_expression_invalid(self, verifier):
        """ForAll x: x > 0 is not always true."""
        expr = "ForAll([Int('x')], Int('x') > 0)"
        result = await verifier.verify(expr)
        assert result["is_valid"] is False
        assert result["counterexample"] is not None

    async def test_parse_error(self, verifier):
        """Invalid syntax returns error."""
        result = await verifier.verify("this is not valid z3")
        assert result["is_valid"] is None
        assert "Parse error" in result["error"]

    async def test_non_boolean_expression_error(self, verifier):
        """Non-boolean expression returns error."""
        result = await verifier.verify("Int('x') + Int('y')")
        assert result["is_valid"] is None
        assert "boolean" in result["error"].lower()

    async def test_builtins_blocked(self, verifier):
        """Built-in functions should be blocked."""
        result = await verifier.verify("__import__('os').system('echo hacked')")
        assert result["is_valid"] is None
        assert result["error"] is not None
