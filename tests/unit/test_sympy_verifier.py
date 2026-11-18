"""Tests for SymPy mathematical verifier."""

from __future__ import annotations

import pytest

from services.sympy_verifier import SymPyVerifier

pytestmark = pytest.mark.unit


@pytest.fixture
def verifier():
    return SymPyVerifier()


class TestSymPyVerifier:
    async def test_algebraic_identity_valid(self, verifier):
        """Eq(x**2 - 1, (x-1)*(x+1)) is an algebraic identity."""
        result = await verifier.verify("Eq(Symbol('x')**2 - 1, (Symbol('x') - 1)*(Symbol('x') + 1))")
        assert result["is_valid"] is True
        assert result["simplified"] is not None
        assert result["proof_steps"] is not None
        assert result["error"] is None

    async def test_false_equality(self, verifier):
        """Eq(x**2, x**3) is not an identity."""
        result = await verifier.verify("Eq(Symbol('x')**2, Symbol('x')**3)")
        assert result["is_valid"] is False
        assert result["error"] is None

    async def test_simple_true_expression(self, verifier):
        """Eq(1, 1) should be trivially true."""
        result = await verifier.verify("Eq(1, 1)")
        assert result["is_valid"] is True

    async def test_simple_false_expression(self, verifier):
        """Eq(1, 2) should be trivially false."""
        result = await verifier.verify("Eq(1, 2)")
        assert result["is_valid"] is False

    async def test_boolean_true(self, verifier):
        """Boolean True should verify as valid."""
        result = await verifier.verify("S.true")
        assert result["is_valid"] is True

    async def test_boolean_false(self, verifier):
        """Boolean False should verify as invalid."""
        result = await verifier.verify("S.false")
        assert result["is_valid"] is False

    async def test_simplification_returns_steps(self, verifier):
        """Verify that proof_steps are populated."""
        result = await verifier.verify("Eq(Symbol('x')**2 - 1, (Symbol('x') - 1)*(Symbol('x') + 1))")
        assert isinstance(result["proof_steps"], list)
        assert len(result["proof_steps"]) > 0

    async def test_indeterminate_expression(self, verifier):
        """Expression that cannot be determined true or false."""
        result = await verifier.verify("Symbol('x') + Symbol('y')")
        # Non-boolean, non-Eq expression → is_valid should be None
        assert result["is_valid"] is None
        assert result["error"] is None

    async def test_parse_error(self, verifier):
        """Invalid expression returns error."""
        result = await verifier.verify("this is not valid sympy !!!")
        assert result["is_valid"] is None
        assert "Parse error" in result["error"]

    async def test_builtins_blocked(self, verifier):
        """Built-in functions should be blocked in eval."""
        result = await verifier.verify("__import__('os').system('echo hacked')")
        assert result["is_valid"] is None
        assert result["error"] is not None
