"""Regression tests: the formal-verify path must not allow code execution.

Guards the fix for the ``eval()``-of-caller-input RCE in the Z3/SymPy verifiers.
Legitimate expressions must still verify; every sandbox-escape payload must be
rejected as a parse error (never executed).
"""

from __future__ import annotations

import pytest

from models.formal import FormalVerifyRequest
from services.safe_expr import UnsafeExpressionError, safe_eval
from services.sympy_verifier import SymPyVerifier
from services.z3_verifier import Z3Verifier

pytestmark = pytest.mark.unit

# Classic Python-sandbox escapes plus a marker-writing payload. If any of these
# executed, the process would traverse to arbitrary callables / imports.
ATTACK_PAYLOADS = [
    "().__class__.__bases__[0].__subclasses__()",
    "().__class__.__mro__[1].__subclasses__()[0]",
    "__import__('os').system('id')",
    "__builtins__",
    "lambda: 1",
    "[x for x in ().__class__.__bases__]",
    "getattr(Bool, '__globals__')",
]


class TestSafeEvalBlocksEscapes:
    def test_legit_z3_still_parses(self):
        import z3

        ns = {n: getattr(z3, n) for n in dir(z3) if not n.startswith("_")}
        expr = safe_eval("Implies(And(Bool('p'), Bool('q')), Bool('p'))", ns)
        assert isinstance(expr, z3.BoolRef)

    def test_legit_sympy_still_parses(self):
        import sympy

        ns = {n: getattr(sympy, n) for n in dir(sympy) if not n.startswith("_")}
        expr = safe_eval("Eq(sin(Symbol('t'))**2 + cos(Symbol('t'))**2, 1)", ns)
        assert isinstance(expr, sympy.Eq)

    @pytest.mark.parametrize("payload", ATTACK_PAYLOADS)
    def test_payload_rejected(self, payload):
        import z3

        ns = {n: getattr(z3, n) for n in dir(z3) if not n.startswith("_")}
        with pytest.raises((UnsafeExpressionError, SyntaxError)):
            safe_eval(payload, ns)


class TestVerifiersRejectEscapes:
    @pytest.mark.parametrize("payload", ATTACK_PAYLOADS)
    async def test_z3_returns_parse_error_not_execution(self, payload):
        result = await Z3Verifier().verify(payload)
        # Rejected as a parse error — never executed, never a valid verdict.
        assert result["is_valid"] is None
        assert result["error"] is not None

    @pytest.mark.parametrize("payload", ATTACK_PAYLOADS)
    async def test_sympy_returns_parse_error_not_execution(self, payload):
        result = await SymPyVerifier().verify(payload)
        assert result["is_valid"] is None
        assert result["error"] is not None

    async def test_z3_still_verifies_tautology(self):
        result = await Z3Verifier().verify("Or(Bool('p'), Not(Bool('p')))")
        assert result["is_valid"] is True


class TestFormalVerifyRequestValidation:
    def test_dunder_rejected_at_model(self):
        with pytest.raises(ValueError, match="__"):
            FormalVerifyRequest(formal_expression="().__class__", language="z3")

    def test_overlong_rejected(self):
        with pytest.raises(ValueError):
            FormalVerifyRequest(formal_expression="A" * 5001, language="z3")

    def test_legit_expression_accepted(self):
        req = FormalVerifyRequest(formal_expression="And(Bool('p'), Bool('q'))", language="z3")
        assert req.language == "z3"
