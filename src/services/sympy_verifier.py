"""SymPy mathematical verification service."""

from __future__ import annotations

import asyncio
from typing import Any

import sympy

from log_config import get_logger
from services.safe_expr import safe_eval

logger = get_logger(__name__)


def _safe_sympify(expression: str) -> Any:
    """Parse a caller-supplied SymPy expression without ``eval``.

    Uses the allowlist AST interpreter in :mod:`services.safe_expr`, resolving
    names only from the public SymPy API. The classic ``__subclasses__`` escape
    is rejected (no dunder access, no builtins).
    """
    sympy_ns: dict[str, Any] = {name: getattr(sympy, name) for name in dir(sympy) if not name.startswith("_")}
    return safe_eval(expression, sympy_ns)


class SymPyVerifier:
    """Verify mathematical expressions using SymPy."""

    async def verify(self, expression: str) -> dict[str, Any]:
        """Verify a SymPy expression string.

        Returns dict with keys: is_valid, simplified, proof_steps, error.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._verify_sync, expression)

    def _verify_sync(self, expression: str) -> dict[str, Any]:
        """Synchronous SymPy verification (runs in executor)."""
        try:
            expr = _safe_sympify(expression)
        except (SyntaxError, TypeError, NameError, ValueError, AttributeError, ZeroDivisionError) as exc:
            logger.warning("SymPy parse error", expression=expression, error=str(exc))
            return {"is_valid": None, "simplified": None, "proof_steps": None, "error": f"Parse error: {exc}"}

        proof_steps: list[str] = []

        # Handle Equality (Eq)
        if isinstance(expr, sympy.Eq):
            proof_steps.append(f"Parsed equality: {expr}")
            diff = sympy.simplify(expr.lhs - expr.rhs)
            proof_steps.append(f"Simplified (lhs - rhs): {diff}")
            if diff == 0:
                return {
                    "is_valid": True,
                    "simplified": str(expr),
                    "proof_steps": proof_steps + ["lhs - rhs simplifies to 0 → equality holds"],
                    "error": None,
                }
            # Try expand then simplify
            diff_expanded = sympy.simplify(sympy.expand(expr.lhs - expr.rhs))
            proof_steps.append(f"Expanded and simplified (lhs - rhs): {diff_expanded}")
            if diff_expanded == 0:
                return {
                    "is_valid": True,
                    "simplified": str(expr),
                    "proof_steps": proof_steps + ["After expansion, lhs - rhs simplifies to 0 → equality holds"],
                    "error": None,
                }
            return {
                "is_valid": False,
                "simplified": str(diff_expanded),
                "proof_steps": proof_steps + [f"Residual: {diff_expanded} (not identically zero)"],
                "error": None,
            }

        # Handle boolean expressions (e.g., True, False, relational)
        simplified = sympy.simplify(expr)
        proof_steps.append(f"Simplified expression: {simplified}")

        if simplified is sympy.true or simplified == True:  # noqa: E712
            return {
                "is_valid": True,
                "simplified": str(simplified),
                "proof_steps": proof_steps + ["Expression simplifies to True"],
                "error": None,
            }
        elif simplified is sympy.false or simplified == False:  # noqa: E712
            return {
                "is_valid": False,
                "simplified": str(simplified),
                "proof_steps": proof_steps + ["Expression simplifies to False"],
                "error": None,
            }

        # Cannot determine truth value — return simplified form
        return {
            "is_valid": None,
            "simplified": str(simplified),
            "proof_steps": proof_steps + ["Could not determine truth value"],
            "error": None,
        }
