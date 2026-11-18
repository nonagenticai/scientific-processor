"""Z3 SMT solver verification service."""

from __future__ import annotations

import asyncio
import re
from typing import Any

import z3

from log_config import get_logger
from services.safe_expr import safe_eval

logger = get_logger(__name__)


def _safe_eval(expression: str) -> Any:
    """Parse a caller-supplied Z3-py expression without ``eval``.

    Uses the allowlist AST interpreter in :mod:`services.safe_expr`, resolving
    names only from the public Z3 API. The classic ``__subclasses__`` escape is
    rejected (no dunder access, no builtins).
    """
    # Sanitize non-identifier characters in token-like strings
    # (e.g., "entity-name" → "entity_name", "concept.type" → "concept_type")
    expression = re.sub(
        r"(?<![\"'])([A-Za-z_]\w*(?:[-.]\w+)+)(?![\"'])",
        lambda m: re.sub(r"[^A-Za-z0-9_]", "_", m.group(0)),
        expression,
    )
    # Strip variable assignment if LLM returned "x = expr" pattern
    assignment_match = re.match(r"^([A-Za-z_]\w*)\s*=(?!=)\s*(.+)$", expression, re.DOTALL)
    if assignment_match:
        expression = assignment_match.group(2).strip()
    z3_ns: dict[str, Any] = {name: getattr(z3, name) for name in dir(z3) if not name.startswith("_")}
    return safe_eval(expression, z3_ns)


class Z3Verifier:
    """Verify formal expressions using Z3 SMT solver."""

    async def verify(self, expression: str) -> dict[str, Any]:
        """Parse and verify a Z3-py expression string.

        Returns dict with keys: is_valid, counterexample, proof, error.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._verify_sync, expression)

    def _verify_sync(self, expression: str) -> dict[str, Any]:
        """Synchronous Z3 verification (runs in executor)."""
        try:
            expr = _safe_eval(expression)
        except (SyntaxError, TypeError, NameError, ValueError, AttributeError, z3.Z3Exception) as exc:
            logger.warning("Z3 parse error", expression=expression, error=str(exc))
            return {"is_valid": None, "counterexample": None, "proof": None, "error": f"Parse error: {exc}"}

        if not isinstance(expr, z3.BoolRef):
            return {
                "is_valid": None,
                "counterexample": None,
                "proof": None,
                "error": f"Expression must be a boolean Z3 expression, got {type(expr).__name__}",
            }

        solver = z3.Solver()
        solver.add(z3.Not(expr))
        result = solver.check()

        if result == z3.unsat:
            # Negation is unsatisfiable → original is always true (valid)
            return {
                "is_valid": True,
                "counterexample": None,
                "proof": "Valid by Z3 (negation is unsatisfiable)",
                "error": None,
            }
        elif result == z3.sat:
            # Negation is satisfiable → found counterexample to validity
            model = solver.model()
            return {"is_valid": False, "counterexample": str(model), "proof": None, "error": None}
        else:
            # Unknown
            return {"is_valid": None, "counterexample": None, "proof": None, "error": "Z3 returned unknown"}
