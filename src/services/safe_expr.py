"""Safe evaluation of caller-supplied Z3/SymPy expression strings.

The ``/api/formal/verify`` endpoint accepts an expression string that is
ultimately caller-controlled. The previous implementation used ``eval()`` with
an emptied ``__builtins__`` mapping, which does **not** sandbox anything:
attribute traversal from a literal (``().__class__.__bases__[0].__subclasses__()``)
reaches arbitrary callables and yields remote code execution.

This module parses the expression to an AST and interprets only an allowlisted
subset of expression nodes, resolving names solely from the supplied namespace.
Two invariants make the classic escapes impossible:

* **No dunder access.** Any name or attribute beginning with ``_`` is rejected,
  so ``__class__`` / ``__subclasses__`` / ``__globals__`` / ``__builtins__`` are
  unreachable.
* **No implicit builtins.** Names resolve exclusively from the passed namespace
  (the public Z3 or SymPy API); there is no fallback to Python builtins.

Anything outside the allowlist raises :class:`UnsafeExpressionError`.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any


class UnsafeExpressionError(ValueError):
    """Raised when an expression uses a construct outside the safe subset."""


# Cap on integer exponents evaluated at parse time to avoid ``10 ** 10**9`` DoS.
# Symbolic (Z3/SymPy) powers are unaffected — this only fires for two plain ints.
_MAX_INT_POW = 1000

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.MatMult: operator.matmul,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,
}

_CMP_OPS: dict[type[ast.cmpop], Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def safe_eval(expression: str, namespace: dict[str, Any]) -> Any:
    """Evaluate a Z3/SymPy expression string against ``namespace`` without eval.

    Only an allowlisted subset of Python expression syntax is permitted and
    names resolve exclusively from ``namespace``. Raises :class:`SyntaxError`
    on malformed input and :class:`UnsafeExpressionError` (a ``ValueError``
    subclass) on any disallowed construct.
    """
    tree = ast.parse(expression, mode="eval")
    return _eval(tree.body, namespace)


def _reject_dunder(name: str, kind: str) -> None:
    if name.startswith("_"):
        raise UnsafeExpressionError(f"{kind} not allowed: {name}")


def _eval(node: ast.AST, ns: dict[str, Any]) -> Any:  # noqa: C901 — flat dispatch is clearer than a table here
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (str, int, float, complex, bool)) or node.value is None:
            return node.value
        raise UnsafeExpressionError(f"constant type not allowed: {type(node.value).__name__}")

    if isinstance(node, ast.Name):
        _reject_dunder(node.id, "name")
        if node.id not in ns:
            raise UnsafeExpressionError(f"unknown name: {node.id}")
        return ns[node.id]

    if isinstance(node, ast.Attribute):
        # Non-dunder attribute access on an already-safe value (e.g. ``S.true``).
        _reject_dunder(node.attr, "attribute")
        return getattr(_eval(node.value, ns), node.attr)

    if isinstance(node, ast.Call):
        if any(isinstance(a, ast.Starred) for a in node.args):
            raise UnsafeExpressionError("*args unpacking not allowed")
        func = _eval(node.func, ns)
        if not callable(func):
            raise UnsafeExpressionError("call target is not callable")
        args = [_eval(a, ns) for a in node.args]
        kwargs: dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise UnsafeExpressionError("**kwargs unpacking not allowed")
            kwargs[kw.arg] = _eval(kw.value, ns)
        return func(*args, **kwargs)

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpressionError(f"operator not allowed: {type(node.op).__name__}")
        left = _eval(node.left, ns)
        right = _eval(node.right, ns)
        if isinstance(node.op, ast.Pow) and isinstance(left, int) and isinstance(right, int) and right > _MAX_INT_POW:
            raise UnsafeExpressionError("integer exponent too large")
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpressionError(f"unary operator not allowed: {type(node.op).__name__}")
        return op(_eval(node.operand, ns))

    if isinstance(node, ast.BoolOp):
        values = [_eval(v, ns) for v in node.values]
        result = values[0]
        for value in values[1:]:
            result = (result and value) if isinstance(node.op, ast.And) else (result or value)
        return result

    if isinstance(node, ast.Compare):
        left = _eval(node.left, ns)
        results = []
        current = left
        for op_node, comparator in zip(node.ops, node.comparators):
            op = _CMP_OPS.get(type(op_node))
            if op is None:
                raise UnsafeExpressionError(f"comparison not allowed: {type(op_node).__name__}")
            right = _eval(comparator, ns)
            results.append(op(current, right))
            current = right
        combined = results[0]
        for extra in results[1:]:  # chained comparisons (a < b < c)
            combined = combined and extra
        return combined

    if isinstance(node, ast.Subscript):
        return _eval(node.value, ns)[_eval(node.slice, ns)]

    if isinstance(node, ast.List):
        return [_eval(elt, ns) for elt in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_eval(elt, ns) for elt in node.elts)

    raise UnsafeExpressionError(f"syntax element not allowed: {type(node).__name__}")
