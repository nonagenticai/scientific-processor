"""Pydantic models for formal verification."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class FormalTranslateRequest(BaseModel):
    claim_text: str = Field(..., max_length=5000)
    target_language: Literal["lean4", "z3", "sympy"] = "z3"


class FormalTranslateResponse(BaseModel):
    claim_text: str
    formal_expression: str
    target_language: str
    confidence: float
    computation_time_ms: float


class FormalVerifyRequest(BaseModel):
    formal_expression: str = Field(..., max_length=5000)
    language: Literal["lean4", "z3", "sympy"] = "z3"

    @field_validator("formal_expression")
    @classmethod
    def _no_dunder(cls, value: str) -> str:
        # Defense-in-depth: the Z3/SymPy parsers reject dunder access at the AST
        # level, but there is no legitimate reason for a formal expression to
        # contain a dunder, so refuse it at the edge before it reaches a solver.
        if "__" in value:
            raise ValueError("formal_expression must not contain '__'")
        return value


class FormalVerifyResponse(BaseModel):
    expression: str
    language: str
    is_valid: bool | None = None
    counterexample: str | None = None
    proof: str | None = None
    lean_elaboration: dict[str, Any] | None = None
    computation_time_ms: float


class ProofStoreRequest(BaseModel):
    expression: str
    language: Literal["lean4", "z3", "sympy"] = "z3"
    is_valid: bool | None = None
    proof: str | None = None


class ProofRecord(BaseModel):
    claim_id: str
    key: str
    language: str = ""
    expression: str = ""
    is_valid: bool | None = None
    proof: str | None = None
    created_at: str = ""
