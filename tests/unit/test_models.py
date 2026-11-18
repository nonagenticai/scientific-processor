"""
Unit tests for Pydantic models (explainability, formal, sandbox) and constraint types.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import BaseModel, ValidationError

from custom_types.constraint_types import (
    SAFE_STRING_PATTERN,
    SafeCodeString,
    SafeString,
    SafeStringLong,
    SafeStringMedium,
    SafeStringShort,
)
from models.explainability import SubgraphXJobStatus, SubgraphXRequest
from models.formal import (
    FormalTranslateRequest,
    FormalTranslateResponse,
    FormalVerifyRequest,
    FormalVerifyResponse,
)
from models.sandbox import SandboxJobStatus, SandboxRequest

pytestmark = pytest.mark.unit


# ===========================================================================
# SubgraphXRequest
# ===========================================================================


class TestSubgraphXRequest:
    """Tests for SubgraphXRequest model."""

    def test_valid_defaults(self) -> None:
        req = SubgraphXRequest(node_id="node-1")
        assert req.node_id == "node-1"
        assert req.model_type == "gcn"
        assert req.num_iterations == 20
        assert req.subgraph_size == 5

    def test_valid_all_fields(self) -> None:
        req = SubgraphXRequest(
            node_id="n42",
            model_type="gat",
            num_iterations=50,
            subgraph_size=10,
        )
        assert req.model_type == "gat"
        assert req.num_iterations == 50
        assert req.subgraph_size == 10

    def test_model_type_graphsage(self) -> None:
        req = SubgraphXRequest(node_id="x", model_type="graphsage")
        assert req.model_type == "graphsage"

    def test_invalid_model_type(self) -> None:
        with pytest.raises(ValidationError, match="model_type"):
            SubgraphXRequest(node_id="x", model_type=cast(Any, "transformer"))

    def test_num_iterations_min_boundary(self) -> None:
        req = SubgraphXRequest(node_id="x", num_iterations=5)
        assert req.num_iterations == 5

    def test_num_iterations_below_min(self) -> None:
        with pytest.raises(ValidationError, match="num_iterations"):
            SubgraphXRequest(node_id="x", num_iterations=4)

    def test_num_iterations_max_boundary(self) -> None:
        req = SubgraphXRequest(node_id="x", num_iterations=100)
        assert req.num_iterations == 100

    def test_num_iterations_above_max(self) -> None:
        with pytest.raises(ValidationError, match="num_iterations"):
            SubgraphXRequest(node_id="x", num_iterations=101)

    def test_subgraph_size_min_boundary(self) -> None:
        req = SubgraphXRequest(node_id="x", subgraph_size=2)
        assert req.subgraph_size == 2

    def test_subgraph_size_below_min(self) -> None:
        with pytest.raises(ValidationError, match="subgraph_size"):
            SubgraphXRequest(node_id="x", subgraph_size=1)

    def test_subgraph_size_max_boundary(self) -> None:
        req = SubgraphXRequest(node_id="x", subgraph_size=20)
        assert req.subgraph_size == 20

    def test_subgraph_size_above_max(self) -> None:
        with pytest.raises(ValidationError, match="subgraph_size"):
            SubgraphXRequest(node_id="x", subgraph_size=21)


# ===========================================================================
# SubgraphXJobStatus
# ===========================================================================


class TestSubgraphXJobStatus:
    """Tests for SubgraphXJobStatus model."""

    def test_valid_defaults(self) -> None:
        status = SubgraphXJobStatus(job_id="job-1", status="pending")
        assert status.job_id == "job-1"
        assert status.status == "pending"
        assert status.progress == 0.0
        assert status.result is None
        assert status.computation_time_ms is None

    def test_valid_completed(self) -> None:
        status = SubgraphXJobStatus(
            job_id="job-2",
            status="completed",
            progress=1.0,
            result={"important_nodes": ["a", "b"]},
            computation_time_ms=1234.5,
        )
        assert status.status == "completed"
        assert status.progress == 1.0
        assert status.result == {"important_nodes": ["a", "b"]}
        assert status.computation_time_ms == 1234.5

    def test_all_valid_statuses(self) -> None:
        for s in ("pending", "running", "completed", "failed"):
            status = SubgraphXJobStatus(job_id="j", status=s)
            assert status.status == s

    def test_invalid_status(self) -> None:
        with pytest.raises(ValidationError, match="status"):
            SubgraphXJobStatus(job_id="j", status=cast(Any, "cancelled"))


# ===========================================================================
# FormalTranslateRequest
# ===========================================================================


class TestFormalTranslateRequest:
    """Tests for FormalTranslateRequest model."""

    def test_valid_defaults(self) -> None:
        req = FormalTranslateRequest(claim_text="Water boils at 100C")
        assert req.claim_text == "Water boils at 100C"
        assert req.target_language == "z3"

    def test_valid_lean4(self) -> None:
        req = FormalTranslateRequest(claim_text="x > 0", target_language="lean4")
        assert req.target_language == "lean4"

    def test_valid_sympy(self) -> None:
        req = FormalTranslateRequest(claim_text="x = 0", target_language="sympy")
        assert req.target_language == "sympy"

    def test_invalid_target_language(self) -> None:
        with pytest.raises(ValidationError, match="target_language"):
            FormalTranslateRequest(claim_text="x", target_language=cast(Any, "coq"))

    def test_claim_text_max_length(self) -> None:
        req = FormalTranslateRequest(claim_text="x" * 5000)
        assert len(req.claim_text) == 5000

    def test_claim_text_too_long(self) -> None:
        with pytest.raises(ValidationError, match="claim_text"):
            FormalTranslateRequest(claim_text="x" * 5001)

    def test_claim_text_required(self) -> None:
        with pytest.raises(ValidationError):
            FormalTranslateRequest()


# ===========================================================================
# FormalTranslateResponse
# ===========================================================================


class TestFormalTranslateResponse:
    """Tests for FormalTranslateResponse model."""

    def test_valid_construction(self) -> None:
        resp = FormalTranslateResponse(
            claim_text="x > 0",
            formal_expression="(assert (> x 0))",
            target_language="z3",
            confidence=0.95,
            computation_time_ms=42.1,
        )
        assert resp.claim_text == "x > 0"
        assert resp.formal_expression == "(assert (> x 0))"
        assert resp.target_language == "z3"
        assert resp.confidence == 0.95
        assert resp.computation_time_ms == 42.1


# ===========================================================================
# FormalVerifyRequest
# ===========================================================================


class TestFormalVerifyRequest:
    """Tests for FormalVerifyRequest model."""

    def test_valid_defaults(self) -> None:
        req = FormalVerifyRequest(formal_expression="(assert true)")
        assert req.formal_expression == "(assert true)"
        assert req.language == "z3"

    def test_valid_sympy(self) -> None:
        req = FormalVerifyRequest(formal_expression="Eq(x, 0)", language="sympy")
        assert req.language == "sympy"

    def test_valid_lean4(self) -> None:
        req = FormalVerifyRequest(formal_expression="theorem test : True := trivial", language="lean4")
        assert req.language == "lean4"

    def test_invalid_language(self) -> None:
        with pytest.raises(ValidationError, match="language"):
            FormalVerifyRequest(formal_expression="x", language=cast(Any, "coq"))


# ===========================================================================
# FormalVerifyResponse
# ===========================================================================


class TestFormalVerifyResponse:
    """Tests for FormalVerifyResponse model."""

    def test_valid_with_defaults(self) -> None:
        resp = FormalVerifyResponse(
            expression="(assert true)",
            language="z3",
            computation_time_ms=10.0,
        )
        assert resp.is_valid is None
        assert resp.counterexample is None
        assert resp.proof is None

    def test_valid_full(self) -> None:
        resp = FormalVerifyResponse(
            expression="(assert (> x 0))",
            language="z3",
            is_valid=True,
            proof="QED",
            computation_time_ms=100.0,
        )
        assert resp.is_valid is True
        assert resp.proof == "QED"

    def test_invalid_with_counterexample(self) -> None:
        resp = FormalVerifyResponse(
            expression="(assert false)",
            language="z3",
            is_valid=False,
            counterexample="x=0",
            computation_time_ms=50.0,
        )
        assert resp.is_valid is False
        assert resp.counterexample == "x=0"


# ===========================================================================
# SandboxRequest
# ===========================================================================


class TestSandboxRequest:
    """Tests for SandboxRequest model."""

    def test_valid_defaults(self) -> None:
        req = SandboxRequest(code="print('hello')")
        assert req.code == "print('hello')"
        assert req.language == "python"
        assert req.timeout_seconds == 60
        assert req.input_data is None

    def test_valid_with_input_data(self) -> None:
        req = SandboxRequest(code="x = data['x']", input_data={"x": 42})
        assert req.input_data == {"x": 42}

    def test_code_max_length(self) -> None:
        req = SandboxRequest(code="x" * 50000)
        assert len(req.code) == 50000

    def test_code_too_long(self) -> None:
        with pytest.raises(ValidationError, match="code"):
            SandboxRequest(code="x" * 50001)

    def test_code_required(self) -> None:
        with pytest.raises(ValidationError):
            SandboxRequest()

    def test_invalid_language(self) -> None:
        with pytest.raises(ValidationError, match="language"):
            SandboxRequest(code="console.log(1)", language=cast(Any, "javascript"))

    def test_timeout_min_boundary(self) -> None:
        req = SandboxRequest(code="x", timeout_seconds=5)
        assert req.timeout_seconds == 5

    def test_timeout_below_min(self) -> None:
        with pytest.raises(ValidationError, match="timeout_seconds"):
            SandboxRequest(code="x", timeout_seconds=4)

    def test_timeout_max_boundary(self) -> None:
        req = SandboxRequest(code="x", timeout_seconds=300)
        assert req.timeout_seconds == 300

    def test_timeout_above_max(self) -> None:
        with pytest.raises(ValidationError, match="timeout_seconds"):
            SandboxRequest(code="x", timeout_seconds=301)


# ===========================================================================
# SandboxJobStatus
# ===========================================================================


class TestSandboxJobStatus:
    """Tests for SandboxJobStatus model."""

    def test_valid_defaults(self) -> None:
        status = SandboxJobStatus(
            job_id="sb-1",
            status="pending",
            created_at="2026-03-19T00:00:00Z",
        )
        assert status.job_id == "sb-1"
        assert status.status == "pending"
        assert status.completed_at is None
        assert status.result_url is None
        assert status.error is None
        assert status.execution_time_ms is None

    def test_valid_completed(self) -> None:
        status = SandboxJobStatus(
            job_id="sb-2",
            status="completed",
            created_at="2026-03-19T00:00:00Z",
            completed_at="2026-03-19T00:01:00Z",
            result_url="/results/sb-2",
            execution_time_ms=5000.0,
        )
        assert status.status == "completed"
        assert status.result_url == "/results/sb-2"

    def test_valid_failed(self) -> None:
        status = SandboxJobStatus(
            job_id="sb-3",
            status="failed",
            created_at="2026-03-19T00:00:00Z",
            error="ImportError: no module named 'evil'",
        )
        assert status.error == "ImportError: no module named 'evil'"

    def test_all_valid_statuses(self) -> None:
        for s in ("pending", "running", "completed", "failed", "timeout"):
            status = SandboxJobStatus(job_id="j", status=s, created_at="2026-01-01")
            assert status.status == s

    def test_invalid_status(self) -> None:
        with pytest.raises(ValidationError, match="status"):
            SandboxJobStatus(job_id="j", status=cast(Any, "cancelled"), created_at="2026-01-01")


# ===========================================================================
# Constraint Types (SafeString variants)
# ===========================================================================


class _SafeStringModel(BaseModel):
    value: SafeString


class _SafeStringShortModel(BaseModel):
    value: SafeStringShort


class _SafeStringMediumModel(BaseModel):
    value: SafeStringMedium


class _SafeStringLongModel(BaseModel):
    value: SafeStringLong


class _SafeCodeStringModel(BaseModel):
    value: SafeCodeString


class TestSafeString:
    """Tests for SafeString (unconstrained length, no control chars)."""

    def test_normal_text(self) -> None:
        m = _SafeStringModel(value="Hello, world!")
        assert m.value == "Hello, world!"

    def test_empty_string(self) -> None:
        m = _SafeStringModel(value="")
        assert m.value == ""

    def test_newlines_allowed(self) -> None:
        m = _SafeStringModel(value="line1\nline2\r\nline3")
        assert "\n" in m.value

    def test_tabs_allowed(self) -> None:
        m = _SafeStringModel(value="col1\tcol2")
        assert "\t" in m.value

    def test_unicode_allowed(self) -> None:
        m = _SafeStringModel(value="Schrodinger's cat is in state |psi>")
        assert m.value == "Schrodinger's cat is in state |psi>"

    def test_null_byte_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _SafeStringModel(value="hello\x00world")

    def test_control_char_x01_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _SafeStringModel(value="hello\x01world")

    def test_control_char_x08_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _SafeStringModel(value="bad\x08text")

    def test_control_char_x0b_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _SafeStringModel(value="bad\x0btext")

    def test_control_char_x0c_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _SafeStringModel(value="bad\x0ctext")

    def test_control_char_x0e_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _SafeStringModel(value="bad\x0etext")

    def test_control_char_x1f_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _SafeStringModel(value="bad\x1ftext")

    def test_del_char_x7f_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _SafeStringModel(value="bad\x7ftext")


class TestSafeStringShort:
    """Tests for SafeStringShort (1-256 chars, no control chars)."""

    def test_valid(self) -> None:
        m = _SafeStringShortModel(value="hello")
        assert m.value == "hello"

    def test_min_length(self) -> None:
        m = _SafeStringShortModel(value="x")
        assert m.value == "x"

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _SafeStringShortModel(value="")

    def test_max_length(self) -> None:
        m = _SafeStringShortModel(value="a" * 256)
        assert len(m.value) == 256

    def test_too_long(self) -> None:
        with pytest.raises(ValidationError):
            _SafeStringShortModel(value="a" * 257)

    def test_control_char_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _SafeStringShortModel(value="bad\x00text")


class TestSafeStringMedium:
    """Tests for SafeStringMedium (1-1000 chars, no control chars)."""

    def test_valid(self) -> None:
        m = _SafeStringMediumModel(value="medium text")
        assert m.value == "medium text"

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _SafeStringMediumModel(value="")

    def test_max_length(self) -> None:
        m = _SafeStringMediumModel(value="b" * 1000)
        assert len(m.value) == 1000

    def test_too_long(self) -> None:
        with pytest.raises(ValidationError):
            _SafeStringMediumModel(value="b" * 1001)


class TestSafeStringLong:
    """Tests for SafeStringLong (1-10000 chars, no control chars)."""

    def test_valid(self) -> None:
        m = _SafeStringLongModel(value="long text here")
        assert m.value == "long text here"

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _SafeStringLongModel(value="")

    def test_max_length(self) -> None:
        m = _SafeStringLongModel(value="c" * 10000)
        assert len(m.value) == 10000

    def test_too_long(self) -> None:
        with pytest.raises(ValidationError):
            _SafeStringLongModel(value="c" * 10001)


class TestSafeCodeString:
    """Tests for SafeCodeString (max 50000 chars, no control chars)."""

    def test_valid(self) -> None:
        m = _SafeCodeStringModel(value="print('hello')")
        assert m.value == "print('hello')"

    def test_empty_allowed(self) -> None:
        """SafeCodeString has no min_length, so empty is allowed."""
        m = _SafeCodeStringModel(value="")
        assert m.value == ""

    def test_max_length(self) -> None:
        m = _SafeCodeStringModel(value="d" * 50000)
        assert len(m.value) == 50000

    def test_too_long(self) -> None:
        with pytest.raises(ValidationError):
            _SafeCodeStringModel(value="d" * 50001)

    def test_control_char_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _SafeCodeStringModel(value="code\x00injection")


class TestSafeStringPattern:
    """Tests for the SAFE_STRING_PATTERN constant itself."""

    def test_pattern_is_string(self) -> None:
        assert isinstance(SAFE_STRING_PATTERN, str)

    def test_pattern_anchored(self) -> None:
        assert SAFE_STRING_PATTERN.startswith("^")
        assert SAFE_STRING_PATTERN.endswith("$")
