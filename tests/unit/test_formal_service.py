"""Tests for FormalService — orchestration of LLM translation + Z3/SymPy verification."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from services.formal_service import FormalService

pytestmark = pytest.mark.unit


def _llm_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


@pytest.fixture
def inference_mock():
    mock = AsyncMock()
    mock.chat_completion = AsyncMock()
    return mock


@pytest.fixture
def minio_mock():
    mock = AsyncMock()
    mock.upload = AsyncMock()
    mock.download = AsyncMock()
    mock.list_objects = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def lean4_mock():
    mock = AsyncMock()
    mock.check = AsyncMock()
    mock.elaborate = AsyncMock()
    mock.prove = AsyncMock()
    mock.health_check = AsyncMock()
    return mock


@pytest.fixture
def service(inference_mock, minio_mock, lean4_mock):
    from services.sympy_verifier import SymPyVerifier
    from services.z3_verifier import Z3Verifier

    return FormalService(
        inference_client=inference_mock,
        z3_verifier=Z3Verifier(),
        sympy_verifier=SymPyVerifier(),
        minio_client=minio_mock,
        lean4_client=lean4_mock,
    )


@pytest.fixture
def service_no_minio(inference_mock, lean4_mock):
    from services.sympy_verifier import SymPyVerifier
    from services.z3_verifier import Z3Verifier

    return FormalService(
        inference_client=inference_mock,
        z3_verifier=Z3Verifier(),
        sympy_verifier=SymPyVerifier(),
        minio_client=None,
        lean4_client=lean4_mock,
    )


@pytest.fixture
def service_no_lean4(inference_mock, minio_mock):
    from services.sympy_verifier import SymPyVerifier
    from services.z3_verifier import Z3Verifier

    return FormalService(
        inference_client=inference_mock,
        z3_verifier=Z3Verifier(),
        sympy_verifier=SymPyVerifier(),
        minio_client=minio_mock,
        lean4_client=None,
    )


class TestTranslate:
    async def test_translate_z3(self, service, inference_mock):
        inference_mock.chat_completion.return_value = _llm_response(
            "Implies(And(Bool('p'), Implies(Bool('p'), Bool('q'))), Bool('q'))"
        )
        result = await service.translate("If p and p implies q, then q", "z3", "test-tenant")
        assert result.claim_text == "If p and p implies q, then q"
        assert result.target_language == "z3"
        assert "Implies" in result.formal_expression
        assert result.computation_time_ms > 0
        inference_mock.chat_completion.assert_called_once()

    async def test_translate_sympy(self, service, inference_mock):
        inference_mock.chat_completion.return_value = _llm_response(
            "Eq(Symbol('x')**2 - 1, (Symbol('x') - 1)*(Symbol('x') + 1))"
        )
        result = await service.translate("x squared minus 1 equals (x-1)(x+1)", "sympy", None)
        assert result.target_language == "sympy"
        assert "Eq" in result.formal_expression

    async def test_translate_strips_code_fences(self, service, inference_mock):
        inference_mock.chat_completion.return_value = _llm_response("```python\nOr(Bool('p'), Not(Bool('p')))\n```")
        result = await service.translate("p or not p", "z3", None)
        assert "```" not in result.formal_expression
        assert "Or(Bool('p'), Not(Bool('p')))" == result.formal_expression


class TestVerify:
    async def test_verify_z3_valid(self, service):
        result = await service.verify("Or(Bool('p'), Not(Bool('p')))", "z3")
        assert result.is_valid is True
        assert result.language == "z3"
        assert result.computation_time_ms > 0

    async def test_verify_z3_invalid(self, service):
        result = await service.verify("And(Bool('p'), Bool('q'))", "z3")
        assert result.is_valid is False
        assert result.counterexample is not None

    async def test_verify_sympy_valid(self, service):
        result = await service.verify("Eq(Symbol('x')**2 - 1, (Symbol('x') - 1)*(Symbol('x') + 1))", "sympy")
        assert result.is_valid is True
        assert result.language == "sympy"

    async def test_verify_sympy_invalid(self, service):
        result = await service.verify("Eq(1, 2)", "sympy")
        assert result.is_valid is False

    async def test_verify_unsupported_language(self, service):
        result = await service.verify("some expr", "coq")
        assert result.is_valid is None
        assert "Unsupported" in result.proof


class TestLean4Translation:
    async def test_translate_lean4_calls_llm_then_elaborate(self, service, inference_mock, lean4_mock):
        """Translation to lean4 should call LLM, then elaborate for type-checking."""
        inference_mock.chat_completion.return_value = _llm_response("theorem test : 1 + 1 = 2 := by norm_num")
        lean4_mock.elaborate.return_value = {
            "success": True,
            "elaborated": "theorem test : 1 + 1 = 2 := by norm_num",
            "type_info": "Nat",
            "errors": [],
        }

        result = await service.translate("1 plus 1 equals 2", "lean4", "tenant-1")
        assert result.target_language == "lean4"
        assert "norm_num" in result.formal_expression
        lean4_mock.elaborate.assert_called_once()

    async def test_translate_lean4_elaborate_fails_retries_once(self, service, inference_mock, lean4_mock):
        """If elaboration fails, retry LLM translation with error feedback (1 retry)."""
        inference_mock.chat_completion.side_effect = [
            _llm_response("theorem bad : True := sorry"),
            _llm_response("theorem good : True := trivial"),
        ]
        lean4_mock.elaborate.side_effect = [
            {"success": False, "elaborated": None, "type_info": None, "errors": ["sorry not allowed"]},
            {"success": True, "elaborated": "theorem good : True := trivial", "type_info": "True", "errors": []},
        ]

        result = await service.translate("trivially true", "lean4", "tenant-1")
        assert result.target_language == "lean4"
        assert inference_mock.chat_completion.call_count == 2
        assert lean4_mock.elaborate.call_count == 2

    async def test_translate_lean4_elaborate_fails_both_attempts(self, service, inference_mock, lean4_mock):
        """If both elaborate attempts fail, return with confidence 0."""
        inference_mock.chat_completion.side_effect = [
            _llm_response("theorem bad1 : True := sorry"),
            _llm_response("theorem bad2 : True := sorry"),
        ]
        lean4_mock.elaborate.side_effect = [
            {"success": False, "elaborated": None, "type_info": None, "errors": ["error1"]},
            {"success": False, "elaborated": None, "type_info": None, "errors": ["error2"]},
        ]

        result = await service.translate("some claim", "lean4", None)
        assert result.confidence == 0.0

    async def test_translate_lean4_no_client_falls_back(self, service_no_lean4, inference_mock):
        """Without lean4 client, lean4 translation still calls LLM but skips elaboration."""
        inference_mock.chat_completion.return_value = _llm_response("theorem test : True := trivial")
        result = await service_no_lean4.translate("true claim", "lean4", None)
        assert result.target_language == "lean4"
        assert result.confidence == 0.8  # default LLM confidence


class TestLean4Verify:
    async def test_verify_lean4_calls_check(self, service, lean4_mock):
        """Lean4 verification routes to lean4_client.check()."""
        lean4_mock.check.return_value = {
            "success": True,
            "errors": [],
            "messages": ["Verification successful"],
        }
        result = await service.verify("theorem test : True := trivial", "lean4")
        assert result.is_valid is True
        assert result.language == "lean4"
        lean4_mock.check.assert_called_once()

    async def test_verify_lean4_check_fails(self, service, lean4_mock):
        lean4_mock.check.return_value = {
            "success": False,
            "errors": ["type mismatch"],
            "messages": [],
        }
        result = await service.verify("theorem bad : False := trivial", "lean4")
        assert result.is_valid is False
        assert result.lean_elaboration is not None

    async def test_verify_lean4_no_client(self, service_no_lean4):
        result = await service_no_lean4.verify("theorem test : True := trivial", "lean4")
        assert result.is_valid is None
        assert "not configured" in (result.proof or "").lower()


class TestProofStorage:
    async def test_store_proof(self, service, minio_mock):
        key = await service.store_proof("claim-1", {"expression": "Or(p, Not(p))", "is_valid": True}, "tenant-1")
        assert key.startswith("proofs/claim-1/")
        assert key.endswith(".json")
        minio_mock.upload.assert_called_once()

    async def test_store_proof_no_minio_raises(self, service_no_minio):
        with pytest.raises(RuntimeError, match="MinIO client not configured"):
            await service_no_minio.store_proof("claim-1", {}, None)

    async def test_get_proofs_returns_list(self, service, minio_mock):
        proof_record = {"claim_id": "claim-1", "key": "proofs/claim-1/test.json", "is_valid": True}
        minio_mock.list_objects.return_value = ["proofs/claim-1/test.json"]
        minio_mock.download.return_value = json.dumps(proof_record).encode()

        proofs = await service.get_proofs("claim-1")
        assert len(proofs) == 1
        assert proofs[0]["claim_id"] == "claim-1"

    async def test_get_proofs_no_minio_returns_empty(self, service_no_minio):
        proofs = await service_no_minio.get_proofs("claim-1")
        assert proofs == []

    async def test_get_proofs_empty(self, service, minio_mock):
        minio_mock.list_objects.return_value = []
        proofs = await service.get_proofs("claim-999")
        assert proofs == []


class TestExtractExpression:
    def test_plain_expression(self):
        assert FormalService._extract_expression("Or(p, q)") == "Or(p, q)"

    def test_code_fence_python(self):
        raw = "```python\nOr(p, q)\n```"
        assert FormalService._extract_expression(raw) == "Or(p, q)"

    def test_code_fence_bare(self):
        raw = "```\nOr(p, q)\n```"
        assert FormalService._extract_expression(raw) == "Or(p, q)"

    def test_whitespace_stripped(self):
        assert FormalService._extract_expression("  Or(p, q)  ") == "Or(p, q)"
