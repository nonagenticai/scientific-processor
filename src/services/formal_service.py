"""Formal verification orchestration — LLM translation + Z3/SymPy/Lean4 verification."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from clients.inference_router import InferenceRouterClient
from clients.minio_client import MinIOClient
from log_config import get_logger
from models.formal import (
    FormalTranslateResponse,
    FormalVerifyResponse,
)
from services.sympy_verifier import SymPyVerifier
from services.z3_verifier import Z3Verifier

if TYPE_CHECKING:
    from clients.lean4_prover import Lean4ProverClient

logger = get_logger(__name__)

_TRANSLATE_PROMPT = """You are a formal logic translator. Convert the following natural language claim into a {language} expression.

Claim: {claim_text}

Rules for Z3:
- Use z3 Python API syntax (e.g., And, Or, Not, Implies, ForAll, Exists)
- Declare variables with appropriate sorts (Int, Real, Bool, etc.)
- Return ONLY the Python expression, no explanation
- Example: Implies(And(Bool('p'), Implies(Bool('p'), Bool('q'))), Bool('q'))

Rules for SymPy:
- Use sympy syntax (e.g., Eq, simplify, symbols, Symbol)
- Declare symbols inline with Symbol('x') or symbols('x y')
- Return ONLY the Python expression, no explanation
- Example: Eq(Symbol('x')**2 - 1, (Symbol('x') - 1)*(Symbol('x') + 1))

Expression:"""

_LEAN4_TRANSLATE_PROMPT = """You are a Lean 4 formal proof translator. Convert the following natural language claim into a Lean 4 theorem statement with proof.

Claim: {claim_text}

Rules:
- Use Lean 4 syntax (not Lean 3)
- Include both the theorem statement and proof
- Use tactics like simp, norm_num, omega, ring, linarith where appropriate
- Return ONLY valid Lean 4 code, no explanation
- Example: theorem test : 1 + 1 = 2 := by norm_num

Lean 4 code:"""

_LEAN4_RETRY_PROMPT = """The Lean 4 code you generated had elaboration errors. Please fix it.

Original claim: {claim_text}
Your previous code: {previous_code}
Elaboration errors: {errors}

Return ONLY the corrected Lean 4 code:"""


class FormalService:
    """Orchestrates LLM translation + Z3/SymPy/Lean4 verification."""

    def __init__(
        self,
        inference_client: InferenceRouterClient,
        z3_verifier: Z3Verifier,
        sympy_verifier: SymPyVerifier,
        minio_client: MinIOClient | None,
        lean4_client: Lean4ProverClient | None = None,
    ) -> None:
        self.inference_client = inference_client
        self.z3_verifier = z3_verifier
        self.sympy_verifier = sympy_verifier
        self.minio_client = minio_client
        self.lean4_client = lean4_client

    async def translate(self, claim_text: str, target_language: str, tenant_id: str | None) -> FormalTranslateResponse:
        """Use LLM to translate natural language claim to formal expression."""
        start = time.monotonic()

        if target_language == "lean4":
            return await self._translate_lean4(claim_text, tenant_id, start)

        prompt = _TRANSLATE_PROMPT.format(language=target_language, claim_text=claim_text)
        messages = [{"role": "user", "content": prompt}]

        resp = await self.inference_client.chat_completion(
            messages=messages,
            model="default",
            temperature=0.0,
            max_tokens=1024,
            tenant_id=tenant_id,
            schema_id=f"sp-formal-translate-{target_language}",
        )

        choices = resp.get("choices", [])
        if not choices:
            raise ValueError(f"No choices in inference response: {resp}")
        raw_content = choices[0].get("message", {}).get("content", "")
        if not raw_content:
            raise ValueError("Empty content in inference response")
        raw_content = raw_content.strip()
        expression = self._extract_expression(raw_content)

        elapsed_ms = (time.monotonic() - start) * 1000

        return FormalTranslateResponse(
            claim_text=claim_text,
            formal_expression=expression,
            target_language=target_language,
            confidence=0.8,
            computation_time_ms=round(elapsed_ms, 2),
        )

    async def _translate_lean4(self, claim_text: str, tenant_id: str | None, start: float) -> FormalTranslateResponse:
        """Lean 4 translation: LLM → elaborate → retry on failure."""
        prompt = _LEAN4_TRANSLATE_PROMPT.format(claim_text=claim_text)
        resp = await self.inference_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="default",
            temperature=0.0,
            max_tokens=2048,
            tenant_id=tenant_id,
            schema_id="sp-formal-translate-lean4",
        )
        choices = resp.get("choices", [])
        if not choices:
            raise ValueError(f"No choices in inference response: {resp}")
        raw_content = choices[0].get("message", {}).get("content", "")
        if not raw_content:
            raise ValueError("Empty content in inference response")
        expression = self._extract_expression(raw_content.strip())

        # Without lean4 client, return LLM output directly
        if self.lean4_client is None:
            elapsed_ms = (time.monotonic() - start) * 1000
            return FormalTranslateResponse(
                claim_text=claim_text,
                formal_expression=expression,
                target_language="lean4",
                confidence=0.8,
                computation_time_ms=round(elapsed_ms, 2),
            )

        # Elaborate to type-check
        try:
            elab = await self.lean4_client.elaborate(expression)
        except httpx.HTTPError, OSError:
            logger.warning("Lean4 elaboration request failed", exc_info=True)
            elab = {"success": False, "errors": ["elaboration request failed"]}
        if elab.get("success"):
            elapsed_ms = (time.monotonic() - start) * 1000
            return FormalTranslateResponse(
                claim_text=claim_text,
                formal_expression=expression,
                target_language="lean4",
                confidence=0.9,
                computation_time_ms=round(elapsed_ms, 2),
            )

        # Retry once with error feedback
        raw_errors = elab.get("errors", [])
        errors: list[str] = [str(e) for e in raw_errors] if isinstance(raw_errors, list) else []
        retry_prompt = _LEAN4_RETRY_PROMPT.format(
            claim_text=claim_text,
            previous_code=expression,
            errors="; ".join(errors),
        )
        resp2 = await self.inference_client.chat_completion(
            messages=[{"role": "user", "content": retry_prompt}],
            model="default",
            temperature=0.0,
            max_tokens=2048,
            tenant_id=tenant_id,
            schema_id="sp-formal-translate-lean4",
        )
        choices2 = resp2.get("choices", [])
        if not choices2:
            raise ValueError(f"No choices in inference response: {resp2}")
        raw_content2 = choices2[0].get("message", {}).get("content", "")
        if not raw_content2:
            raise ValueError("Empty content in inference response")
        expression2 = self._extract_expression(raw_content2.strip())

        try:
            elab2 = await self.lean4_client.elaborate(expression2)
        except httpx.HTTPError, OSError:
            logger.warning("Lean4 retry elaboration request failed", exc_info=True)
            elab2 = {"success": False, "errors": ["elaboration request failed"]}
        if elab2.get("success"):
            elapsed_ms = (time.monotonic() - start) * 1000
            return FormalTranslateResponse(
                claim_text=claim_text,
                formal_expression=expression2,
                target_language="lean4",
                confidence=0.9,
                computation_time_ms=round(elapsed_ms, 2),
            )

        # Both attempts failed
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.warning("Lean4 elaboration failed after retry", errors=elab2.get("errors"))
        return FormalTranslateResponse(
            claim_text=claim_text,
            formal_expression=expression2,
            target_language="lean4",
            confidence=0.0,
            computation_time_ms=round(elapsed_ms, 2),
        )

    async def verify(
        self, formal_expression: str, language: str, *, tenant_id: str | None = None
    ) -> FormalVerifyResponse:
        """Route to Z3, SymPy, or Lean4 verifier based on language."""
        start = time.monotonic()

        if language == "z3":
            result = await self.z3_verifier.verify(formal_expression)
            proof_text = result.get("proof") or result.get("error")
            elapsed_ms = (time.monotonic() - start) * 1000
            return FormalVerifyResponse(
                expression=formal_expression,
                language=language,
                is_valid=bool(result["is_valid"]) if result.get("is_valid") is not None else None,
                counterexample=result.get("counterexample"),
                proof=proof_text,
                computation_time_ms=round(elapsed_ms, 2),
            )

        if language == "sympy":
            result = await self.sympy_verifier.verify(formal_expression)
            steps = result.get("proof_steps")
            proof_text = "\n".join(steps) if steps else result.get("error")
            elapsed_ms = (time.monotonic() - start) * 1000
            return FormalVerifyResponse(
                expression=formal_expression,
                language=language,
                is_valid=bool(result["is_valid"]) if result.get("is_valid") is not None else None,
                counterexample=result.get("counterexample"),
                proof=proof_text,
                computation_time_ms=round(elapsed_ms, 2),
            )

        if language == "lean4":
            return await self._verify_lean4(formal_expression, start)

        elapsed_ms = (time.monotonic() - start) * 1000
        return FormalVerifyResponse(
            expression=formal_expression,
            language=language,
            is_valid=None,
            proof=f"Unsupported language: {language}",
            computation_time_ms=round(elapsed_ms, 2),
        )

    async def _verify_lean4(self, lean_code: str, start: float) -> FormalVerifyResponse:
        """Verify Lean 4 code via lean4-prover check endpoint."""
        if self.lean4_client is None:
            elapsed_ms = (time.monotonic() - start) * 1000
            return FormalVerifyResponse(
                expression=lean_code,
                language="lean4",
                is_valid=None,
                proof="Lean4 prover not configured",
                computation_time_ms=round(elapsed_ms, 2),
            )

        check_result = await self.lean4_client.check(lean_code)
        elapsed_ms = (time.monotonic() - start) * 1000

        is_valid = check_result.get("success", False)
        errors = check_result.get("errors", [])
        messages = check_result.get("messages", [])

        return FormalVerifyResponse(
            expression=lean_code,
            language="lean4",
            is_valid=is_valid,
            proof="\n".join(messages) if messages else None,
            lean_elaboration={"errors": errors, "messages": messages},
            computation_time_ms=round(elapsed_ms, 2),
        )

    async def store_proof(self, claim_id: str, proof_data: dict[str, Any], tenant_id: str | None) -> str:
        """Store proof artifact to MinIO. Returns key."""
        if self.minio_client is None:
            raise RuntimeError("MinIO client not configured — cannot store proofs")

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        key = f"proofs/{claim_id}/{timestamp}.json"

        record = {
            "claim_id": claim_id,
            "key": key,
            "created_at": timestamp,
            "tenant_id": tenant_id,
            **proof_data,
        }

        await self.minio_client.upload(
            key=key,
            data=json.dumps(record).encode(),
            content_type="application/json",
        )
        logger.info("Stored proof", claim_id=claim_id, key=key)
        return key

    async def get_proofs(self, claim_id: str, *, tenant_id: str | None = None) -> list[dict[str, Any]]:
        """List proofs for a claim from MinIO."""
        if self.minio_client is None:
            return []

        prefix = f"proofs/{claim_id}/"
        keys = await self.minio_client.list_objects(prefix=prefix)

        proofs: list[dict[str, Any]] = []
        for key in keys:
            data = await self.minio_client.download(key)
            proofs.append(json.loads(data))

        return proofs

    @staticmethod
    def _extract_expression(raw: str) -> str:
        """Strip markdown code fences and whitespace from LLM output."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        return text
