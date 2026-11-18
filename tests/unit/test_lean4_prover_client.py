"""Tests for Lean4ProverClient — HTTP client for lean4-prover service."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from auth.tenant import license_id_var, license_token_var
from clients.lean4_prover import Lean4LicenceMissingError, Lean4ProverClient

pytestmark = pytest.mark.unit

BASE_URL = "http://lean4-prover:8812"
TENANT = "tenant-abc"

#: This client now builds its headers with ``kag_common.auth.outbound_headers``
#: (the fleet's single inter-service header contract) rather than a hand-rolled
#: dict, and that builder always sets ``Content-Type``. The wire is unchanged —
#: httpx already set this for every ``json=`` POST below — but the dict handed to
#: httpx, which is what these tests assert on, now carries it explicitly.
JSON_CT = {"Content-Type": "application/json"}

#: scientific-processor's own service licence (fallback) and an end tenant's
#: licence presented on the inbound request.
SVC_TOKEN = "svc-licence-token-sp"  # noqa: S105 - test fixture, not a credential
CALLER_TOKEN = "caller-licence-token-t1"  # noqa: S105 - test fixture, not a credential
#: Every lean4 call carries a Bearer licence in X-Tenant-Id; with no inbound
#: token it is the service licence.
SVC_HEADERS = {**JSON_CT, "X-Tenant-Id": f"Bearer {SVC_TOKEN}"}


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


def _inject_mock_client(client_obj: Any, response: MagicMock) -> AsyncMock:
    """Inject a mock httpx.AsyncClient into a client's _client attribute."""
    mock = AsyncMock()
    mock.get.return_value = response
    mock.post.return_value = response
    mock.is_closed = False
    client_obj._client = mock
    return mock


class TestLean4ProverClientInit:
    def test_strips_trailing_slash(self) -> None:
        c = Lean4ProverClient("http://lean4:8812/")
        assert c.base_url == "http://lean4:8812"

    def test_default_timeout(self) -> None:
        c = Lean4ProverClient(BASE_URL, service_license_token=SVC_TOKEN)
        assert c.timeout == 120.0

    def test_custom_timeout(self) -> None:
        c = Lean4ProverClient(BASE_URL, timeout=30.0)
        assert c.timeout == 30.0


class TestCheck:
    async def test_check_posts_to_correct_endpoint(self) -> None:
        resp = _mock_response({"success": True, "errors": [], "messages": ["OK"]})
        client = Lean4ProverClient(BASE_URL, service_license_token=SVC_TOKEN)
        mock = _inject_mock_client(client, resp)

        result = await client.check("theorem test : True := trivial")

        mock.post.assert_called_once_with(
            "/api/check",
            json={"source": "theorem test : True := trivial", "timeout": 30},
            headers=SVC_HEADERS,
        )
        assert result["success"] is True

    async def test_check_with_custom_timeout(self) -> None:
        resp = _mock_response({"success": True, "errors": [], "messages": []})
        client = Lean4ProverClient(BASE_URL, service_license_token=SVC_TOKEN)
        mock = _inject_mock_client(client, resp)

        await client.check("example : 1 + 1 = 2 := rfl", timeout=60)

        call_json = mock.post.call_args[1]["json"]
        assert call_json["timeout"] == 60

    async def test_check_with_tenant(self) -> None:
        resp = _mock_response({"success": True, "errors": [], "messages": []})
        client = Lean4ProverClient(BASE_URL, service_license_token=SVC_TOKEN)
        mock = _inject_mock_client(client, resp)

        await client.check("def x := 1", tenant_id=TENANT)

        call_headers = mock.post.call_args[1]["headers"]
        # The bare tenant is NOT sent: X-Tenant-Id carries the credential and
        # lean4-prover derives the tenant from introspection.
        assert call_headers == SVC_HEADERS

    async def test_check_raises_on_http_error(self) -> None:
        resp = _mock_response({}, status_code=500)
        resp.raise_for_status.side_effect = httpx.HTTPStatusError("Error", request=MagicMock(), response=resp)
        client = Lean4ProverClient(BASE_URL, service_license_token=SVC_TOKEN)
        mock = _inject_mock_client(client, resp)
        mock.post.return_value = resp

        with pytest.raises(httpx.HTTPStatusError):
            await client.check("bad code")


class TestElaborate:
    async def test_elaborate_posts_to_correct_endpoint(self) -> None:
        resp = _mock_response(
            {
                "success": True,
                "elaborated": "theorem test : True := trivial",
                "type_info": "True",
                "errors": [],
            }
        )
        client = Lean4ProverClient(BASE_URL, service_license_token=SVC_TOKEN)
        mock = _inject_mock_client(client, resp)

        result = await client.elaborate("theorem test : True := trivial")

        mock.post.assert_called_once_with(
            "/api/elaborate",
            json={"term": "theorem test : True := trivial", "timeout": 30},
            headers=SVC_HEADERS,
        )
        assert result["success"] is True
        assert result["elaborated"] == "theorem test : True := trivial"

    async def test_elaborate_failure(self) -> None:
        resp = _mock_response(
            {
                "success": False,
                "elaborated": None,
                "type_info": None,
                "errors": ["type mismatch"],
            }
        )
        client = Lean4ProverClient(BASE_URL, service_license_token=SVC_TOKEN)
        _inject_mock_client(client, resp)

        result = await client.elaborate("theorem bad : False := trivial")

        assert result["success"] is False
        assert "type mismatch" in result["errors"]


class TestProve:
    async def test_prove_posts_to_correct_endpoint(self) -> None:
        resp = _mock_response(
            {
                "success": True,
                "lean_code": "theorem test : 1 + 1 = 2 := by norm_num",
                "proof": "norm_num",
                "errors": [],
            }
        )
        client = Lean4ProverClient(BASE_URL, service_license_token=SVC_TOKEN)
        mock = _inject_mock_client(client, resp)

        result = await client.prove("1 plus 1 equals 2")

        mock.post.assert_called_once_with(
            "/api/prove",
            json={"theorem": "1 plus 1 equals 2", "tactics": ["auto"], "timeout": 60},
            headers=SVC_HEADERS,
        )
        assert result["success"] is True
        assert "norm_num" in result["lean_code"]

    async def test_prove_with_strategy_and_tenant(self) -> None:
        resp = _mock_response({"success": True, "lean_code": "...", "proof": "...", "errors": []})
        client = Lean4ProverClient(BASE_URL, service_license_token=SVC_TOKEN)
        mock = _inject_mock_client(client, resp)

        await client.prove("claim", strategy="omega", tenant_id=TENANT)

        call_json = mock.post.call_args[1]["json"]
        assert call_json["tactics"] == ["omega"]
        call_headers = mock.post.call_args[1]["headers"]
        # The bare tenant is NOT sent: X-Tenant-Id carries the credential and
        # lean4-prover derives the tenant from introspection.
        assert call_headers == SVC_HEADERS

    async def test_prove_failure(self) -> None:
        resp = _mock_response(
            {
                "success": False,
                "lean_code": None,
                "proof": None,
                "errors": ["unable to prove"],
            }
        )
        client = Lean4ProverClient(BASE_URL, service_license_token=SVC_TOKEN)
        _inject_mock_client(client, resp)

        result = await client.prove("impossible claim")

        assert result["success"] is False


class TestHealthCheck:
    async def test_health_check_success(self) -> None:
        resp = _mock_response({"status": "healthy", "service": "lean4-prover"})
        client = Lean4ProverClient(BASE_URL, service_license_token=SVC_TOKEN)
        mock = _inject_mock_client(client, resp)

        result = await client.health_check()

        mock.get.assert_called_once_with("/health")
        assert result["status"] == "healthy"

    async def test_health_check_returns_unavailable_on_error(self) -> None:
        client = Lean4ProverClient(BASE_URL, service_license_token=SVC_TOKEN)
        mock = AsyncMock()
        mock.get.side_effect = httpx.ConnectError("Connection refused")
        mock.is_closed = False
        client._client = mock

        result = await client.health_check()

        assert result == {"status": "unavailable", "service": "lean4-prover"}

    # -- close ---------------------------------------------------------------

    async def test_close_closes_client(self) -> None:
        client = Lean4ProverClient(BASE_URL, service_license_token=SVC_TOKEN)
        mock = AsyncMock()
        mock.is_closed = False
        client._client = mock

        await client.close()

        mock.aclose.assert_called_once()
        assert client._client is None


class TestLicenceIsAlwaysSent:
    """Owner ruling: every GPU-backend call carries a licence. lean4-prover 401s
    tokenless calls, so a missing Bearer here is an outage, not a degradation."""

    async def test_service_licence_is_the_fallback(self) -> None:
        client = Lean4ProverClient(BASE_URL, service_license_token=SVC_TOKEN)
        mock = _inject_mock_client(client, _mock_response({"valid": True}))

        await client.check("def x := 1", tenant_id=TENANT)

        assert mock.post.call_args[1]["headers"]["X-Tenant-Id"] == f"Bearer {SVC_TOKEN}"

    async def test_end_tenant_token_wins_over_service_licence(self) -> None:
        """The caller's own licence is forwarded so lean4 attributes the work to
        THAT tenant, not to scientific-processor's service tenant."""
        client = Lean4ProverClient(BASE_URL, service_license_token=SVC_TOKEN)
        mock = _inject_mock_client(client, _mock_response({"valid": True}))

        tok = license_token_var.set(CALLER_TOKEN)
        lic = license_id_var.set("lic-caller")
        try:
            await client.prove("claim", tenant_id=TENANT)
        finally:
            license_id_var.reset(lic)
            license_token_var.reset(tok)

        headers = mock.post.call_args[1]["headers"]
        assert headers["X-Tenant-Id"] == f"Bearer {CALLER_TOKEN}"
        assert headers["X-License-Id"] == "lic-caller"

    async def test_end_tenant_token_works_without_a_service_licence(self) -> None:
        client = Lean4ProverClient(BASE_URL)
        mock = _inject_mock_client(client, _mock_response({"valid": True}))

        tok = license_token_var.set(CALLER_TOKEN)
        try:
            await client.elaborate("def x := 1")
        finally:
            license_token_var.reset(tok)

        assert mock.post.call_args[1]["headers"]["X-Tenant-Id"] == f"Bearer {CALLER_TOKEN}"

    @pytest.mark.parametrize("blank", [None, "", "   "])
    async def test_no_token_at_all_raises_before_the_network(self, blank: str | None) -> None:
        client = Lean4ProverClient(BASE_URL, service_license_token=blank)
        mock = _inject_mock_client(client, _mock_response({"valid": True}))

        with pytest.raises(Lean4LicenceMissingError):
            await client.check("def x := 1", tenant_id=TENANT)

        mock.post.assert_not_called()

    async def test_bare_tenant_is_never_sent_as_identity(self) -> None:
        client = Lean4ProverClient(BASE_URL, service_license_token=SVC_TOKEN)
        mock = _inject_mock_client(client, _mock_response({"valid": True}))

        await client.check("def x := 1", tenant_id=TENANT)

        assert TENANT not in mock.post.call_args[1]["headers"].values()
