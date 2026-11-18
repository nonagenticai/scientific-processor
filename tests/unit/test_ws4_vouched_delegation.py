"""WS-4: only a VOUCHED tenant may be delegated to the metered modality-gateway.

THE DEFECT THESE TESTS REPRODUCE
--------------------------------
scientific-processor's SA subject ``019f83f2-ddd1-779f-9c70-7a0c6de1dacb`` is on
the modality-gateway's ``AUTH_DELEGATING_SUBJECTS``, so when this service sends
``X-Effective-Tenant-Id: <T>`` the gateway ACTS AS tenant T — it resolves T's own
BYO LLM provider credentials and BILLS T.

``AuthMiddleware``'s trusted-network branch accepts a BARE, uncredentialed
``X-Tenant-Id`` from any peer inside ``TRUSTED_CIDRS`` (in production the whole
pod CIDR), and ``clients.pytorch_backend`` forwarded that value straight into
``X-Effective-Tenant-Id``. So any unauthenticated pod could name a victim tenant
and spend the victim's credentials.

WHAT IS AND IS NOT ASSERTED HERE
--------------------------------
The tenant is still fully ACCEPTED as a data scope on every route — that is an
explicit availability control below (``TestNothingIsNewlyRejected``), because
gating *acceptance* would 401 every live in-cluster caller. Only the DELEGATION
is gated, and an unvouched delegation is OMITTED (not refused) — see the
``clients/pytorch_backend`` module docstring for why raising would strand the
SubgraphX background job instead of producing a clean 403.

These run the REAL ``AuthMiddleware`` and a REAL ``httpx`` transport, so the
headers under assertion are the ones that would actually go on the wire.
"""

# NB: deliberately NO module-level `os.environ[...] = ...` here. Every app in
# this file is a bare `FastAPI()` with `AuthMiddleware` added directly, so it
# never consults ENABLE_AUTH — and setting it at import time leaks into sibling
# test modules that DO read it, whose behaviour then depends on pytest
# collection order (see the `_auth_disabled_settings` note in
# tests/unit/test_cache_router.py for the same trap).
import asyncio
import logging
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import auth.vouch as vouch_mod
from auth.middleware import AuthMiddleware
from auth.tenant import (
    SOURCE_LICENSE_TOKEN,
    SOURCE_NONE,
    SOURCE_TRUSTED_NETWORK_DEFAULT,
    SOURCE_UNVOUCHED_HEADER,
    SOURCE_VOUCHED_ONBEHALF,
    VOUCHED_TENANT_SOURCES,
    may_delegate,
    tenant_source_var,
)
from clients.pytorch_backend import EFFECTIVE_TENANT_HEADER, PyTorchBackendClient, delegation_headers

pytestmark = pytest.mark.unit

#: A tenant UUID an attacker simply made up. It belongs to nobody the attacker
#: can prove anything about — that is the entire point.
FORGED_TENANT = "0195f48b-dead-7000-8000-000000000666"
#: A tenant that arrived the legitimate way (behind an introspected license).
REAL_TENANT = "0195f48b-d000-7000-8000-000000000042"
DOWNSTREAM_TOKEN = "lic-svc-scientific-processor"  # noqa: S105 - test fixture, not a credential
BASE_URL = "http://modality-gateway.example:8080"
#: A REAL loopback peer. ``auth.middleware._is_trusted_network`` no longer trusts
#: Starlette's default ``"testclient"`` sentinel, so a test that wants the
#: trusted-network branch must present a peer production would also trust —
#: which is exactly the branch this file exists to exercise.
TRUSTED_PEER = ("127.0.0.1", 50000)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _capturing_client(sent: list[httpx.Request]) -> PyTorchBackendClient:
    """A real PyTorchBackendClient in gateway mode over a real MockTransport.

    Nothing about header construction is stubbed: the request recorded in
    ``sent`` is the one httpx would have put on the wire.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"embeddings": [[0.1]], "predictions": [[0.2]]})

    client = PyTorchBackendClient(BASE_URL, downstream_license_token=DOWNSTREAM_TOKEN)
    client._client = httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    return client


def _app_that_calls_the_gateway(sent: list[httpx.Request]) -> FastAPI:
    """The real middleware in front of a route that makes the delegated hop."""
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.post("/api/explain/embed")
    async def embed(request: Request) -> dict[str, Any]:
        client = _capturing_client(sent)
        try:
            await client.get_embeddings(["text"], tenant_id=str(request.state.tenant_id))
        finally:
            await client.close()
        return {"tenant_seen_as_data_scope": request.state.tenant_id}

    return app


def _effective_tenant(request: httpx.Request) -> str | None:
    return request.headers.get(EFFECTIVE_TENANT_HEADER)


# ---------------------------------------------------------------------------
# THE EXPLOIT
# ---------------------------------------------------------------------------


class TestForgedTenantIsNotDelegated:
    """The reproduction: a fabricated tenant, no credential of any kind."""

    def test_bare_forged_tenant_is_never_delegated_to_the_metered_gateway(self) -> None:
        """A pod-CIDR peer names a victim tenant with NO credential.

        Before WS-4 this stamped ``X-Effective-Tenant-Id: <forged>``, the gateway
        logged ``effective-tenant delegation ... acting as tenant=<forged>`` and
        200'd the metered call against the VICTIM's BYO provider credentials.
        """
        sent: list[httpx.Request] = []
        with TestClient(_app_that_calls_the_gateway(sent), client=TRUSTED_PEER) as tc:
            response = tc.post("/api/explain/embed", headers={"X-Tenant-Id": FORGED_TENANT})

        assert response.status_code == 200, "the call must still succeed — we omit, we do not refuse"
        assert len(sent) == 1
        assert _effective_tenant(sent[0]) is None, (
            "a tenant nobody vouched for was delegated to the metered gateway — this is the credential-theft path"
        )
        # The SA Bearer is still present, so the hop authenticates as this
        # service and bills THIS service's own tenant. Fail-safe, not fail-open.
        assert sent[0].headers["X-Tenant-Id"] == f"Bearer {DOWNSTREAM_TOKEN}"

    def test_gnn_inference_is_gated_on_the_same_provenance(self) -> None:
        """The second delegated route. Reached from the SubgraphX background job."""
        sent: list[httpx.Request] = []
        client = _capturing_client(sent)
        token = tenant_source_var.set(SOURCE_UNVOUCHED_HEADER)
        try:
            asyncio.run(client.gnn_inference([[1.0]], [[0]], tenant_id=FORGED_TENANT))
        finally:
            tenant_source_var.reset(token)
        assert _effective_tenant(sent[0]) is None


# ---------------------------------------------------------------------------
# FALSE-POSITIVE CONTROLS — a legitimate delegation still happens
# ---------------------------------------------------------------------------


class TestProvenTenantsAreStillDelegated:
    """The guard must not be a blanket "never delegate" — that would be trivially
    passable by deleting the feature. Each control uses the SAME harness."""

    @pytest.mark.parametrize("source", sorted(VOUCHED_TENANT_SOURCES))
    def test_a_proven_provenance_is_delegated(self, source: str) -> None:
        sent: list[httpx.Request] = []
        client = _capturing_client(sent)
        token = tenant_source_var.set(source)
        try:
            asyncio.run(client.get_embeddings(["text"], tenant_id=REAL_TENANT))
        finally:
            tenant_source_var.reset(token)
        assert _effective_tenant(sent[0]) == REAL_TENANT, (
            f"provenance {source!r} is in VOUCHED_TENANT_SOURCES and MUST still delegate"
        )

    def test_license_token_request_delegates_end_to_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Through the REAL middleware: an introspected license token delegates.

        The recovered draft of this test patched ``AuthMiddleware._introspect``
        with ``raising=False``. No such method exists — introspection is the
        MODULE-LEVEL ``_introspect_license`` — so the patch silently no-op'd, the
        request 401'd, and a ``pytest.skip`` on non-200 turned the miss into a
        green skip. That made the single most important false-positive control in
        this file structurally incapable of failing. Patch the real symbol and
        assert the 200 instead of skipping on it.
        """
        import auth.middleware as mw

        assert hasattr(mw, "_introspect_license"), "introspection entrypoint moved — this control would go blind"

        async def fake_introspect(token: str) -> dict[str, Any]:
            return {"active": True, "tenant_id": REAL_TENANT, "license_id": "lic-1", "roles": []}

        monkeypatch.setattr(mw, "_introspect_license", fake_introspect)

        sent: list[httpx.Request] = []
        with TestClient(_app_that_calls_the_gateway(sent), client=TRUSTED_PEER) as tc:
            response = tc.post("/api/explain/embed", headers={"X-Tenant-Id": "Bearer real-customer-token"})

        assert response.status_code == 200, response.text
        assert _effective_tenant(sent[0]) == REAL_TENANT


class TestNothingIsNewlyRejected:
    """Availability control: a bare header is still a fully accepted DATA SCOPE.

    Gating ACCEPTANCE rather than delegation would have 401'd every live
    in-cluster caller, which is exactly why this axis is pinned separately.
    """

    def test_bare_tenant_header_is_still_accepted_and_scopes_the_request(self) -> None:
        sent: list[httpx.Request] = []
        with TestClient(_app_that_calls_the_gateway(sent), client=TRUSTED_PEER) as tc:
            response = tc.post("/api/explain/embed", headers={"X-Tenant-Id": FORGED_TENANT})
        assert response.status_code == 200
        assert response.json()["tenant_seen_as_data_scope"] == FORGED_TENANT

    def test_non_gateway_mode_is_byte_identical(self) -> None:
        """Token unset => raw X-Tenant-Id, no delegation header exists at all.

        The direct backend treats the tenant as a data scope: no BYO credential
        is selected and no third party is billed, so provenance does not gate it
        and the pre-flip path loses no attribution.
        """
        sent: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            return httpx.Response(200, json={"embeddings": []})

        client = PyTorchBackendClient(BASE_URL)  # no downstream token
        client._client = httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))
        asyncio.run(client.get_embeddings(["text"], tenant_id=FORGED_TENANT))

        assert sent[0].headers["X-Tenant-Id"] == FORGED_TENANT
        assert _effective_tenant(sent[0]) is None


# ---------------------------------------------------------------------------
# The provenance vocabulary itself
# ---------------------------------------------------------------------------


class TestProvenanceLadder:
    """Every non-proven provenance must fail closed."""

    @pytest.mark.parametrize(
        "source",
        [SOURCE_UNVOUCHED_HEADER, SOURCE_TRUSTED_NETWORK_DEFAULT, SOURCE_NONE, "", "made-up", "auth-disabled"],
    )
    def test_unproven_provenance_may_not_delegate(self, source: str) -> None:
        assert may_delegate(source) is False
        assert delegation_headers(tenant_id=REAL_TENANT, tenant_source=source) == {}

    @pytest.mark.parametrize("source", [SOURCE_LICENSE_TOKEN, SOURCE_VOUCHED_ONBEHALF])
    def test_proven_provenance_may_delegate(self, source: str) -> None:
        assert may_delegate(source) is True
        assert delegation_headers(tenant_id=REAL_TENANT, tenant_source=source) == {EFFECTIVE_TENANT_HEADER: REAL_TENANT}

    def test_contextvar_default_is_not_delegatable(self) -> None:
        """An unbound context — startup probe, import-time worker, a task spawned
        from a scope that never had a request — reads SOURCE_NONE. Fail closed."""
        assert tenant_source_var.get() == SOURCE_NONE
        assert may_delegate(tenant_source_var.get()) is False

    def test_empty_tenant_never_produces_a_header(self) -> None:
        assert delegation_headers(tenant_id="", tenant_source=SOURCE_LICENSE_TOKEN) == {}

    def test_tenant_source_is_required_with_no_default(self) -> None:
        """A new call site cannot silently re-open the hole: omitting the
        provenance is a TypeError, not a permissive default."""
        with pytest.raises(TypeError):
            delegation_headers(tenant_id=REAL_TENANT)  # type: ignore[call-arg]

    def test_omission_is_logged_with_a_greppable_token(self, caplog: pytest.LogCaptureFixture) -> None:
        """scientific-processor exposes no /metrics endpoint, so this WARN — not a
        Prometheus counter — is the SCIENTIFIC_PROCESSOR_ONBEHALF_SUBJECTS
        rollout burn-down signal. Keep the token stable."""
        with caplog.at_level(logging.WARNING):
            delegation_headers(tenant_id=FORGED_TENANT, tenant_source=SOURCE_UNVOUCHED_HEADER)
        assert "WS4_DELEGATION omitted" in caplog.text
        assert FORGED_TENANT in caplog.text
        assert SOURCE_UNVOUCHED_HEADER in caplog.text


# ---------------------------------------------------------------------------
# Provenance recorded by the real middleware
# ---------------------------------------------------------------------------


class TestMiddlewareRecordsProvenance:
    """The middleware is the only writer of the provenance."""

    @staticmethod
    def _probe_app() -> FastAPI:
        app = FastAPI()
        app.add_middleware(AuthMiddleware)

        # MUST sit under a PROTECTED_PREFIXES path: AuthMiddleware short-circuits
        # anything that is neither public nor protected, so a bare "/api/probe"
        # would never reach the identity binding at all and the probe would
        # silently assert nothing.
        @app.get("/api/topology/probe")
        async def probe(request: Request) -> dict[str, Any]:
            # Read from BOTH writers — they must never drift.
            return {
                "state": getattr(request.state, "tenant_source", None),
                "ctxvar": tenant_source_var.get(),
                "tenant": request.state.tenant_id,
            }

        return app

    def test_bare_header_is_unvouched(self) -> None:
        with TestClient(self._probe_app(), client=TRUSTED_PEER) as tc:
            body = tc.get("/api/topology/probe", headers={"X-Tenant-Id": FORGED_TENANT}).json()
        assert body["state"] == SOURCE_UNVOUCHED_HEADER
        assert body["ctxvar"] == SOURCE_UNVOUCHED_HEADER
        assert body["tenant"] == FORGED_TENANT

    def test_no_header_from_a_trusted_peer_is_the_services_own_default(self) -> None:
        with TestClient(self._probe_app(), client=TRUSTED_PEER) as tc:
            body = tc.get("/api/topology/probe").json()
        assert body["state"] == SOURCE_TRUSTED_NETWORK_DEFAULT
        assert may_delegate(body["state"]) is False


# ---------------------------------------------------------------------------
# Vouching — the ONLY way a bare tenant becomes delegatable
# ---------------------------------------------------------------------------


class TestVouchFailsClosed:
    """Every missing link in the vouch chain must yield "not vouched"."""

    @staticmethod
    def _request(headers: dict[str, str] | None = None) -> Any:
        from starlette.requests import Request as StarletteRequest

        raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
        return StarletteRequest({"type": "http", "method": "GET", "path": "/", "headers": raw})

    def test_empty_allowlist_vouches_for_nobody(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The DEFAULT. Unset ⇒ vouching off ⇒ nothing is ever promoted."""
        monkeypatch.delenv(vouch_mod.ONBEHALF_SUBJECTS_ENV, raising=False)
        assert vouch_mod.onbehalf_subjects() == frozenset()
        assert asyncio.run(vouch_mod.authorized_onbehalf(self._request({"Authorization": "Bearer t"}))) is False

    def test_allowlist_set_but_introspect_url_unset_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(vouch_mod.ONBEHALF_SUBJECTS_ENV, "svc-a")
        monkeypatch.delenv("LICENSE_INTROSPECT_URL", raising=False)
        assert asyncio.run(vouch_mod.authorized_onbehalf(self._request({"Authorization": "Bearer t"}))) is False

    @pytest.mark.parametrize(
        "headers",
        [{}, {"Authorization": "Bearer"}, {"Authorization": "Bearer "}, {"Authorization": "Basic abc"}],
    )
    def test_missing_or_malformed_authorization_fails_closed(
        self, headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(vouch_mod.ONBEHALF_SUBJECTS_ENV, "svc-a")
        monkeypatch.setenv("LICENSE_INTROSPECT_URL", "http://licenses.example/api/introspect")
        assert asyncio.run(vouch_mod.authorized_onbehalf(self._request(headers))) is False

    @pytest.mark.parametrize(
        "payload",
        [
            None,  # transport fault / licenses service down
            {"active": False, "subject_id": "svc-a"},  # revoked
            {"active": True},  # no subject at all
            {"active": True, "subject_id": ""},  # empty subject
            {"active": True, "subject_id": "svc-NOT-on-the-list"},  # unlisted
        ],
    )
    def test_introspection_ladder_fails_closed(self, payload: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(vouch_mod.ONBEHALF_SUBJECTS_ENV, "svc-a")
        monkeypatch.setenv("LICENSE_INTROSPECT_URL", "http://licenses.example/api/introspect")

        async def fake(token: str) -> Any:
            return payload

        monkeypatch.setattr(vouch_mod, "introspect_license", fake)
        assert asyncio.run(vouch_mod.authorized_onbehalf(self._request({"Authorization": "Bearer t"}))) is False

    def test_an_allowlisted_active_subject_vouches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The GREEN half — without it "always False" would pass every test above."""
        monkeypatch.setenv(vouch_mod.ONBEHALF_SUBJECTS_ENV, "svc-a, svc-b")
        monkeypatch.setenv("LICENSE_INTROSPECT_URL", "http://licenses.example/api/introspect")

        async def fake(token: str) -> Any:
            return {"active": True, "subject_id": "svc-b"}

        monkeypatch.setattr(vouch_mod, "introspect_license", fake)
        assert asyncio.run(vouch_mod.authorized_onbehalf(self._request({"Authorization": "Bearer t"}))) is True

    def test_a_vouched_request_promotes_the_bare_tenant_end_to_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Through the REAL middleware: allowlisted vouch ⇒ the SAME bare tenant
        is delegated. This is the acceptance criterion for the gitops rollout."""
        monkeypatch.setenv(vouch_mod.ONBEHALF_SUBJECTS_ENV, "svc-caller")
        monkeypatch.setenv("LICENSE_INTROSPECT_URL", "http://licenses.example/api/introspect")

        async def fake(token: str) -> Any:
            return {"active": True, "subject_id": "svc-caller"}

        monkeypatch.setattr(vouch_mod, "introspect_license", fake)

        sent: list[httpx.Request] = []
        with TestClient(_app_that_calls_the_gateway(sent), client=TRUSTED_PEER) as tc:
            response = tc.post(
                "/api/explain/embed",
                headers={"X-Tenant-Id": REAL_TENANT, "Authorization": "Bearer svc-caller-token"},
            )
        assert response.status_code == 200
        assert _effective_tenant(sent[0]) == REAL_TENANT

    def test_a_failed_vouch_leaves_the_request_working(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Vouching can only WIDEN: an unrelated Authorization header (e.g. a
        forwarded Keycloak JWT) must not reject the caller."""
        monkeypatch.setenv(vouch_mod.ONBEHALF_SUBJECTS_ENV, "svc-caller")
        monkeypatch.setenv("LICENSE_INTROSPECT_URL", "http://licenses.example/api/introspect")

        async def fake(token: str) -> Any:
            return {"active": True, "subject_id": "some-other-service"}

        monkeypatch.setattr(vouch_mod, "introspect_license", fake)

        sent: list[httpx.Request] = []
        with TestClient(_app_that_calls_the_gateway(sent), client=TRUSTED_PEER) as tc:
            response = tc.post(
                "/api/explain/embed",
                headers={"X-Tenant-Id": FORGED_TENANT, "Authorization": "Bearer an-end-user-jwt"},
            )
        assert response.status_code == 200
        assert _effective_tenant(sent[0]) is None


# ---------------------------------------------------------------------------
# Structural guards
# ---------------------------------------------------------------------------


class TestStructuralGuards:
    def test_provenance_reaches_a_fire_and_forget_background_task(self) -> None:
        """SubgraphXService.explain spawns _run_explanation via asyncio.create_task
        and returns; the request scope is then reset. create_task snapshots the
        context AT CREATION, so the job still observes its originating
        provenance — measured here, not assumed, because the whole guard would be
        a no-op on that path otherwise."""
        seen: dict[str, str] = {}

        async def job(tag: str) -> None:
            await asyncio.sleep(0.01)  # outlive the scope
            seen[tag] = tenant_source_var.get()

        async def main() -> None:
            token = tenant_source_var.set(SOURCE_LICENSE_TOKEN)
            spawned_in_scope = asyncio.create_task(job("in-scope"))
            tenant_source_var.reset(token)  # the request ends here
            spawned_after = asyncio.create_task(job("after-reset"))
            await asyncio.gather(spawned_in_scope, spawned_after)

        asyncio.run(main())
        assert seen["in-scope"] == SOURCE_LICENSE_TOKEN, "the guard would be inert in the background job"
        assert seen["after-reset"] == SOURCE_NONE, "a task with no request context must fail closed"

    def test_delegation_headers_is_the_only_emitter_of_the_header(self) -> None:
        """Grep-guard. If a future call site writes X-Effective-Tenant-Id directly
        it bypasses the provenance check entirely, so pin that structurally."""
        import pathlib

        src = pathlib.Path(__file__).resolve().parents[2] / "src"
        offenders = []
        for path in src.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), 1):
                if EFFECTIVE_TENANT_HEADER in line and path.name != "pytorch_backend.py":
                    # Docstrings elsewhere may NAME the header; only assignments matter.
                    if "=" in line and "#" not in line.split(EFFECTIVE_TENANT_HEADER)[0]:
                        offenders.append(f"{path.relative_to(src)}:{lineno}: {line.strip()}")
        assert offenders == [], (
            "X-Effective-Tenant-Id must only be emitted by clients.pytorch_backend.delegation_headers, "
            f"which is the only place provenance is checked. Found: {offenders}"
        )
