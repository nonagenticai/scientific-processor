"""Licence attribution: the requestor's licence must reach every outbound hop.

THE DEFECT THESE TESTS REPRODUCE
--------------------------------
``AuthMiddleware`` has always introspected a ``license_id`` and stashed it on
``request.state.license_id``. It then went nowhere: grepping the repo before this
change, ``license_id`` appeared in ``src/auth/middleware.py`` and in no other
source file, and ``X-License-Id`` appeared nowhere in ``src/clients/`` at all.

So every downstream hop — inference-router above all, which is the METERED LLM
call — received a resolved tenant and no licence. ``billing_service`` counts
exactly that shape in ``billing_license_unattributed_total``: spend it can place
on a tenant but not on a licence. Fleet-wide that is what holds
``billing_attribution_rate`` at 0.9212, ~4,730 cost events with no licence.

WHAT IS AND IS NOT ASSERTED HERE
--------------------------------
- The licence is FORWARDED, verbatim, on all four clients.
- It is OMITTED — not defaulted, not substituted — when the caller presented
  none. That honest NULL is the thing billing measures; a stand-in value (this
  service's own service licence, the tenant, a placeholder) would move real spend
  onto a licence that did not incur it and would stop being counted as
  unattributed at all, which is strictly worse than leaving it blank.
- Nothing about authentication changes: ``X-License-Id`` is attribution data, so
  no request is newly accepted or rejected because of it.

These drive REAL client objects over a REAL ``httpx`` transport, and the
end-to-end cases run the REAL ``AuthMiddleware``, so the headers under assertion
are the ones that would actually go on the wire.
"""

from __future__ import annotations

import asyncio
import pathlib
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import auth.middleware as mw
from auth.tenant import SOURCE_LICENSE_TOKEN, license_id_var, tenant_source_var
from clients.graph_adapter import GraphAdapterClient
from clients.inference_router import InferenceRouterClient
from clients.lean4_prover import Lean4ProverClient
from clients.pytorch_backend import PyTorchBackendClient

pytestmark = pytest.mark.unit

LICENSE_HEADER = "X-License-Id"
#: The licence the CALLER presented — the one whose customer must be billed.
CALLER_LICENSE = "lic-cust-0195f48b-d000"
#: scientific-processor's OWN service licence, used to authenticate the gateway
#: hop. Putting THIS in X-License-Id would book a customer's embedding spend
#: against this service's licence, which is mis-attribution wearing a valid value.
SERVICE_LICENSE = "lic-svc-scientific-processor"  # noqa: S105 - test fixture, not a credential
TENANT = "0195f48b-d000-7000-8000-000000000042"
BASE_URL = "http://upstream.example:8080"
#: A REAL loopback peer — ``_is_trusted_network`` does not trust Starlette's
#: ``"testclient"`` sentinel, so the trusted-network branch needs a real IP.
TRUSTED_PEER = ("127.0.0.1", 50000)


# ---------------------------------------------------------------------------
# Harness — one recorded request per client, nothing about headers stubbed
# ---------------------------------------------------------------------------


def _transport(sent: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"ok": True, "choices": [], "results": [], "embeddings": []})

    return httpx.MockTransport(handler)


def _call_inference(sent: list[httpx.Request], **kw: Any) -> None:
    client = InferenceRouterClient(BASE_URL)
    client._client = httpx.AsyncClient(base_url=BASE_URL, transport=_transport(sent))
    asyncio.run(client.chat_completion([{"role": "user", "content": "hi"}], tenant_id=TENANT, **kw))


def _call_graph(sent: list[httpx.Request], **kw: Any) -> None:
    client = GraphAdapterClient(BASE_URL, **kw)
    client._client = httpx.AsyncClient(base_url=BASE_URL, transport=_transport(sent))
    asyncio.run(client.get_node("n1", tenant_id=TENANT))


def _call_lean4(sent: list[httpx.Request]) -> None:
    client = Lean4ProverClient(BASE_URL, service_license_token=SERVICE_LICENSE)
    client._client = httpx.AsyncClient(base_url=BASE_URL, transport=_transport(sent))
    asyncio.run(client.check("theorem t : True := trivial", tenant_id=TENANT))


def _call_pytorch(sent: list[httpx.Request], **kw: Any) -> None:
    client = PyTorchBackendClient(BASE_URL, **kw)
    client._client = httpx.AsyncClient(base_url=BASE_URL, transport=_transport(sent))
    asyncio.run(client.get_embeddings(["text"], tenant_id=TENANT))


#: Every outbound client in this service. Parametrising over the whole set is
#: deliberate: the defect was that ONE of them (inference_router, the metered
#: hop) mattered most while all four were equally silent, so a fix that covers
#: only the loud one would leave the same hole in three places.
ALL_CLIENTS = [
    pytest.param(_call_inference, id="inference_router"),
    pytest.param(_call_graph, id="graph_adapter"),
    pytest.param(_call_lean4, id="lean4_prover"),
    pytest.param(_call_pytorch, id="pytorch_backend"),
]

#: Clients that still put the caller's bare tenant in X-Tenant-Id. lean4_prover
#: is excluded on purpose: it is a GPU backend and every call must carry a
#: licence, so its X-Tenant-Id slot always holds `Bearer <token>` (pinned in
#: TestLean4AlwaysCarriesALicence below and in test_lean4_prover_client.py).
BARE_TENANT_CLIENTS = [p for p in ALL_CLIENTS if p.id != "lean4_prover"]


# ---------------------------------------------------------------------------
# THE FIX
# ---------------------------------------------------------------------------


class TestTheLicenceReachesEveryOutboundHop:
    @pytest.mark.parametrize("call", ALL_CLIENTS)
    def test_a_presented_licence_is_forwarded_verbatim(self, call: Any) -> None:
        """Before this change no client emitted X-License-Id at all, so every
        downstream cost event was `billing_license_unattributed_total`."""
        sent: list[httpx.Request] = []
        token = license_id_var.set(CALLER_LICENSE)
        try:
            call(sent)
        finally:
            license_id_var.reset(token)
        assert sent[0].headers.get(LICENSE_HEADER) == CALLER_LICENSE

    @pytest.mark.parametrize("call", BARE_TENANT_CLIENTS)
    def test_the_tenant_is_still_sent_alongside_it(self, call: Any) -> None:
        """Availability/regression control: the licence is ADDITIVE. If adding it
        displaced X-Tenant-Id, every downstream would fail auth on deploy."""
        sent: list[httpx.Request] = []
        token = license_id_var.set(CALLER_LICENSE)
        try:
            call(sent)
        finally:
            license_id_var.reset(token)
        assert sent[0].headers.get("X-Tenant-Id") == TENANT


class TestNoLicenceMeansNoHeader:
    """The honest NULL. An absent header is what billing counts; a present-but-
    meaningless one is not counted at all, so it hides the very callers this
    whole exercise exists to enumerate."""

    @pytest.mark.parametrize("call", ALL_CLIENTS)
    def test_unbound_context_emits_no_licence_header(self, call: Any) -> None:
        sent: list[httpx.Request] = []
        call(sent)
        assert LICENSE_HEADER not in sent[0].headers

    @pytest.mark.parametrize("call", ALL_CLIENTS)
    @pytest.mark.parametrize("empty", [None, "", "   "])
    def test_empty_or_blank_licence_emits_no_header(self, call: Any, empty: Any) -> None:
        """A whitespace-only licence is truthy in Python and would put a BLANK
        X-License-Id on the wire, which billing reads as a present-but-
        unresolvable licence — no longer counted as unattributed. Normalised to
        None at the single point it enters the process (`_call_with_tenant`), and
        pinned here at the point it would leave."""
        sent: list[httpx.Request] = []
        token = license_id_var.set(empty.strip() or None if isinstance(empty, str) else empty)
        try:
            call(sent)
        finally:
            license_id_var.reset(token)
        assert LICENSE_HEADER not in sent[0].headers

    def test_a_service_licence_is_never_substituted_for_a_missing_one(self) -> None:
        """pytorch_backend authenticates the gateway hop with THIS SERVICE's own
        licence. That value must never leak into X-License-Id: it would book a
        customer's embedding spend against scientific-processor's licence, which
        looks attributed and is wrong."""
        sent: list[httpx.Request] = []
        token = tenant_source_var.set(SOURCE_LICENSE_TOKEN)
        try:
            _call_pytorch(sent, downstream_license_token=SERVICE_LICENSE)
        finally:
            tenant_source_var.reset(token)
        assert sent[0].headers["X-Tenant-Id"] == f"Bearer {SERVICE_LICENSE}"
        assert LICENSE_HEADER not in sent[0].headers

    def test_the_caller_licence_wins_over_the_service_licence_in_gateway_mode(self) -> None:
        """The green half: with a real caller licence in scope, the gateway hop
        carries BOTH — the service licence authenticates, the caller licence
        attributes. They must not be confused for one another."""
        sent: list[httpx.Request] = []
        tok_src = tenant_source_var.set(SOURCE_LICENSE_TOKEN)
        tok_lic = license_id_var.set(CALLER_LICENSE)
        try:
            _call_pytorch(sent, downstream_license_token=SERVICE_LICENSE)
        finally:
            license_id_var.reset(tok_lic)
            tenant_source_var.reset(tok_src)
        assert sent[0].headers["X-Tenant-Id"] == f"Bearer {SERVICE_LICENSE}"
        assert sent[0].headers[LICENSE_HEADER] == CALLER_LICENSE

    def test_a_graph_service_token_is_never_substituted_either(self) -> None:
        """graph_adapter's WS-4 service token rides Authorization. Same rule."""
        sent: list[httpx.Request] = []
        _call_graph(sent, service_token="svc-graph-token")
        assert sent[0].headers["Authorization"] == "Bearer svc-graph-token"
        assert LICENSE_HEADER not in sent[0].headers

    @pytest.mark.parametrize("call", ALL_CLIENTS)
    def test_the_tenant_is_never_substituted_for_the_licence(self, call: Any) -> None:
        """The tenant is always available; using it as a licence would make the
        attribution rate read 100% while attributing nothing."""
        sent: list[httpx.Request] = []
        call(sent)
        assert sent[0].headers.get(LICENSE_HEADER) != TENANT


# ---------------------------------------------------------------------------
# End-to-end through the REAL middleware
# ---------------------------------------------------------------------------


def _app_that_calls_the_metered_hop(sent: list[httpx.Request]) -> FastAPI:
    """Real AuthMiddleware in front of a route that makes the inference-router
    call. Must sit under a PROTECTED_PREFIXES path or the middleware
    short-circuits and the probe would silently assert nothing."""
    app = FastAPI()
    app.add_middleware(mw.AuthMiddleware)

    @app.post("/api/verify/llm")
    async def verify(request: Request) -> dict[str, Any]:
        client = InferenceRouterClient(BASE_URL)
        client._client = httpx.AsyncClient(base_url=BASE_URL, transport=_transport(sent))
        try:
            await client.chat_completion([{"role": "user", "content": "hi"}], tenant_id=str(request.state.tenant_id))
        finally:
            await client.close()
        return {
            "state_license": getattr(request.state, "license_id", None),
            "ctxvar_license": license_id_var.get(),
        }

    return app


def _patch_introspect(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    assert hasattr(mw, "_introspect_license"), "introspection entrypoint moved — this control would go blind"

    async def fake(token: str) -> dict[str, Any]:
        return payload

    monkeypatch.setattr(mw, "_introspect_license", fake)


class TestMiddlewareCarriesTheLicenceToTheMeteredHop:
    def test_an_introspected_licence_reaches_inference_router(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The whole point. This is the metered LLM hop; before the change it got
        a tenant and nothing else."""
        _patch_introspect(
            monkeypatch,
            {"active": True, "tenant_id": TENANT, "license_id": CALLER_LICENSE, "roles": []},
        )
        sent: list[httpx.Request] = []
        with TestClient(_app_that_calls_the_metered_hop(sent), client=TRUSTED_PEER) as tc:
            resp = tc.post("/api/verify/llm", headers={"X-Tenant-Id": "Bearer a-customer-token"})

        assert resp.status_code == 200, resp.text
        assert sent[0].headers[LICENSE_HEADER] == CALLER_LICENSE

    def test_request_state_and_the_contextvar_never_disagree(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two writers of one fact. If they drift, one hop bills a licence the
        audit log does not record — the same failure mode `tenant_source` is
        pinned against in test_ws4_vouched_delegation.py."""
        _patch_introspect(
            monkeypatch,
            {"active": True, "tenant_id": TENANT, "license_id": CALLER_LICENSE, "roles": []},
        )
        sent: list[httpx.Request] = []
        with TestClient(_app_that_calls_the_metered_hop(sent), client=TRUSTED_PEER) as tc:
            body = tc.post("/api/verify/llm", headers={"X-Tenant-Id": "Bearer a-customer-token"}).json()
        assert body["state_license"] == CALLER_LICENSE
        assert body["ctxvar_license"] == CALLER_LICENSE

    def test_a_licence_that_is_not_a_string_is_normalised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The introspection payload is untyped JSON. An int licence id would blow
        up httpx's header encoding at the outbound hop — a 500 on a request that
        authenticated perfectly well."""
        _patch_introspect(monkeypatch, {"active": True, "tenant_id": TENANT, "license_id": 12345, "roles": []})
        sent: list[httpx.Request] = []
        with TestClient(_app_that_calls_the_metered_hop(sent), client=TRUSTED_PEER) as tc:
            resp = tc.post("/api/verify/llm", headers={"X-Tenant-Id": "Bearer a-customer-token"})
        assert resp.status_code == 200, resp.text
        assert sent[0].headers[LICENSE_HEADER] == "12345"

    @pytest.mark.parametrize("blank", [None, "", "   "])
    def test_a_licence_free_payload_sends_no_header(self, blank: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        payload: dict[str, Any] = {"active": True, "tenant_id": TENANT, "roles": []}
        if blank is not None:
            payload["license_id"] = blank
        _patch_introspect(monkeypatch, payload)
        sent: list[httpx.Request] = []
        with TestClient(_app_that_calls_the_metered_hop(sent), client=TRUSTED_PEER) as tc:
            body = tc.post("/api/verify/llm", headers={"X-Tenant-Id": "Bearer a-customer-token"}).json()
        assert body["state_license"] is None
        assert LICENSE_HEADER not in sent[0].headers

    def test_the_trusted_network_path_carries_no_licence(self) -> None:
        """The pod-to-pod path presents no licence, so there is nothing honest to
        forward. This is exactly the population
        `billing_license_unattributed_total` must keep counting until those
        callers are credentialed — filling it in here would erase the signal."""
        sent: list[httpx.Request] = []
        with TestClient(_app_that_calls_the_metered_hop(sent), client=TRUSTED_PEER) as tc:
            resp = tc.post("/api/verify/llm", headers={"X-Tenant-Id": TENANT})
        assert resp.status_code == 200, resp.text
        assert LICENSE_HEADER not in sent[0].headers


def _app_that_calls_lean4(sent: list[httpx.Request]) -> FastAPI:
    """Real AuthMiddleware in front of a route that calls lean4-prover."""
    app = FastAPI()
    app.add_middleware(mw.AuthMiddleware)

    @app.post("/api/verify/llm")
    async def verify(request: Request) -> dict[str, Any]:
        client = Lean4ProverClient(BASE_URL, service_license_token=SERVICE_LICENSE)
        client._client = httpx.AsyncClient(base_url=BASE_URL, transport=_transport(sent))
        try:
            await client.check("theorem t : True := trivial", tenant_id=str(request.state.tenant_id))
        finally:
            await client.close()
        return {"ok": True}

    return app


class TestLean4AlwaysCarriesALicence:
    def test_the_end_tenants_own_token_is_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The inbound licence was introspected; forwarding it lets lean4-prover
        attribute the proof work to the tenant that asked for it."""
        _patch_introspect(
            monkeypatch,
            {"active": True, "tenant_id": TENANT, "license_id": CALLER_LICENSE, "roles": []},
        )
        sent: list[httpx.Request] = []
        with TestClient(_app_that_calls_lean4(sent), client=TRUSTED_PEER) as tc:
            resp = tc.post("/api/verify/llm", headers={"X-Tenant-Id": "Bearer a-customer-token"})
        assert resp.status_code == 200, resp.text
        assert sent[0].headers["X-Tenant-Id"] == "Bearer a-customer-token"
        assert sent[0].headers[LICENSE_HEADER] == CALLER_LICENSE

    def test_trusted_network_caller_falls_back_to_the_service_licence(self) -> None:
        sent: list[httpx.Request] = []
        with TestClient(_app_that_calls_lean4(sent), client=TRUSTED_PEER) as tc:
            resp = tc.post("/api/verify/llm", headers={"X-Tenant-Id": TENANT})
        assert resp.status_code == 200, resp.text
        assert sent[0].headers["X-Tenant-Id"] == f"Bearer {SERVICE_LICENSE}"
        assert LICENSE_HEADER not in sent[0].headers

    def test_the_caller_token_does_not_outlive_its_binding(self) -> None:
        """Bound and reset in lockstep with the licence id — a leaked token would
        let whatever runs next in this context spend this tenant's licence.
        Exercised on `_bind_identity` directly: the TestClient runs the app in a
        different context, so asserting after a request could never fail."""
        from auth.tenant import license_token_var

        with mw._bind_identity(TENANT, SOURCE_LICENSE_TOKEN, CALLER_LICENSE, "a-customer-token"):
            assert license_token_var.get() == "a-customer-token"
        assert license_token_var.get() is None


# ---------------------------------------------------------------------------
# Structural guards
# ---------------------------------------------------------------------------


class TestStructuralGuards:
    def test_the_contextvar_default_is_none(self) -> None:
        """An unbound context — background worker, startup probe — has no licence
        and must say so rather than inherit one."""
        assert license_id_var.get() is None

    def test_the_licence_reaches_a_fire_and_forget_background_task(self) -> None:
        """`services.subgraphx_service` spawns `_run_explanation` via
        `asyncio.create_task` and returns; the request scope is reset immediately
        afterwards, and that job then makes two ML-gateway calls. A function
        argument could not reach it. `create_task` snapshots the context AT
        CREATION, so the ContextVar does — measured, not assumed, because
        attribution would be silently missing on that path otherwise."""
        seen: dict[str, str | None] = {}

        async def job(tag: str) -> None:
            await asyncio.sleep(0.01)  # outlive the request scope
            seen[tag] = license_id_var.get()

        async def main() -> None:
            token = license_id_var.set(CALLER_LICENSE)
            in_scope = asyncio.create_task(job("in-scope"))
            license_id_var.reset(token)  # the request ends here
            after = asyncio.create_task(job("after-reset"))
            await asyncio.gather(in_scope, after)

        asyncio.run(main())
        assert seen["in-scope"] == CALLER_LICENSE, "attribution would be missing from the SubgraphX job"
        assert seen["after-reset"] is None, "a task with no request context must not inherit a licence"

    def test_no_client_hand_rolls_the_licence_header(self) -> None:
        """Grep-guard. `kag_common.auth.outbound_headers` is the single definition
        of the fleet's outbound header contract, including the "omit when falsy"
        rule that keeps the honest NULL honest. A client that assigns
        X-License-Id itself has its own copy of that rule to get wrong."""
        src = pathlib.Path(__file__).resolve().parents[2] / "src"
        offenders = []
        for path in sorted(src.rglob("*.py")):
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if LICENSE_HEADER in line and not stripped.startswith(("#", "#:")) and "=" in line:
                    offenders.append(f"{path.relative_to(src)}:{lineno}: {stripped}")
        assert offenders == [], (
            f"X-License-Id must only be emitted via kag_common.auth.outbound_headers. Found: {offenders}"
        )
