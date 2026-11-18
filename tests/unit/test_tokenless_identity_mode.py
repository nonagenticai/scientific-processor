"""``AUTH_TOKENLESS_IDENTITY_MODE`` — procedure_management #116 semantics.

Three properties, all against the REAL ``AuthMiddleware``:

1. Boot: the mode is ``observe`` by default, and any other value than
   observe / enforce refuses to start (``config.Settings``).
2. Observe (default) is identical to origin/main for every live caller shape:
   the ``Authorization`` licence carrier never runs, no new introspection call
   is made, a plain ``X-Tenant-Id`` from a trusted peer is still served.
3. Enforce: a plain tenant without a licence is a 401; ``Authorization:
   Bearer <licence>`` binds the licence's tenant (403 for a foreign tenant unless
   the subject delegates, 401 for a non-licence, 503 when introspection gives no
   verdict).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import auth.middleware as mw
import auth.vouch as vouch_mod
from auth.middleware import AuthMiddleware
from config import Settings, parse_tokenless_identity_mode

pytestmark = pytest.mark.unit

MODE = mw.TOKENLESS_IDENTITY_MODE_ENV

UNTRUSTED_PEER = ("203.0.113.9", 44444)
TRUSTED_PEER = ("10.42.3.17", 50000)  # an in-cluster pod (kag proxy, discovery-engine)

CALLER_TENANT = "0195f48b-1111-7000-8000-000000000001"
CALLER_SUBJECT = "0195f48b-2222-7000-8000-000000000002"
PLAIN_TENANT = "0195f48b-4444-7000-8000-000000000004"
VOUCHING_SUBJECT = "svc-vouching-caller"

#: Shapes only — none of these is a real credential.
LICENCE_SHAPED = "A" * 20 + "_" + "b" * 20 + "-9"  # 43 urlsafe chars, introspects inactive
# Assembled at runtime so secret scanners do not read a fixture as a credential.
JWT_SHAPED = ".".join(["eyJ" + "x" * 8, "y" * 8, "z" * 4])

GOOD = "good-licence-token"
VOUCH = "vouching-service-token"
TENANTLESS = "tenantless-licence-token"
BADTENANT = "bad-tenant-licence-token"
NOVERDICT = "no-verdict-token"
BADBODY = "bad-body-token"


def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.api_route("/api/{rest:path}", methods=["GET", "POST"])
    async def _echo(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "tenant_id": getattr(request.state, "tenant_id", None),
                "tenant_source": getattr(request.state, "tenant_source", None),
                "subject_id": getattr(request.state, "subject_id", None),
                "license_id": getattr(request.state, "license_id", None),
            }
        )

    return app


_APP = _make_app()

#: Every token handed to the middleware's introspection, in order.
INTROSPECTED: list[str] = []


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    mw.reset_tenant_source_log_gate()
    INTROSPECTED.clear()
    monkeypatch.delenv(MODE, raising=False)
    monkeypatch.delenv(mw.ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT_ENV, raising=False)
    monkeypatch.setenv(vouch_mod.ONBEHALF_SUBJECTS_ENV, VOUCHING_SUBJECT)
    monkeypatch.setenv("LICENSE_INTROSPECT_URL", "http://licenses.example/api/introspect")

    async def _fake(token: str) -> Any:
        INTROSPECTED.append(token)
        if token in (GOOD, VOUCH):
            return {
                "active": True,
                "tenant_id": CALLER_TENANT,
                "subject_id": VOUCHING_SUBJECT if token == VOUCH else CALLER_SUBJECT,
                "license_id": "lic-1",
            }
        if token == TENANTLESS:
            return {"active": True, "subject_id": CALLER_SUBJECT}
        if token == BADTENANT:
            return {"active": True, "tenant_id": "acme:eu", "subject_id": CALLER_SUBJECT}
        if token == NOVERDICT:
            return None
        if token == BADBODY:
            raise AttributeError("'list' object has no attribute 'get'")
        return {"active": False}

    async def _fake_vouch(token: str) -> Any:
        # The observe-path vouch (auth/vouch.py) — kag_common's introspection,
        # which origin/main already calls; tracked separately.
        return await _fake(token) if token == VOUCH else {"active": False}

    monkeypatch.setattr(mw, "_introspect_license", _fake)
    monkeypatch.setattr(vouch_mod, "introspect_license", _fake_vouch)
    yield
    mw.reset_tenant_source_log_gate()


def _call(headers: dict[str, str], *, peer: tuple[str, int] = TRUSTED_PEER, path: str = "/api/topology/probe") -> Any:
    with TestClient(_APP, client=peer) as tc:
        return tc.post(path, headers=headers, json={})


def _count(source: str, has_licence: str, carrier: str, route_group: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "scientific_processor_tenant_source_total",
            {"source": source, "has_licence": has_licence, "carrier": carrier, "route_group": route_group},
        )
        or 0.0
    )


def _total() -> float:
    total = 0.0
    for metric in REGISTRY.collect():
        if metric.name == "scientific_processor_tenant_source":
            for sample in metric.samples:
                if sample.name == "scientific_processor_tenant_source_total":
                    total += sample.value
    return total


def _strip_timestamp(body: Any) -> Any:
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        body["error"].pop("timestamp", None)
    return body


# ---------------------------------------------------------------------------
# 1. Boot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, "observe"), ("", "observe"), ("  ", "observe"), ("observe", "observe"), (" ENFORCE ", "enforce")],
)
def test_mode_parses_and_defaults_to_observe(raw: str | None, expected: str) -> None:
    assert parse_tokenless_identity_mode(raw) == expected


@pytest.mark.parametrize("raw", ["enforced", "on", "true", "1", "observ", "deny"])
def test_unknown_mode_refuses_boot(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODE, raw)
    with pytest.raises(ValueError, match=MODE):
        Settings()


@pytest.mark.parametrize("environment", ["test", "development", "production"])
def test_unknown_mode_refuses_boot_in_every_environment(environment: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Checked BEFORE the production-only required-field check, so a missing
    production field cannot mask it and a non-production env cannot skip it."""
    monkeypatch.setenv("ENVIRONMENT", environment)
    monkeypatch.setenv(MODE, "enforc")
    with pytest.raises(ValueError, match=MODE):
        Settings()


def test_settings_carries_the_normalised_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings().auth_tokenless_identity_mode == "observe"
    monkeypatch.setenv(MODE, "Enforce")
    assert Settings().auth_tokenless_identity_mode == "enforce"


def test_an_unknown_mode_that_bypassed_boot_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODE, "garbage")
    assert mw._tokenless_identity_enforced() is True
    resp = _call({"X-Tenant-Id": PLAIN_TENANT})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 2. Observe identity — the live caller shapes
# ---------------------------------------------------------------------------
# kag proxy (knowledge/kag origin/main, routers/scientific.py + services/
# proxy.py): `X-Tenant-Id: <resolved tenant uuid>` and the INBOUND
# `Authorization` forwarded verbatim — absent, a Keycloak JWT, or a licence.
# scientific-processor is not one of kag's vouch destinations, so kag never
# swaps in its own service token here. discovery-engine: a raw uuid.
# kag_orchestrator: configures scientific_processor_url but makes no call to it
# on origin/main.
LIVE_SHAPES: list[tuple[str, dict[str, str]]] = [
    ("kag_proxy_no_authorization", {"X-Tenant-Id": PLAIN_TENANT}),
    ("kag_proxy_forwarded_keycloak_jwt", {"X-Tenant-Id": PLAIN_TENANT, "Authorization": f"Bearer {JWT_SHAPED}"}),
    (
        "kag_proxy_forwarded_licence_shaped",
        {"X-Tenant-Id": PLAIN_TENANT, "Authorization": f"Bearer {LICENCE_SHAPED}"},
    ),
    ("kag_proxy_forwarded_valid_licence", {"X-Tenant-Id": PLAIN_TENANT, "Authorization": f"Bearer {GOOD}"}),
    ("kag_proxy_x_request_id", {"X-Tenant-Id": PLAIN_TENANT, "X-Request-ID": "r-1", "X-Subject-Id": "u-1"}),
    ("discovery_engine_raw_uuid", {"X-Tenant-Id": CALLER_TENANT, "X-Tenant-Name": "acme"}),
]


@pytest.mark.parametrize("mode", [None, "observe"])
@pytest.mark.parametrize(("_id", "headers"), LIVE_SHAPES, ids=[s[0] for s in LIVE_SHAPES])
def test_observe_serves_every_live_caller_shape_as_before(
    _id: str, headers: dict[str, str], mode: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if mode is not None:
        monkeypatch.setenv(MODE, mode)
    called: list[str] = []

    async def _never(self: Any, request: Any, call_next: Any, token: str) -> Any:
        called.append(token)
        raise AssertionError("the Authorization carrier ran in observe")

    monkeypatch.setattr(AuthMiddleware, "_dispatch_authorization_licence", _never)
    resp = _call(headers)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["tenant_id"] == headers["X-Tenant-Id"]
    assert body["tenant_source"] == "unvouched-header"
    assert body["license_id"] is None
    assert called == []
    assert INTROSPECTED == [], "observe makes no licence introspection for a plain tenant"


def test_observe_still_vouches_exactly_as_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-existing trusted-path vouch is untouched in observe."""
    resp = _call({"X-Tenant-Id": PLAIN_TENANT, "Authorization": f"Bearer {VOUCH}"})
    assert resp.status_code == 200
    assert resp.json()["tenant_source"] == "vouched-onbehalf"
    assert resp.json()["tenant_id"] == PLAIN_TENANT


def test_observe_ignores_an_authorization_licence_without_a_tenant_header() -> None:
    """No X-Tenant-Id + Authorization licence → today's default fallback, not a licence bind."""
    resp = _call({"Authorization": f"Bearer {GOOD}"})
    assert resp.status_code == 200
    assert resp.json()["tenant_source"] == "trusted-network-default"
    assert INTROSPECTED == []


def test_unset_and_observe_are_byte_identical_for_every_counter_case(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.test_tenant_source_counter import CASES

    for _id, headers, peer, path, gate, _labels, _status in CASES:
        if gate is not None:
            monkeypatch.setenv(mw.ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT_ENV, gate)
        else:
            monkeypatch.delenv(mw.ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT_ENV, raising=False)
        monkeypatch.delenv(MODE, raising=False)
        unset = _call(headers, peer=peer, path=path)
        monkeypatch.setenv(MODE, "observe")
        observe = _call(headers, peer=peer, path=path)
        assert unset.status_code == observe.status_code, _id
        assert _strip_timestamp(unset.json()) == _strip_timestamp(observe.json()), _id
        assert dict(unset.headers) == dict(observe.headers), _id


# ---------------------------------------------------------------------------
# 3. Enforce
# ---------------------------------------------------------------------------

# (id, headers, peer, path, expected labels, expected status)
ENFORCE_CASES: list[tuple[str, dict[str, str], tuple[str, int], str, tuple[str, str, str, str], int]] = [
    (
        "plain_header_refused",
        {"X-Tenant-Id": PLAIN_TENANT},
        TRUSTED_PEER,
        "/api/topology/probe",
        ("plain_header_refused", "false", "none", "topology"),
        401,
    ),
    (
        "plain_header_refused_basic_authorization",
        {"X-Tenant-Id": PLAIN_TENANT, "Authorization": "Basic dXNlcjpwYXNz"},
        TRUSTED_PEER,
        "/api/verify/cove",
        ("plain_header_refused", "false", "non_bearer", "verify"),
        401,
    ),
    (
        "licence_authorization_no_tenant_header",
        {"Authorization": f"Bearer {GOOD}"},
        TRUSTED_PEER,
        "/api/stats/trend",
        ("licence_authorization", "true", "bearer_other", "stats"),
        200,
    ),
    (
        "licence_authorization_matching_tenant_untrusted_peer",
        {"X-Tenant-Id": CALLER_TENANT, "Authorization": f"Bearer {GOOD}"},
        UNTRUSTED_PEER,
        "/api/formal/verify",
        ("licence_authorization", "true", "bearer_other", "formal"),
        200,
    ),
    (
        "licence_authorization_onbehalf",
        {"X-Tenant-Id": PLAIN_TENANT, "Authorization": f"Bearer {VOUCH}"},
        TRUSTED_PEER,
        "/api/explain/subgraphx",
        ("licence_authorization_onbehalf", "true", "bearer_other", "explain"),
        200,
    ),
    (
        "licence_authorization_foreign_tenant",
        {"X-Tenant-Id": PLAIN_TENANT, "Authorization": f"Bearer {GOOD}"},
        TRUSTED_PEER,
        "/api/cache/clear",
        ("licence_authorization_foreign_tenant", "true", "bearer_other", "cache"),
        403,
    ),
    (
        "licence_authorization_invalid_jwt",
        {"X-Tenant-Id": PLAIN_TENANT, "Authorization": f"Bearer {JWT_SHAPED}"},
        TRUSTED_PEER,
        "/api/sandbox/execute",
        ("licence_authorization_invalid", "false", "bearer_jwt", "sandbox"),
        401,
    ),
    (
        "licence_authorization_malformed_tenant_missing",
        {"Authorization": f"Bearer {TENANTLESS}"},
        TRUSTED_PEER,
        "/api/pipeline/run",
        ("licence_authorization_malformed_tenant", "true", "bearer_other", "pipeline"),
        401,
    ),
    (
        "licence_authorization_malformed_tenant_grammar",
        {"Authorization": f"Bearer {BADTENANT}"},
        TRUSTED_PEER,
        "/api/pipeline/run",
        ("licence_authorization_malformed_tenant", "true", "bearer_other", "pipeline"),
        401,
    ),
    (
        "licence_authorization_unavailable",
        {"X-Tenant-Id": PLAIN_TENANT, "Authorization": f"Bearer {NOVERDICT}"},
        TRUSTED_PEER,
        "/api/topology/probe",
        ("licence_authorization_unavailable", "false", "bearer_other", "topology"),
        503,
    ),
    (
        "licence_authorization_unparseable_body",
        {"Authorization": f"Bearer {BADBODY}"},
        TRUSTED_PEER,
        "/api/topology/probe",
        ("licence_authorization_unavailable", "false", "bearer_other", "topology"),
        503,
    ),
    (
        "licence_authorization_invalid_asserted_tenant",
        {"X-Tenant-Id": "acme:eu", "Authorization": f"Bearer {GOOD}"},
        TRUSTED_PEER,
        "/api/cache/clear",
        ("invalid_tenant_id", "true", "bearer_other", "cache"),
        400,
    ),
    (
        "x_tenant_id_licence_keeps_precedence",
        {"X-Tenant-Id": f"Bearer {GOOD}", "Authorization": f"Bearer {JWT_SHAPED}"},
        TRUSTED_PEER,
        "/api/verify/cove",
        ("licence_x_tenant_id", "true", "bearer_jwt", "verify"),
        200,
    ),
    (
        "blank_x_tenant_id_bearer_keeps_precedence",
        {"X-Tenant-Id": "Bearer", "Authorization": f"Bearer {GOOD}"},
        TRUSTED_PEER,
        "/api/verify/cove",
        ("malformed_credential", "false", "bearer_other", "verify"),
        401,
    ),
    (
        "default_fallback_unchanged",
        {},
        TRUSTED_PEER,
        "/api/stats/trend",
        ("default_fallback", "false", "none", "stats"),
        200,
    ),
    (
        "untrusted_plain_tenant_still_missing",
        {"X-Tenant-Id": PLAIN_TENANT},
        UNTRUSTED_PEER,
        "/api/topology/probe",
        ("missing", "false", "none", "topology"),
        401,
    ),
]


@pytest.mark.parametrize(
    ("headers", "peer", "path", "labels", "status"),
    [c[1:] for c in ENFORCE_CASES],
    ids=[c[0] for c in ENFORCE_CASES],
)
def test_enforce_each_exit_status_and_one_series(
    headers: dict[str, str],
    peer: tuple[str, int],
    path: str,
    labels: tuple[str, str, str, str],
    status: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MODE, "enforce")
    before, before_total = _count(*labels), _total()
    resp = _call(headers, peer=peer, path=path)
    assert resp.status_code == status, resp.text
    assert _count(*labels) - before == 1.0
    assert _total() - before_total == 1.0


def test_enforce_plain_tenant_401_names_both_carriers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODE, "enforce")
    resp = _call({"X-Tenant-Id": PLAIN_TENANT})
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"
    msg = resp.json()["error"]["message"]
    assert "Authorization: Bearer <licence>" in msg
    assert "X-Tenant-Id: Bearer <licence>" in msg
    assert INTROSPECTED == []


def test_enforce_authorization_licence_binds_the_licence_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODE, "enforce")
    for headers in (
        {"Authorization": f"Bearer {GOOD}"},
        {"X-Tenant-Id": CALLER_TENANT, "Authorization": f"bearer {GOOD}"},
    ):
        resp = _call(headers)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["tenant_id"] == CALLER_TENANT, "the licence's tenant_id, never its subject_id"
        assert body["tenant_source"] == "license-token"
        assert body["subject_id"] == CALLER_SUBJECT
        assert body["license_id"] == "lic-1"


def test_enforce_onbehalf_acts_as_the_asserted_tenant_without_the_service_licence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MODE, "enforce")
    resp = _call({"X-Tenant-Id": PLAIN_TENANT, "Authorization": f"Bearer {VOUCH}"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["tenant_id"] == PLAIN_TENANT
    assert body["tenant_source"] == "vouched-onbehalf"
    assert body["license_id"] is None, "the service's licence is never billed as the tenant's"
    assert body["subject_id"] is None


def test_enforce_foreign_tenant_is_403_unless_the_subject_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODE, "enforce")
    resp = _call({"X-Tenant-Id": PLAIN_TENANT, "Authorization": f"Bearer {GOOD}"})
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "TENANT_ASSERTION_NOT_PERMITTED"
    # The same licence becomes a delegator once its subject is allow-listed.
    monkeypatch.setenv(vouch_mod.ONBEHALF_SUBJECTS_ENV, f"{VOUCHING_SUBJECT},{CALLER_SUBJECT}")
    resp = _call({"X-Tenant-Id": PLAIN_TENANT, "Authorization": f"Bearer {GOOD}"})
    assert resp.status_code == 200
    assert resp.json()["tenant_id"] == PLAIN_TENANT
    # And an empty allowlist delegates nothing.
    monkeypatch.setenv(vouch_mod.ONBEHALF_SUBJECTS_ENV, "")
    assert _call({"X-Tenant-Id": PLAIN_TENANT, "Authorization": f"Bearer {VOUCH}"}).status_code == 403


def test_enforce_no_verdict_is_503_with_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODE, "enforce")
    for token in (NOVERDICT, BADBODY):
        resp = _call({"Authorization": f"Bearer {token}"})
        assert resp.status_code == 503, resp.text
        assert resp.headers["retry-after"] == "5"
        assert resp.json()["error"]["code"] == "LICENSE_INTROSPECTION_UNAVAILABLE"


def test_enforce_unset_introspect_url_is_503_not_a_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    """The REAL introspection (not the fake) with no URL configured gives no verdict."""
    monkeypatch.setenv(MODE, "enforce")
    monkeypatch.undo()  # drop the fake _introspect_license
    monkeypatch.setenv(MODE, "enforce")
    monkeypatch.delenv("LICENSE_INTROSPECT_URL", raising=False)
    mw._license_cache.clear()
    mw._license_neg_cache.clear()
    resp = _call({"Authorization": "Bearer some-uncached-token"})
    assert resp.status_code == 503


def test_enforce_non_licence_authorization_is_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODE, "enforce")
    resp = _call({"Authorization": f"Bearer {JWT_SHAPED}"})
    assert resp.status_code == 401
    assert INTROSPECTED == [JWT_SHAPED]


def test_enforce_x_tenant_id_carrier_wins_and_authorization_is_not_introspected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MODE, "enforce")
    resp = _call({"X-Tenant-Id": f"Bearer {GOOD}", "Authorization": f"Bearer {VOUCH}"})
    assert resp.status_code == 200
    assert INTROSPECTED == [GOOD]


def test_enforce_refusals_emit_no_tenant_source_line(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    caplog.set_level(logging.INFO, logger="auth.middleware")
    monkeypatch.setenv(MODE, "enforce")
    _call({"X-Tenant-Id": PLAIN_TENANT})
    _call({"X-Tenant-Id": PLAIN_TENANT, "Authorization": f"Bearer {GOOD}"})
    _call({"Authorization": f"Bearer {JWT_SHAPED}"})
    assert not [r for r in caplog.records if r.getMessage().startswith("TENANT_SOURCE ")]
    _call({"Authorization": f"Bearer {GOOD}"})
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("TENANT_SOURCE ")]
    assert len(lines) == 1 and "source=licence_authorization " in lines[0]
    assert GOOD not in lines[0]
