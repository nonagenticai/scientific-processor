"""``scientific_processor_tenant_source_total`` + the TENANT_SOURCE log line.

Measurement only. Two properties, both asserted against the REAL
``AuthMiddleware``:

1. Every exit of ``dispatch`` on a protected path increments the counter
   exactly once, with the label that names how the tenant was resolved.
2. Observe identity: the status, body and headers of every path are identical
   with the measurement hook replaced by a no-op — i.e. the counter changes
   nothing about how a request is served.
"""

from __future__ import annotations

import dataclasses
import logging
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
from metrics import TENANT_SOURCE_CARRIERS, TENANT_SOURCES

pytestmark = pytest.mark.unit

UNTRUSTED_PEER = ("203.0.113.9", 44444)
TRUSTED_PEER = ("10.42.3.17", 50000)

CALLER_TENANT = "0195f48b-1111-7000-8000-000000000001"
CALLER_SUBJECT = "0195f48b-2222-7000-8000-000000000002"
PLAIN_TENANT = "0195f48b-4444-7000-8000-000000000004"
VOUCHING_SUBJECT = "svc-vouching-caller"

#: Shapes only — none of these is a real credential.
LICENCE_SHAPED = "A" * 20 + "_" + "b" * 20 + "-9"  # 43 urlsafe chars
# Assembled at runtime so secret scanners do not read a fixture as a credential.
JWT_SHAPED = ".".join(["eyJ" + "x" * 8, "y" * 8, "z" * 4])

GOOD = "good-licence-token"
TENANTLESS = "tenantless-licence-token"
BADTENANT = "bad-tenant-licence-token"
BAD = "bad-licence-token"
VOUCH = "vouching-service-token"

GATE = mw.ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT_ENV


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
            }
        )

    @app.get("/health")
    async def _health() -> dict[str, str]:
        return {"status": "ok"}

    return app


_APP = _make_app()


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    mw.reset_tenant_source_log_gate()
    monkeypatch.delenv(GATE, raising=False)
    monkeypatch.setenv(vouch_mod.ONBEHALF_SUBJECTS_ENV, VOUCHING_SUBJECT)
    monkeypatch.setenv("LICENSE_INTROSPECT_URL", "http://licenses.example/api/introspect")

    async def _fake(token: str) -> dict[str, Any] | None:
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
        return {"active": False}

    monkeypatch.setattr(mw, "_introspect_license", _fake)
    monkeypatch.setattr(vouch_mod, "introspect_license", _fake)
    yield
    mw.reset_tenant_source_log_gate()


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


def _call(headers: dict[str, str], *, peer: tuple[str, int], path: str = "/api/topology/probe") -> Any:
    with TestClient(_APP, client=peer) as tc:
        return tc.post(path, headers=headers, json={})


# (id, headers, peer, path, env-gate value or None, expected labels, expected status)
CASES: list[tuple[str, dict[str, str], tuple[str, int], str, str | None, tuple[str, str, str, str], int]] = [
    (
        "licence_x_tenant_id",
        {"X-Tenant-Id": f"Bearer {GOOD}"},
        UNTRUSTED_PEER,
        "/api/topology/subgraph",
        None,
        ("licence_x_tenant_id", "true", "none", "topology"),
        200,
    ),
    (
        "licence_x_tenant_id_with_jwt_authorization",
        {"X-Tenant-Id": f"Bearer {GOOD}", "Authorization": f"Bearer {JWT_SHAPED}"},
        TRUSTED_PEER,
        "/api/verify/cove",
        None,
        ("licence_x_tenant_id", "true", "bearer_jwt", "verify"),
        200,
    ),
    (
        "plain_header_vouched",
        {"X-Tenant-Id": PLAIN_TENANT, "Authorization": f"Bearer {VOUCH}"},
        TRUSTED_PEER,
        "/api/explain/subgraphx",
        None,
        ("plain_header_vouched", "true", "bearer_other", "explain"),
        200,
    ),
    (
        # A licence in the WRONG header + a plain tenant.
        "plain_header_trusted_licence_in_authorization",
        {"X-Tenant-Id": PLAIN_TENANT, "Authorization": f"Bearer {LICENCE_SHAPED}"},
        TRUSTED_PEER,
        "/api/stats/trend",
        None,
        ("plain_header_trusted", "false", "bearer_licence_shaped", "stats"),
        200,
    ),
    (
        "plain_header_trusted_bare",
        {"X-Tenant-Id": PLAIN_TENANT},
        TRUSTED_PEER,
        "/api/formal/verify/deeper",
        None,
        ("plain_header_trusted", "false", "none", "formal"),
        200,
    ),
    (
        "plain_header_trusted_basic_authorization_unknown_prefix",
        {"X-Tenant-Id": PLAIN_TENANT, "Authorization": "Basic dXNlcjpwYXNz"},
        TRUSTED_PEER,
        "/api/topologyx",  # protected by prefix match, not a registered group
        None,
        ("plain_header_trusted", "false", "non_bearer", "other"),
        200,
    ),
    (
        # A blank bearer is a malformed credential (was: plain tenant "Bearer").
        "blank_bearer_is_malformed",
        {"X-Tenant-Id": "Bearer"},
        TRUSTED_PEER,
        "/api/pipeline/run",
        None,
        ("malformed_credential", "false", "none", "pipeline"),
        401,
    ),
    (
        "default_fallback",
        {},
        TRUSTED_PEER,
        "/api/sandbox/execute",
        None,
        ("default_fallback", "false", "none", "sandbox"),
        200,
    ),
    (
        "default_fallback_refused",
        {},
        TRUSTED_PEER,
        "/api/cache/stats",
        "false",
        ("default_fallback_refused", "false", "none", "cache"),
        401,
    ),
    (
        "invalid_tenant_id",
        {"X-Tenant-Id": "acme:eu"},
        TRUSTED_PEER,
        "/api/cache/clear",
        None,
        ("invalid_tenant_id", "false", "none", "cache"),
        400,
    ),
    (
        "licence_invalid",
        {"X-Tenant-Id": f"Bearer {BAD}"},
        UNTRUSTED_PEER,
        "/api/topology/probe",
        None,
        ("licence_invalid", "false", "none", "topology"),
        401,
    ),
    (
        "licence_missing_tenant",
        {"X-Tenant-Id": f"Bearer {TENANTLESS}"},
        UNTRUSTED_PEER,
        "/api/topology/probe",
        None,
        ("licence_missing_tenant", "true", "none", "topology"),
        401,
    ),
    (
        "licence_malformed_tenant",
        {"X-Tenant-Id": f"Bearer {BADTENANT}"},
        UNTRUSTED_PEER,
        "/api/topology/probe",
        None,
        ("licence_malformed_tenant", "true", "none", "topology"),
        401,
    ),
    (
        "missing",
        {"X-Tenant-Id": PLAIN_TENANT},
        UNTRUSTED_PEER,
        "/api/topology/probe",
        None,
        ("missing", "false", "none", "topology"),
        401,
    ),
]


@pytest.mark.parametrize(
    ("headers", "peer", "path", "gate", "labels", "status"),
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_each_path_increments_exactly_one_series(
    headers: dict[str, str],
    peer: tuple[str, int],
    path: str,
    gate: str | None,
    labels: tuple[str, str, str, str],
    status: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if gate is not None:
        monkeypatch.setenv(GATE, gate)
    before, before_total = _count(*labels), _total()

    resp = _call(headers, peer=peer, path=path)

    assert resp.status_code == status, resp.text
    assert _count(*labels) - before == 1.0
    assert _total() - before_total == 1.0, "exactly ONE increment per request"


def test_every_case_label_is_in_the_declared_vocabulary() -> None:
    for _id, _h, _p, _path, _g, (source, has_licence, carrier, _rg), _s in CASES:
        assert source in TENANT_SOURCES
        assert carrier in TENANT_SOURCE_CARRIERS
        assert has_licence in {"true", "false"}
    from tests.unit.test_tokenless_identity_mode import ENFORCE_CASES

    exercised = {c[5][0] for c in CASES} | {c[4][0] for c in ENFORCE_CASES}
    assert exercised == set(TENANT_SOURCES), "every source is exercised"


@pytest.mark.parametrize("path", ["/health", "/api", "/docs", "/not-protected/x"])
def test_public_and_unprotected_paths_are_not_counted(path: str) -> None:
    before = _total()
    _call({"X-Tenant-Id": PLAIN_TENANT}, peer=UNTRUSTED_PEER, path=path)
    assert _total() == before


@pytest.mark.parametrize(
    ("headers", "peer", "path", "gate"),
    [(c[1], c[2], c[3], c[4]) for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_observe_identity_responses_unchanged(
    headers: dict[str, str],
    peer: tuple[str, int],
    path: str,
    gate: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same request with measurement ON and OFF → identical status, body, headers."""
    if gate is not None:
        monkeypatch.setenv(GATE, gate)
    with_measurement = _call(headers, peer=peer, path=path)

    monkeypatch.setattr(mw, "_record_tenant_source", lambda *a, **k: None)
    before = _total()
    without = _call(headers, peer=peer, path=path)
    assert _total() == before, "the no-op really disabled measurement"

    assert with_measurement.status_code == without.status_code
    # The error envelope carries a wall-clock timestamp; compare everything else.
    body_on, body_off = with_measurement.json(), without.json()
    for body in (body_on, body_off):
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            body["error"].pop("timestamp", None)
    assert body_on == body_off
    assert dict(with_measurement.headers) == dict(without.headers)


def test_measurement_failure_never_breaks_serving(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> str:
        raise RuntimeError("metrics backend exploded")

    monkeypatch.setattr(mw, "authorization_carrier", _boom)
    resp = _call({"X-Tenant-Id": PLAIN_TENANT}, peer=TRUSTED_PEER)
    assert resp.status_code == 200
    assert resp.json()["tenant_id"] == PLAIN_TENANT

    monkeypatch.setattr(mw.TENANT_SOURCE, "labels", _boom)
    resp = _call({"X-Tenant-Id": f"Bearer {BAD}"}, peer=UNTRUSTED_PEER)
    assert resp.status_code == 401


def test_log_line_names_the_peer_and_is_deduped_hourly(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="auth.middleware")
    for _ in range(5):
        _call(
            {
                "X-Tenant-Id": PLAIN_TENANT,
                "Authorization": f"Bearer {LICENCE_SHAPED}",
                "User-Agent": "kag proxy\r\nx=1",
            },
            peer=TRUSTED_PEER,
        )
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("TENANT_SOURCE ")]
    assert len(lines) == 1, lines
    line = lines[0]
    assert "source=plain_header_trusted" in line
    assert "carrier=bearer_licence_shaped" in line
    assert "route_group=topology" in line
    assert f"peer={TRUSTED_PEER[0]}" in line
    assert "user_agent=kag_proxy__x_1" in line, "caller-supplied fields are sanitised"
    # ⛔ never the tenant id, never the credential
    assert PLAIN_TENANT not in line
    assert LICENCE_SHAPED not in line

    # A different peer is a different caller → its own line.
    _call({"X-Tenant-Id": PLAIN_TENANT}, peer=("10.42.9.9", 1))
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("TENANT_SOURCE ")]
    assert len(lines) == 2


def test_denials_do_not_emit_a_tenant_source_line(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="auth.middleware")
    _call({"X-Tenant-Id": PLAIN_TENANT}, peer=UNTRUSTED_PEER)
    _call({"X-Tenant-Id": f"Bearer {BAD}"}, peer=UNTRUSTED_PEER)
    assert not [r for r in caplog.records if r.getMessage().startswith("TENANT_SOURCE ")]


def test_log_gate_window_and_key_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(mw.time, "monotonic", lambda: clock[0])
    gate = mw._should_log_tenant_source
    assert gate("s", "p", "r") is True
    assert gate("s", "p", "r") is False
    clock[0] += mw._TENANT_SOURCE_LOG_INTERVAL_SECONDS
    assert gate("s", "p", "r") is True

    monkeypatch.setattr(mw, "_TENANT_SOURCE_LOG_MAX_KEYS", 3)
    mw.reset_tenant_source_log_gate()
    for i in range(3):
        assert gate("s", f"p{i}", "r") is True
    assert gate("s", "p-new", "r") is True
    assert len(mw._tenant_source_log_seen) == 1


def test_route_groups_match_protected_prefixes_and_mounted_routers() -> None:
    """A new protected router that is not in SCIENTIFIC_ROUTE_GROUPS would read as ``other``."""
    from app import create_app
    from config import Settings

    from_prefixes = {p.split("/")[2] for p in mw.PROTECTED_PREFIXES}
    assert from_prefixes == set(mw.SCIENTIFIC_ROUTE_GROUPS)

    app = create_app(dataclasses.replace(Settings(), enable_auth=False))
    # FastAPI wraps included routers, so read the mounted paths from the
    # generated schema (every /api route is in it; test_openapi pins that).
    mounted = {path.split("/")[2] for path in app.openapi()["paths"] if path.startswith("/api/")}
    assert mounted == set(mw.SCIENTIFIC_ROUTE_GROUPS)


def test_counter_is_exposed_by_the_in_process_registry() -> None:
    """No /metrics route exists; the series must still render from the registry."""
    from prometheus_client import generate_latest

    _call({"X-Tenant-Id": PLAIN_TENANT}, peer=TRUSTED_PEER)
    text = generate_latest(REGISTRY).decode()
    assert "scientific_processor_tenant_source_total{" in text
    assert 'source="plain_header_trusted"' in text


def test_app_serves_no_metrics_route() -> None:
    from app import create_app
    from config import Settings

    app = create_app(dataclasses.replace(Settings(), enable_auth=False))
    with TestClient(app, client=TRUSTED_PEER) as tc:
        assert tc.get("/metrics").status_code == 404


@pytest.mark.parametrize("mode", [None, "observe", "enforce"])
@pytest.mark.parametrize("peer", [TRUSTED_PEER, UNTRUSTED_PEER])
@pytest.mark.parametrize("value", ["Bearer", "Bearer ", "bearer", "bearer ", "BEARER", "Bearer   ", " Bearer\t"])
def test_blank_bearer_is_a_malformed_credential_in_every_mode(
    value: str, peer: tuple[str, int], mode: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Formerly a KNOWN DEFECT: ``X-Tenant-Id: Bearer`` (no token) skipped the
    licence path (``"bearer "`` needs the trailing space, which header parsing
    strips) AND the trusted branch's ``startswith("bearer ")`` guard, so the
    literal string ``"Bearer"`` was bound as the tenant. Fixed: a
    present-but-empty credential is a 401 MALFORMED_CREDENTIAL, in both modes,
    from any peer, never introspected, never reaching a handler.
    """
    if mode is not None:
        monkeypatch.setenv(mw.TOKENLESS_IDENTITY_MODE_ENV, mode)
    calls: list[str] = []

    async def _spy(token: str) -> dict[str, Any] | None:
        calls.append(token)
        return {"active": True, "tenant_id": CALLER_TENANT}

    monkeypatch.setattr(mw, "_introspect_license", _spy)
    resp = _call({"X-Tenant-Id": value}, peer=peer)
    assert resp.status_code == 401, resp.text
    assert resp.headers.get("www-authenticate") == "Bearer"
    body = resp.json()
    assert body["error"]["code"] == "MALFORMED_CREDENTIAL"
    assert "tenant_id" not in body  # the echo handler was never reached
    assert calls == []


def test_a_valid_bearer_still_binds_the_licence_tenant() -> None:
    """Control: the fix keys on an EMPTY token only."""
    resp = _call({"X-Tenant-Id": f"Bearer {GOOD}"}, peer=UNTRUSTED_PEER)
    assert resp.status_code == 200, resp.text
    assert resp.json()["tenant_id"] == CALLER_TENANT


def test_a_plain_uuid_tenant_is_unchanged() -> None:
    """Control: a raw uuid from a trusted peer is still served as that tenant."""
    resp = _call({"X-Tenant-Id": PLAIN_TENANT}, peer=TRUSTED_PEER)
    assert resp.status_code == 200, resp.text
    assert resp.json()["tenant_id"] == PLAIN_TENANT


@pytest.mark.parametrize("tenant", ["bearerco", "Bearer-co", "bearer.tenant"])
def test_a_tenant_name_starting_with_bearer_is_not_the_scheme(tenant: str) -> None:
    """The fix keys on the SCHEME, not a prefix: these stay plain tenants."""
    resp = _call({"X-Tenant-Id": tenant}, peer=TRUSTED_PEER)
    assert resp.status_code == 200, resp.text
    assert resp.json()["tenant_id"] == tenant


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "none"),
        ("", "none"),
        ("   ", "none"),
        ("Bearer", "bearer_other"),
        ("Bearer ", "bearer_other"),
        (f"bearer {LICENCE_SHAPED}", "bearer_licence_shaped"),
        (f"Bearer {JWT_SHAPED}", "bearer_jwt"),
        (f"Bearer {LICENCE_SHAPED}x", "bearer_other"),
        ("Token abc", "non_bearer"),
    ],
)
def test_authorization_carrier_shapes(value: str | None, expected: str) -> None:
    assert mw.authorization_carrier(value) == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/api", "root"),
        ("/api/", "root"),
        ("/api/topology", "topology"),
        ("/api/topology/x/y", "topology"),
        ("/api/verify/", "verify"),
        ("/api/unknown", "other"),
        ("/api/" + "z" * 500, "other"),
    ],
)
def test_route_group_is_bounded(path: str, expected: str) -> None:
    assert mw.tenant_route_group(path) == expected
