"""The DEFAULT_TENANT_ID fallback: gated, and loud enough to burn down.

WHAT IS WRONG TODAY
-------------------
``AuthMiddleware``'s trusted-network branch does ``raw_tenant or
DEFAULT_TENANT_ID``. That default is not a placeholder: measured live, 38 cluster
workloads set ``DEFAULT_TENANT_ID`` and 37 of them point at the SAME REAL
customer tenant ``0195f48b-d000-7000-8000-000000000001``. So every anonymous
in-cluster caller is served — and has its downstream token spend billed — as a
real paying customer that never made the request. Silently: the response looks
identical either way.

WHY THE FIX IS A LOG AND A SWITCH, NOT A DIFFERENT DEFAULT
----------------------------------------------------------
Substituting a different value makes it worse, and both obvious candidates are
already ruled out by measurement: the nil UUID is quarantined by
``billing_service`` as malformed (``_cost_schema.py``) into a dead-letter stream
that skips the token rollup AND the budget debit, converting mis-billing into
UN-billing; and the slug ``"n8n-dev"`` resolves through ``canonical_tenant_id``
to yet another real tenant. The only correct fix is to CREDENTIAL these callers,
and that requires first knowing who they are — hence a WARNING naming the peer
and the path, which is the entire deliverable here.

WHY THE SWITCH DEFAULTS TO ON
-----------------------------
``ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT`` defaults to True because that is today's
behaviour and closing it 401s every live in-cluster caller. Shipping it closed
would be an outage dressed as a security fix. ``TestTheDefaultIsOn`` is the
availability control that pins that, and it is the test that must never be
"fixed" by changing the default.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import auth.middleware as mw
from auth.middleware import AuthMiddleware, _allow_trusted_network_default_tenant
from auth.tenant import SOURCE_TRUSTED_NETWORK_DEFAULT

pytestmark = pytest.mark.unit

GATE = mw.ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT_ENV
#: A REAL loopback peer: ``_is_trusted_network`` does not trust Starlette's
#: ``"testclient"`` sentinel, and the trusted-network branch is the only branch
#: this file exercises.
TRUSTED_PEER = ("127.0.0.1", 50000)
#: Must be under PROTECTED_PREFIXES — the middleware short-circuits anything that
#: is neither public nor protected, and a probe outside them would never reach
#: the fallback at all and would assert nothing.
PROBE_PATH = "/api/topology/probe"
NAMED_TENANT = "0195f48b-d000-7000-8000-000000000042"


def _probe_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get(PROBE_PATH)
    async def probe(request: Request) -> dict[str, Any]:
        return {"tenant": request.state.tenant_id, "source": request.state.tenant_source}

    return app


def _get(headers: dict[str, str] | None = None) -> Any:
    with TestClient(_probe_app(), client=TRUSTED_PEER) as tc:
        return tc.get(PROBE_PATH, headers=headers or {})


# ---------------------------------------------------------------------------
# THE AVAILABILITY CONTROL — do not "fix" this by flipping the default
# ---------------------------------------------------------------------------


class TestTheDefaultIsOn:
    def test_unset_env_means_the_fallback_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(GATE, raising=False)
        assert _allow_trusted_network_default_tenant() is True

    def test_a_headerless_trusted_peer_is_still_served(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every live in-cluster caller that names no tenant lands here. Shipping
        the gate closed would 401 all of them on deploy."""
        monkeypatch.delenv(GATE, raising=False)
        resp = _get()
        assert resp.status_code == 200, resp.text
        assert resp.json()["tenant"] == mw.DEFAULT_TENANT_ID
        assert resp.json()["source"] == SOURCE_TRUSTED_NETWORK_DEFAULT

    @pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on", "", "   "])
    def test_true_or_blank_stays_on(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(GATE, value)
        assert _allow_trusted_network_default_tenant() is True
        assert _get().status_code == 200

    @pytest.mark.parametrize("value", ["maybe", "flase", "enforce", "closed", "2"])
    def test_an_unknown_value_refuses_boot(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Formerly fail-OPEN (a typo'd close, ``flase``, silently left the
        fallback serving). Now the process refuses to start: a rolling deploy keeps
        the old pods serving, so the typo is loud without 401-ing callers."""
        from config import Settings

        monkeypatch.setenv(GATE, value)
        with pytest.raises(ValueError, match=GATE):
            Settings()

    @pytest.mark.parametrize("value", ["maybe", "flase"])
    def test_an_unknown_value_that_bypassed_boot_fails_closed(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(GATE, value)
        assert _allow_trusted_network_default_tenant() is False
        assert _get().status_code == 401


# ---------------------------------------------------------------------------
# THE GATE ACTUALLY GATES — otherwise it is decoration
# ---------------------------------------------------------------------------


class TestTheGateIsWired:
    @pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "off", " Off "])
    def test_closing_the_gate_rejects_the_headerless_caller(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(GATE, value)
        assert _allow_trusted_network_default_tenant() is False
        resp = _get()
        assert resp.status_code == 401, resp.text

    def test_closing_the_gate_does_not_touch_a_caller_that_names_its_tenant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scope control. The switch closes the FALLBACK only. If it also refused
        credentialed pod-to-pod traffic, the migration could never be completed —
        there would be no working position to move callers to."""
        monkeypatch.setenv(GATE, "false")
        resp = _get({"X-Tenant-Id": NAMED_TENANT})
        assert resp.status_code == 200, resp.text
        assert resp.json()["tenant"] == NAMED_TENANT

    def test_the_gate_is_read_per_request_not_at_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A gitops env flip must take effect on the next request. Reading it at
        import time would make the rollout require a pod restart and would make
        the two positions untestable in one process."""
        monkeypatch.setenv(GATE, "false")
        assert _get().status_code == 401
        monkeypatch.setenv(GATE, "true")
        assert _get().status_code == 200


# ---------------------------------------------------------------------------
# THE BURN-DOWN SIGNAL — the whole reason the gate can ever be closed
# ---------------------------------------------------------------------------


class TestTenantFallbackIsObservable:
    def test_the_warning_names_the_peer_and_the_path(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the peer and the path nobody can enumerate the callers still on
        the fallback, and the gate above can never be safely closed. This is the
        deliverable; keep TENANT_FALLBACK a stable, greppable token."""
        monkeypatch.delenv(GATE, raising=False)
        with caplog.at_level(logging.WARNING, logger="auth.middleware"):
            assert _get().status_code == 200
        assert "TENANT_FALLBACK" in caplog.text
        assert f"peer={TRUSTED_PEER[0]}" in caplog.text
        assert f"path={PROBE_PATH}" in caplog.text
        assert mw.DEFAULT_TENANT_ID in caplog.text

    def test_it_is_a_warning_not_an_info(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prod sets no LOG_LEVEL for this module, and the line has to survive
        whatever floor Loki is filtering at — an INFO burn-down signal that nobody
        can see is the same as no signal."""
        monkeypatch.delenv(GATE, raising=False)
        with caplog.at_level(logging.WARNING, logger="auth.middleware"):
            _get()
        records = [r for r in caplog.records if "TENANT_FALLBACK" in r.getMessage()]
        assert records, "the burn-down line was not emitted at WARNING"
        assert all(r.levelno == logging.WARNING for r in records)

    def test_a_caller_that_names_its_tenant_produces_no_line(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The signal must be able to reach ZERO. Logging credentialed callers too
        would bury it in noise that no amount of migration work can remove."""
        monkeypatch.delenv(GATE, raising=False)
        with caplog.at_level(logging.WARNING, logger="auth.middleware"):
            assert _get({"X-Tenant-Id": NAMED_TENANT}).status_code == 200
        assert "TENANT_FALLBACK" not in caplog.text

    def test_a_licence_bearing_caller_produces_no_line(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The end state. A caller that has been credentialed is off the fallback
        and must stop appearing — that is what "burnt down" looks like."""
        monkeypatch.delenv(GATE, raising=False)

        async def fake(token: str) -> dict[str, Any]:
            return {"active": True, "tenant_id": NAMED_TENANT, "license_id": "lic-1", "roles": []}

        monkeypatch.setattr(mw, "_introspect_license", fake)
        with caplog.at_level(logging.WARNING, logger="auth.middleware"):
            assert _get({"X-Tenant-Id": "Bearer a-customer-token"}).status_code == 200
        assert "TENANT_FALLBACK" not in caplog.text

    def test_no_line_when_the_gate_is_closed(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once the fallback is refused there is no fallback to burn down; a line
        here would re-fill a counter that is supposed to have reached zero."""
        monkeypatch.setenv(GATE, "false")
        with caplog.at_level(logging.WARNING, logger="auth.middleware"):
            assert _get().status_code == 401
        assert "TENANT_FALLBACK" not in caplog.text


# ---------------------------------------------------------------------------
# One DEFAULT_TENANT_ID, not three
# ---------------------------------------------------------------------------


class TestDefaultTenantHasOneDefinition:
    def test_middleware_and_dependencies_read_the_same_object(self) -> None:
        """They used to call os.getenv independently with the fallback uuid
        spelled out twice. If those ever disagreed, a header-less caller would be
        SERVED as one tenant by the middleware and READ BACK as another by
        `resolve_tenant_id` — one caller, two cache namespaces, two RLS scopes."""
        import auth.dependencies as deps
        import auth.tenant as tenant_mod

        assert mw.DEFAULT_TENANT_ID is tenant_mod.DEFAULT_TENANT_ID
        assert deps.DEFAULT_TENANT_ID is tenant_mod.DEFAULT_TENANT_ID

    def test_config_agrees_with_it(self) -> None:
        """`config.Settings.default_tenant_id` is deliberately still its own read
        (it is lru_cached, so folding it in would change when the env freezes),
        which makes it the one place a divergence could still appear."""
        import auth.tenant as tenant_mod
        from config import Settings

        assert Settings().default_tenant_id == tenant_mod.DEFAULT_TENANT_ID

    def test_only_two_modules_read_the_env_var(self) -> None:
        """Grep-guard against a fourth reader appearing. Deliberately NOT done by
        `importlib.reload(auth.tenant)`: reloading that module mints fresh
        ContextVar objects while `auth.middleware` keeps binding the old ones, so
        the WS-4 provenance and licence guards would silently go inert for every
        test that ran afterwards in the same session — an order-dependent hole in
        exactly the controls this work adds.
        """
        src = pathlib.Path(__file__).resolve().parents[2] / "src"
        readers = sorted(
            str(p.relative_to(src)) for p in src.rglob("*.py") if '"DEFAULT_TENANT_ID"' in p.read_text(encoding="utf-8")
        )
        assert readers == ["auth/tenant.py", "config.py"], (
            "DEFAULT_TENANT_ID must be read in exactly two places — auth/tenant.py (the single "
            f"definition every auth module imports) and config.py (lru_cached Settings). Found: {readers}"
        )
