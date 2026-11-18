"""Authentication layer tests -- models, license-introspection middleware, deps.

Covers the User model (tenant + subject axes), the dual-path AuthMiddleware
(Bearer-introspection vs trusted-network raw-uuid routing), and the FastAPI
dependency functions.

Security contract under test:
- Client headers ``x-user-id`` / ``x-user-roles`` / ``X-Subject-Id`` are NEVER
  trusted as identity on any path.
- Identity (tenant + subject + roles) comes ONLY from an introspected license
  carried in ``X-Tenant-Id: Bearer <token>``.
- A trusted-network peer with a raw ``X-Tenant-Id: <uuid>`` routes tenant-only
  (roles empty, subject None) — for in-cluster pod-to-pod traffic.
- An untrusted peer with no valid token is rejected 401 (fail-closed).
"""

import os

# Auth must be enabled for middleware tests -- set BEFORE any src imports.
os.environ["ENABLE_AUTH"] = "true"
os.environ["ENABLE_DOCS"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from typing import Any

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

import auth.middleware as mw
from auth.dependencies import (
    get_current_user,
    get_current_user_id,
    require_all_roles,
    require_any_role,
    require_role,
    resolve_tenant_id,
)
from auth.middleware import AuthMiddleware, _is_trusted_network
from auth.models import User

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# User model tests
# ---------------------------------------------------------------------------


class TestUserModel:
    """auth.models.User pydantic model."""

    def test_model_post_init_defaults_tenant_to_org(self):
        user = User(sub="u1", organization_id="org-42", tenant_id=None)
        assert user.tenant_id == "org-42"

    def test_model_post_init_defaults_tenant_to_sub(self):
        user = User(sub="u1", tenant_id=None)
        assert user.tenant_id == "u1"

    def test_model_post_init_preserves_explicit_tenant(self):
        user = User(sub="u1", organization_id="org-42", tenant_id="t-99")
        assert user.tenant_id == "t-99"

    def test_effective_tenant_id_returns_tenant(self):
        user = User(sub="u1", tenant_id="t-1")
        assert user.effective_tenant_id == "t-1"

    def test_effective_tenant_id_falls_back_to_org(self):
        user = User(sub="u1", organization_id="org-7", tenant_id=None)
        assert user.effective_tenant_id == "org-7"

    def test_effective_tenant_id_falls_back_to_sub(self):
        user = User(sub="u1")
        assert user.effective_tenant_id == "u1"

    def test_effective_subject_id_returns_subject(self):
        user = User(sub="u1", tenant_id="t-1", subject_id="s-9")
        assert user.effective_subject_id == "s-9"

    def test_effective_subject_id_falls_back_to_tenant(self):
        user = User(sub="u1", tenant_id="t-1")
        assert user.effective_subject_id == "t-1"

    def test_effective_tenant_name_returns_name(self):
        user = User(sub="u1", tenant_id="t-1", tenant_name="Acme")
        assert user.effective_tenant_name == "Acme"

    def test_effective_tenant_name_falls_back_to_tenant(self):
        user = User(sub="u1", tenant_id="t-1")
        assert user.effective_tenant_name == "t-1"

    def test_has_realm_role_true(self):
        user = User(sub="u1", realm_access={"roles": ["admin", "viewer"]})
        assert user.has_realm_role("admin") is True

    def test_has_realm_role_false(self):
        user = User(sub="u1", realm_access={"roles": ["viewer"]})
        assert user.has_realm_role("admin") is False

    def test_has_realm_role_empty(self):
        user = User(sub="u1")
        assert user.has_realm_role("admin") is False

    def test_has_client_role_true(self):
        user = User(sub="u1", resource_access={"my-app": {"roles": ["editor"]}})
        assert user.has_client_role("my-app", "editor") is True

    def test_has_client_role_false_wrong_role(self):
        user = User(sub="u1", resource_access={"my-app": {"roles": ["viewer"]}})
        assert user.has_client_role("my-app", "editor") is False

    def test_get_all_roles_combines_realm_and_client(self):
        user = User(
            sub="u1",
            realm_access={"roles": ["admin"]},
            resource_access={
                "app-a": {"roles": ["editor"]},
                "app-b": {"roles": ["viewer", "writer"]},
            },
        )
        assert user.get_all_roles() == ["admin", "editor", "viewer", "writer"]

    def test_is_admin_with_admin_role(self):
        user = User(sub="u1", realm_access={"roles": ["admin"]})
        assert user.is_admin() is True

    def test_is_admin_false(self):
        user = User(sub="u1", realm_access={"roles": ["viewer"]})
        assert user.is_admin() is False


# ---------------------------------------------------------------------------
# _is_trusted_network helper
# ---------------------------------------------------------------------------


class TestIsTrustedNetwork:
    def test_loopback_ipv4(self):
        assert _is_trusted_network("127.0.0.1") is True

    def test_loopback_ipv6(self):
        assert _is_trusted_network("::1") is True

    def test_rfc1918_10(self):
        assert _is_trusted_network("10.0.0.5") is True

    def test_pod_cidr_10_42(self):
        assert _is_trusted_network("10.42.7.3") is True

    def test_testclient_sentinel_is_not_trusted(self):
        """INVERTED contract (was ``is True``).

        This test used to ASSERT the escape hatch, which is what kept it alive
        through green CI. Starlette's default ASGI peer is the literal string
        ``"testclient"``; production auth trusting that name meant a test-only
        bypass shipped in the auth path. uvicorn's ``X-Forwarded-For`` parser does
        not validate that the forwarded value is an IP, so with
        ``proxy_headers=True`` (its default) a loopback peer could set
        ``scope["client"]`` to exactly this string. Only real IPs are trusted now.
        """
        assert _is_trusted_network("testclient") is False

    def test_testclient_sentinel_is_not_trusted_in_any_casing(self):
        """A case-folded or suffixed variant must not sneak back in either."""
        for variant in ("TESTCLIENT", "TestClient", "testclient.example", "testclient "):
            assert _is_trusted_network(variant) is False, variant

    def test_public_ip_rejected(self):
        assert _is_trusted_network("8.8.8.8") is False

    def test_none_host(self):
        assert _is_trusted_network(None) is False

    def test_garbage_string(self):
        assert _is_trusted_network("not-an-ip") is False


# ---------------------------------------------------------------------------
# Middleware integration tests (real FastAPI + TestClient)
# ---------------------------------------------------------------------------

_VALID_TOKEN = "valid-license-token"  # noqa: S105 — test fixture value
_INACTIVE_TOKEN = "revoked-token"  # noqa: S105 — test fixture value


async def _fake_introspect(token: str) -> dict[str, Any] | None:
    """Stand-in for the licenses endpoint used by the middleware."""
    if token == _VALID_TOKEN:
        return {
            "active": True,
            "tenant_id": "tenant-from-license",
            "subject_id": "subject-from-license",
            "roles": ["admin", "editor"],
            "tenant_name": "Acme Labs",
            "license_id": "lic-123",
        }
    if token == _INACTIVE_TOKEN:
        return {"active": False}
    return None


@pytest.fixture
def patched_introspect(monkeypatch):
    monkeypatch.setattr(mw, "_introspect_license", _fake_introspect)


def _make_auth_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/docs")
    def docs_page():
        return {"docs": True}

    @app.get("/api/topology/test")
    def protected_endpoint(request: Request):
        user = getattr(request.state, "user", None)
        if user is None:
            return {"sub": None, "tenant_id": None}
        return {
            "sub": user.sub,
            "tenant_id": request.state.tenant_id,
            "subject_id": getattr(request.state, "subject_id", None),
            "username": user.preferred_username,
            "roles": user.get_all_roles(),
        }

    @app.get("/random/path")
    def random_path():
        return {"non_protected": True}

    return app


#: The peer address every in-cluster test speaks from. A REAL loopback IP, not
#: Starlette's default ``"testclient"`` sentinel: production auth no longer has a
#: name-based escape hatch (see ``auth.middleware._is_trusted_network``), so a
#: test that wants the trusted-network branch must present a peer production
#: would also trust. Anything else and the test would silently stop exercising
#: the branch its name claims.
TRUSTED_PEER_IP = "127.0.0.1"


def _client(app: FastAPI, client_ip: str = TRUSTED_PEER_IP) -> TestClient:
    """TestClient with an explicit peer address.

    Defaults to a trusted loopback peer; pass e.g. ``client_ip="8.8.8.8"`` to
    exercise the untrusted (fail-closed 401) branch.
    """

    async def scope_patching_app(scope, receive, send):
        if scope["type"] == "http":
            scope["client"] = (client_ip, 50000)
        await app(scope, receive, send)

    return TestClient(scope_patching_app)


class TestPublicAndPassthrough:
    def test_public_health(self):
        with _client(_make_auth_app()) as c:
            assert c.get("/health").status_code == 200

    def test_public_docs(self):
        with _client(_make_auth_app()) as c:
            assert c.get("/docs").status_code == 200

    def test_non_protected_passthrough(self):
        with _client(_make_auth_app()) as c:
            assert c.get("/random/path").json() == {"non_protected": True}


class TestTrustedNetworkPath:
    """No token, trusted peer (real loopback IP) → tenant routing ONLY."""

    def test_raw_uuid_routes_tenant_only(self):
        with _client(_make_auth_app()) as c:
            resp = c.get("/api/topology/test", headers={"X-Tenant-Id": "tenant-abc"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["tenant_id"] == "tenant-abc"
        assert body["sub"] == "tenant-abc"
        assert body["subject_id"] is None
        assert body["roles"] == []
        assert body["username"] == "service"

    def test_no_tenant_header_uses_default(self):
        with _client(_make_auth_app()) as c:
            resp = c.get("/api/topology/test")
        assert resp.status_code == 200
        assert resp.json()["tenant_id"] == mw.DEFAULT_TENANT_ID

    def test_forged_identity_headers_ignored(self):
        """x-user-id / x-user-roles / X-Subject-Id must NOT become identity."""
        with _client(_make_auth_app()) as c:
            resp = c.get(
                "/api/topology/test",
                headers={
                    "X-Tenant-Id": "tenant-abc",
                    "x-user-id": "attacker",
                    "x-user-roles": "admin,platform-admin",
                    "X-Subject-Id": "victim-subject",
                },
            )
        body = resp.json()
        assert body["roles"] == []  # forged roles dropped
        assert body["subject_id"] is None  # forged subject dropped
        assert body["sub"] == "tenant-abc"  # not "attacker"


class TestBearerIntrospectionPath:
    def test_valid_token_sets_identity_from_payload(self, patched_introspect):
        with _client(_make_auth_app()) as c:
            resp = c.get(
                "/api/topology/test",
                headers={"X-Tenant-Id": f"Bearer {_VALID_TOKEN}"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["tenant_id"] == "tenant-from-license"
        assert body["subject_id"] == "subject-from-license"
        assert body["roles"] == ["admin", "editor"]

    def test_forged_roles_header_ignored_on_bearer_path(self, patched_introspect):
        with _client(_make_auth_app()) as c:
            resp = c.get(
                "/api/topology/test",
                headers={
                    "X-Tenant-Id": f"Bearer {_VALID_TOKEN}",
                    "x-user-roles": "superadmin",
                    "X-Subject-Id": "victim",
                },
            )
        body = resp.json()
        assert body["roles"] == ["admin", "editor"]  # from payload, not header
        assert body["subject_id"] == "subject-from-license"  # not "victim"

    def test_inactive_token_401(self, patched_introspect):
        with _client(_make_auth_app()) as c:
            resp = c.get(
                "/api/topology/test",
                headers={"X-Tenant-Id": f"Bearer {_INACTIVE_TOKEN}"},
            )
        assert resp.status_code == 401

    def test_unknown_token_401(self, patched_introspect):
        with _client(_make_auth_app()) as c:
            resp = c.get(
                "/api/topology/test",
                headers={"X-Tenant-Id": "Bearer nope"},
            )
        assert resp.status_code == 401

    def test_valid_token_works_from_untrusted_ip(self, patched_introspect):
        """Introspection is IP-independent — a valid token authenticates even
        from a public peer."""
        with _client(_make_auth_app(), client_ip="8.8.8.8") as c:
            resp = c.get(
                "/api/topology/test",
                headers={"X-Tenant-Id": f"Bearer {_VALID_TOKEN}"},
            )
        assert resp.status_code == 200
        assert resp.json()["tenant_id"] == "tenant-from-license"


class TestUntrustedFailClosed:
    def test_untrusted_no_token_401(self):
        with _client(_make_auth_app(), client_ip="8.8.8.8") as c:
            resp = c.get("/api/topology/test", headers={"X-Tenant-Id": "tenant-abc"})
        assert resp.status_code == 401

    def test_graceful_degradation_raw_uuid_still_routes_when_url_unset(self, monkeypatch):
        """LICENSE_INTROSPECT_URL unset: raw-uuid pod-to-pod callers still route."""
        monkeypatch.delenv("LICENSE_INTROSPECT_URL", raising=False)
        with _client(_make_auth_app()) as c:  # loopback peer = trusted network
            resp = c.get("/api/topology/test", headers={"X-Tenant-Id": "tenant-abc"})
        assert resp.status_code == 200
        assert resp.json()["tenant_id"] == "tenant-abc"

    def test_bearer_rejected_when_url_unset(self, monkeypatch):
        """LICENSE_INTROSPECT_URL unset: a bearer caller fails closed (401)."""
        monkeypatch.delenv("LICENSE_INTROSPECT_URL", raising=False)
        with _client(_make_auth_app()) as c:
            resp = c.get(
                "/api/topology/test",
                headers={"X-Tenant-Id": "Bearer some-uncached-token"},
            )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Dependency function tests
# ---------------------------------------------------------------------------


def _make_dep_app(*, enable_auth: bool = True) -> FastAPI:
    import auth.dependencies as deps_mod

    deps_mod.ENABLE_AUTH = enable_auth

    app = FastAPI()
    if enable_auth:
        app.add_middleware(AuthMiddleware)

    @app.get("/api/topology/me")
    def me(user: User = Depends(get_current_user)):
        return {"sub": user.sub, "tenant_id": user.effective_tenant_id}

    @app.get("/api/topology/me/id")
    def me_id(uid: str = Depends(get_current_user_id)):
        return {"user_id": uid}

    @app.get("/api/topology/admin-only")
    def admin_only(user: User = Depends(require_role("admin"))):
        return {"sub": user.sub, "is_admin": True}

    @app.get("/api/topology/any-role")
    def any_role_endpoint(user: User = Depends(require_any_role("admin", "editor"))):
        return {"sub": user.sub}

    @app.get("/api/topology/all-roles")
    def all_roles_endpoint(user: User = Depends(require_all_roles("admin", "editor"))):
        return {"sub": user.sub}

    return app


class TestGetCurrentUser:
    def test_auth_disabled_returns_anonymous(self):
        import auth.dependencies as deps_mod

        original = deps_mod.ENABLE_AUTH
        with _client(_make_dep_app(enable_auth=False)) as c:
            resp = c.get("/api/topology/me")
        deps_mod.ENABLE_AUTH = original
        assert resp.status_code == 200
        assert resp.json()["sub"] == mw.DEFAULT_TENANT_ID

    def test_bearer_sets_user(self, patched_introspect):
        import auth.dependencies as deps_mod

        original = deps_mod.ENABLE_AUTH
        with _client(_make_dep_app(enable_auth=True)) as c:
            resp = c.get(
                "/api/topology/me",
                headers={"X-Tenant-Id": f"Bearer {_VALID_TOKEN}"},
            )
        deps_mod.ENABLE_AUTH = original
        assert resp.status_code == 200
        assert resp.json()["tenant_id"] == "tenant-from-license"

    def test_untrusted_no_user_401(self):
        import auth.dependencies as deps_mod

        original = deps_mod.ENABLE_AUTH
        deps_mod.ENABLE_AUTH = True
        app = FastAPI()  # no middleware → request.state.user never set

        @app.get("/unprotected")
        def endpoint(user: User = Depends(get_current_user)):
            return {"sub": user.sub}

        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/unprotected")
        deps_mod.ENABLE_AUTH = original
        assert resp.status_code == 401


class TestRoleDependencies:
    """Roles now come ONLY from the introspected license payload."""

    def test_admin_role_from_license_passes(self, patched_introspect):
        with _client(_make_dep_app()) as c:
            resp = c.get(
                "/api/topology/admin-only",
                headers={"X-Tenant-Id": f"Bearer {_VALID_TOKEN}"},
            )
        assert resp.status_code == 200

    def test_forged_role_header_gets_403(self):
        """A trusted-net caller forging x-user-roles=admin has NO roles → 403."""
        with _client(_make_dep_app()) as c:
            resp = c.get(
                "/api/topology/admin-only",
                headers={"X-Tenant-Id": "tenant-abc", "x-user-roles": "admin"},
            )
        assert resp.status_code == 403

    def test_any_role_from_license(self, patched_introspect):
        with _client(_make_dep_app()) as c:
            resp = c.get(
                "/api/topology/any-role",
                headers={"X-Tenant-Id": f"Bearer {_VALID_TOKEN}"},
            )
        assert resp.status_code == 200

    def test_all_roles_from_license(self, patched_introspect):
        with _client(_make_dep_app()) as c:
            resp = c.get(
                "/api/topology/all-roles",
                headers={"X-Tenant-Id": f"Bearer {_VALID_TOKEN}"},
            )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tenant-id grammar boundary
# ---------------------------------------------------------------------------


class TestTenantIdIsValidatedAtTheBoundary:
    """A tenant id is interpolated verbatim into several delimiter-based grammars.

    ``services/cache.py`` builds ``subgraph:<tenant>:<entity>:<hops>`` (delimiter
    ``:``); ``services/history_service.py`` builds ``history/<tenant>/<entity>/``
    (delimiter ``/``). A tenant id carrying one of those delimiters shifts the
    grammar and makes the parser report the WRONG owner.

    Reproduced end-to-end before this check: tenant ``acme:eu`` wrote
    ``subgraph:acme:eu:e1:2``, whose segment 1 is ``acme``. ``POST /api/cache/clear``
    as ``acme`` then returned 200 ``previous_size=1``, having bulk-deleted
    ``acme:eu``'s entry — while ``acme:eu`` itself got 403 invalidating its OWN key.

    The check lives HERE, at the single point the value enters the process, rather
    than as an escape at each interpolation site: escaping fixes one grammar and
    every future site has to remember to repeat it.
    """

    @pytest.mark.parametrize(
        "bad",
        [
            "acme:eu",  # cache-key delimiter — the reproduced defect
            "acme/eu",  # MinIO history-prefix delimiter
            "acme eu",  # whitespace
            "acme\teu",
            "acme%3Aeu",  # pre-encoded: still not an id we issue
            "../etc/passwd",
            "A" * 129,  # length bound
        ],
        ids=["colon", "slash", "space", "tab", "percent", "traversal", "too-long"],
    )
    def test_trusted_peer_with_a_delimiter_bearing_tenant_id_is_refused(self, bad):
        with _client(_make_auth_app()) as c:
            resp = c.get("/api/topology/test", headers={"X-Tenant-Id": bad})
        assert resp.status_code == 400, f"{bad!r} reached the router"
        assert resp.json()["error"]["code"] == "INVALID_TENANT_ID"

    @pytest.mark.parametrize(
        "good",
        [
            "0195f48b-d000-7000-8000-000000000001",
            "tenant-abc",
            "tenant_abc",
            "tenant.abc",
            "t1",
            "acme",
            "acme2",
            "A" * 128,
        ],
    )
    def test_every_tenant_id_this_system_issues_still_passes(self, good):
        """Non-vacuity guard: the charset must not have narrowed onto real callers."""
        with _client(_make_auth_app()) as c:
            resp = c.get("/api/topology/test", headers={"X-Tenant-Id": good})
        assert resp.status_code == 200, f"{good!r} was wrongly refused"
        assert resp.json()["tenant_id"] == good

    def test_a_license_carrying_a_malformed_tenant_id_is_refused(self, monkeypatch):
        """A license is more trustworthy than a header — not a reason to skip the check."""

        async def _bad_tenant_introspect(token: str):
            return {"active": True, "tenant_id": "acme:eu", "license_id": "lic-1"}

        monkeypatch.setattr(mw, "_introspect_license", _bad_tenant_introspect)
        with _client(_make_auth_app()) as c:
            resp = c.get("/api/topology/test", headers={"X-Tenant-Id": f"Bearer {_VALID_TOKEN}"})
        assert resp.status_code == 401
        assert "malformed tenant id" in resp.json()["error"]["message"]

    def test_the_cache_key_grammar_is_now_unambiguous_for_every_admitted_id(self):
        """The end of the defect: the refused ids are exactly the ambiguous ones.

        For every tenant id the boundary admits, building a key and parsing it back
        returns the SAME tenant — so no caller can be 403-ed on its own key, and no
        caller can reach another tenant's keys by clearing.
        """
        from services.cache import CacheService

        for tenant in ("acme", "acme2", "tenant-abc", "tenant_abc", "tenant.abc", "t1"):
            assert mw._is_valid_tenant_id(tenant)
            key = CacheService.make_subgraph_key(tenant, "urn:doi:10.1/x", 2)
            assert CacheService.key_tenant(key) == tenant

        assert not mw._is_valid_tenant_id("acme:eu")
        with pytest.raises(ValueError, match="delimiter"):
            CacheService.make_subgraph_key("acme:eu", "e1", 2)


class TestTenantIdMustBeAUuidAtTheBoundary:
    """A non-uuid tenant is a 400, never a 500 and never a 404.

    ``jobs.tenant_id`` / ``verification_history.tenant_id`` became real Postgres
    ``uuid`` columns in migration ``005``. ``AuthMiddleware`` still admits anything
    matching ``_TENANT_ID_RE`` (``[A-Za-z0-9._-]{1,128}``), so a legacy slug like
    ``n8n-dev`` passes the header check and reaches the query layer — where
    Postgres now raises ``invalid input syntax for type uuid`` and FastAPI turns
    that into a **500**. Malformed client input must not become a server error, so
    ``resolve_tenant_id`` validates and returns 400 at the single choke point.
    """

    @staticmethod
    def _app() -> FastAPI:
        app = FastAPI()
        app.add_middleware(AuthMiddleware)

        @app.get("/api/topology/test")
        def endpoint(request: Request):
            # Exactly what every real router does.
            return {"tenant_id": resolve_tenant_id(request)}

        return app

    @pytest.mark.parametrize("bad", ["n8n-dev", "anonymous", "tenant-abc", "not-a-uuid"])
    def test_non_uuid_tenant_header_is_400(self, bad: str):
        with _client(self._app()) as c:
            resp = c.get("/api/topology/test", headers={"X-Tenant-Id": bad})
        assert resp.status_code == 400, (
            f"{bad!r} returned {resp.status_code}; a non-uuid tenant must be a 400 "
            "(400 = malformed input), not a 500 from the uuid cast and not a 404."
        )
        assert resp.status_code != 404

    def test_raw_uuid_pod_to_pod_path_still_works_and_yields_a_str(self):
        """The trusted-network ``X-Tenant-Id: <uuid>`` path must be untouched.

        And the value must come back as a ``str``: it is handed to
        ``set_config('app.current_tenant', …)``, which is ``(text, text, bool)``
        with no uuid overload — a ``uuid.UUID`` there raises ``UndefinedFunction``
        and kills every transaction under RLS.
        """
        tenant = "019de842-c92a-7e37-8fdd-eb0d8c5396f9"
        with _client(self._app()) as c:
            resp = c.get("/api/topology/test", headers={"X-Tenant-Id": tenant})
        assert resp.status_code == 200
        got = resp.json()["tenant_id"]
        assert got == tenant  # byte-identical, not reformatted
        assert isinstance(got, str)

    def test_default_tenant_is_itself_a_uuid(self):
        """The no-header fallback must not trip the guard it just installed."""
        with _client(self._app()) as c:
            resp = c.get("/api/topology/test")
        assert resp.status_code == 200
        assert resp.json()["tenant_id"] == mw.DEFAULT_TENANT_ID
