"""License-token authentication middleware.

Replaces the previous "trust client headers as identity" pass-through with
license-token introspection so scientific-processor honours the global
"internal services use license-authz" rule. Ported from discovery-engine's
middleware (the fleet reference for this pattern).

Wire format
-----------
- License tokens travel in ``X-Tenant-Id: Bearer <token>``. That carrier
  always takes precedence; a Bearer there with an EMPTY token is a 401
  ``MALFORMED_CREDENTIAL`` in every mode (it used to be bound as a plain tenant
  literally named "Bearer").
- ``Authorization: Bearer <licence>`` is a second licence carrier ACTIVE ONLY
  under ``AUTH_TOKENLESS_IDENTITY_MODE=enforce`` (port of procedure_management
  #116). In observe (the default) ``Authorization`` is consulted only by
  ``auth/vouch.py`` (which can only widen) and is otherwise classified by SHAPE
  for the counter — no network call — exactly as before.
- ``X-Tenant-Name`` is optional display context, never trusted as identity.
- ``X-Subject-Id`` / ``x-user-id`` / ``x-user-roles`` are **ignored on every
  path**. Subject, roles and user-id come only from an introspected license.
  The trusted set is every pod in the cluster (the k3s pod CIDR 10.42.0.0/16
  is inside the default 10.0.0.0/8 trusted range), so honouring those headers
  as identity is role forgery and subject impersonation, not
  defense-in-depth. (scientific-processor's RLS is tenant-only today, so the
  subject half was latent — no reader consumes request.state.subject_id — but
  it is closed here regardless.)
- Public endpoints (health, docs, root) bypass introspection.
- RFC1918 / loopback peers (by real IP — never a hostname; see
  ``_is_trusted_network``) with NO token bypass introspection
  with a raw ``X-Tenant-Id: <uuid>`` used for tenant ROUTING ONLY. This keeps
  in-cluster pod-to-pod traffic (discovery-engine + kag proxy, which both send
  a raw uuid) working, including when ``LICENSE_INTROSPECT_URL`` is not yet
  wired (graceful degradation). Roles/subject/user-id are never derived on
  this path.
- ``AUTH_TOKENLESS_IDENTITY_MODE=enforce`` closes that bypass for a caller that
  NAMES a tenant: a plain ``X-Tenant-Id`` with no licence binding it is a 401.
  The header-LESS default fallback is governed separately by
  ``ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT``.
- Untrusted peer with no valid token → 401 (fail-closed). scientific-processor
  is not edge-exposed (not in AUTH_REQUIRED_HOSTS), so this only ever bites a
  genuinely unauthenticated caller.

Successful introspection responses are cached for 5 minutes by SHA-256 prefix
of the token (TTLCache); ``active: false`` payloads are negatively cached; a
cold-cache burst for the same token is collapsed via a per-token single-flight
lock.

WS-4 TENANT PROVENANCE
----------------------
Alongside the tenant, this middleware records HOW the tenant was established
(``auth.tenant.SOURCE_*``) on ``request.state.tenant_source`` and binds it to
``auth.tenant.tenant_source_var`` in lockstep with ``context.tenant_id_var``.

That is load-bearing, not bookkeeping. ``clients.pytorch_backend`` stamps
``X-Effective-Tenant-Id`` on the outbound ML-gateway calls, and this service's SA
subject ``019f83f2-ddd1-779f-9c70-7a0c6de1dacb`` is on the modality-gateway's
``AUTH_DELEGATING_SUBJECTS`` — so the gateway ACTS AS that tenant: it selects
that tenant's BYO LLM provider credentials and bills it. The trusted-network
branch below accepts a BARE ``X-Tenant-Id`` from any peer in ``TRUSTED_CIDRS``
(in prod: the whole pod CIDR, no credential at all), so delegating an unvouched
value let any pod spend an arbitrary victim tenant's credentials. The provenance
is what stops that.

Nothing here rejects anything new: a bare ``X-Tenant-Id`` is still ACCEPTED
everywhere as a data scope, exactly as before, and the RLS GUC contract
(``context.tenant_id_var`` → ``app.current_tenant``) is untouched. Only the
DELEGATION is gated, and only in ``clients.pytorch_backend``. See
``auth/vouch.py`` for how a service earns ``vouched-onbehalf``.

LICENCE ATTRIBUTION
-------------------
The introspected ``license_id`` is bound to ``auth.tenant.license_id_var`` in the
same lockstep, and every client in ``src/clients/`` stamps it as ``X-License-Id``.
Before that it reached ``request.state.license_id`` and stopped: no outbound hop
carried it, so every downstream — inference-router above all — saw a tenant and
no licence, which is exactly the shape ``billing_service`` counts in
``billing_license_unattributed_total``.

TENANT FALLBACK BURN-DOWN
-------------------------
The trusted-network branch's ``raw_tenant or DEFAULT_TENANT_ID`` fallback now
emits a ``TENANT_FALLBACK`` WARNING naming the peer and the path, and is gated on
``ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT`` (default **on** — closing it 401s live
callers and is a separate, measured gitops step).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import ipaddress
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

import httpx
from cachetools import TTLCache
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from config import (
    ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT_ENV,
    TOKENLESS_IDENTITY_MODE_ENFORCE,
    TOKENLESS_IDENTITY_MODE_ENV,
    parse_auth_flag,
    parse_tokenless_identity_mode,
)
from context import tenant_id_var
from metrics import TENANT_SOURCE

from .models import User
from .tenant import (
    DEFAULT_TENANT_ID,
    SOURCE_LICENSE_TOKEN,
    SOURCE_TRUSTED_NETWORK_DEFAULT,
    SOURCE_UNVOUCHED_HEADER,
    SOURCE_VOUCHED_ONBEHALF,
    license_id_var,
    license_token_var,
    tenant_source_var,
)
from .vouch import authorized_onbehalf, onbehalf_subjects

logger = logging.getLogger(__name__)

PUBLIC_ENDPOINTS: set[str] = {
    "/",
    "/favicon.ico",
    "/health",
    "/health/ready",
    "/health/live",
    "/health/detailed",
    "/docs",
    "/redoc",
    "/openapi.json",
}

PROTECTED_PREFIXES: set[str] = {
    "/api/topology",
    "/api/verify",
    "/api/stats",
    "/api/explain",
    "/api/formal",
    "/api/sandbox",
    "/api/pipeline",
    "/api/cache",
}

# ``DEFAULT_TENANT_ID`` is imported from ``auth.tenant``, which is now its single
# definition (see the comment there). The import still binds it as a module-level
# name, so ``mw.DEFAULT_TENANT_ID`` — read by ``tests/unit/test_auth.py`` — and
# per-module monkeypatching keep working exactly as when it was defined here.

# ``ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT_ENV`` (imported from ``config``): the
# migration switch for the header-less trusted-network fallback below.
#
# MUST DEFAULT TO ON — that is today's behaviour, and flipping it off 401s every
# live in-cluster caller that reaches scientific-processor without naming a
# tenant. Turning it off is a SEPARATE, MEASURED gitops step, and the measurement
# is the ``TENANT_FALLBACK`` warning emitted alongside it: each line names a peer
# and a path that still has to be credentialed. When that line stops being
# emitted in Loki for a full traffic cycle, and only then, the switch is safe to
# close. Flipping it before that is an outage, not a tightening.
#
# Anything unrecognised REFUSES BOOT (``config.parse_auth_flag`` via
# ``Settings``): it used to be read as ON, so a typo'd close (``flase``) left the
# fallback silently open. A rolling deploy keeps the old pods serving while the
# new one refuses to start, so the typo is loud without being an outage. One
# that bypassed boot fails CLOSED at call time.

_TRUSTED_CIDRS_RAW = os.getenv(
    "TRUSTED_CIDRS",
    "127.0.0.1/32,::1/128,192.168.0.0/16,172.16.0.0/12,10.0.0.0/8",
)
TRUSTED_NETWORKS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.ip_network(cidr.strip(), strict=False) for cidr in _TRUSTED_CIDRS_RAW.split(",") if cidr.strip()
]

# License introspection cache — keyed by SHA-256 prefix so the cache never
# holds plaintext tokens. TTL matches the fleet standard (5 minutes).
_license_cache: TTLCache[str, dict[str, Any]] = TTLCache(maxsize=1000, ttl=300)
# Negative cache for ``active: false`` so a revoked token can't pin every
# request against the licenses endpoint. Short TTL so re-issuance is picked up.
_license_neg_cache: TTLCache[str, dict[str, Any]] = TTLCache(
    maxsize=1000, ttl=int(os.getenv("LICENSE_INTROSPECT_NEG_TTL", "60"))
)

# Per-token single-flight locks. A cold-cache burst of concurrent requests for
# the SAME token would otherwise each miss the cache and independently POST the
# licenses endpoint; on a CPU-saturated box that stampede starves the loop and
# trips the introspect timeout → spurious 401. The lock collapses the burst
# into one upstream call; the rest read the cache.
_introspect_locks: dict[str, asyncio.Lock] = {}

# Wall-clock budget for a single introspection POST. Default 10s: on a
# CPU-saturated host the loop can starve long enough that a tighter budget
# fires before the licenses endpoint replies → spurious 401.
LICENSE_INTROSPECT_TIMEOUT = float(os.getenv("LICENSE_INTROSPECT_TIMEOUT", "10.0"))


def _license_introspect_url() -> str:
    """Resolve the introspection endpoint each call so test overrides win."""
    return os.getenv("LICENSE_INTROSPECT_URL", "")


def _allow_trusted_network_default_tenant() -> bool:
    """Whether a header-less trusted peer may still be served as ``DEFAULT_TENANT_ID``.

    Read at CALL time, not import time, matching ``_license_introspect_url`` and
    ``auth.vouch.onbehalf_subjects`` — a gitops env flip must take effect on the
    next request, not require the module to be re-imported, and the burn-down
    tests must be able to drive both positions.

    Defaults to True (today's behaviour). See
    ``ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT_ENV`` for why closing it is a separate
    step gated on the ``TENANT_FALLBACK`` log going quiet.

    An unknown spelling refuses boot; one that bypassed ``Settings`` is read as
    CLOSED (False) — fail closed, never a silent open.
    """
    try:
        return parse_auth_flag(
            ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT_ENV, os.getenv(ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT_ENV), True
        )
    except ValueError:
        return False


def _tokenless_identity_enforced() -> bool:
    """``AUTH_TOKENLESS_IDENTITY_MODE`` == enforce, read at CALL time.

    ``config.Settings`` validates the value at boot, so an unknown spelling never
    reaches a request — the process refuses to start. Should one get here anyway
    (an env that bypassed Settings), it is treated as ENFORCE: fail closed, never
    a silent observe.
    """
    try:
        return parse_tokenless_identity_mode(os.getenv(TOKENLESS_IDENTITY_MODE_ENV)) == TOKENLESS_IDENTITY_MODE_ENFORCE
    except ValueError:
        return True


def _is_bearer_scheme(value: str | None) -> bool:
    """True iff ``value`` is the Bearer scheme itself (``Bearer`` / ``Bearer <tok>``).

    A tenant NAME that merely starts with "bearer" (``bearerco``) is not the scheme.
    """
    low = (value or "").strip().lower()
    return low == "bearer" or low.startswith("bearer ")


def _authorization_bearer(request: Request) -> str | None:
    """The token in ``Authorization: Bearer <token>`` (scheme case-insensitive), or None."""
    raw = (request.headers.get("Authorization") or "").strip()
    if raw[:7].lower() != "bearer ":
        return None
    return raw[7:].strip() or None


def _cached_license(cache_key: str) -> dict[str, Any] | None:
    """Return a positive- or negative-cache hit for ``cache_key`` (or ``None``)."""
    hit = _license_cache.get(cache_key)
    if hit is not None:
        return hit
    return _license_neg_cache.get(cache_key)


def _is_trusted_network(host: str | None) -> bool:
    """RFC1918 / loopback peer check. IP addresses ONLY — never a name.

    ⚠️ This used to short-circuit ``host == "testclient"`` to True so Starlette's
    ``TestClient`` (whose default ASGI peer is the literal string ``"testclient"``)
    landed on the trusted path. A test-only escape hatch in production auth code
    is a latent bypass, and this one had a concrete route to exploitability:

    - ``request.client.host`` is the ASGI peer, not a header, so ``Host:
      testclient`` can never reach it directly. Verified against the live pod: an
      unauthenticated probe carrying a forged ``X-Forwarded-For`` produced no
      rewrite.
    - BUT uvicorn ships ``proxy_headers=True`` by default with
      ``forwarded_allow_ips="127.0.0.1"``, ``src/server.py`` passes neither, and
      uvicorn's ``X-Forwarded-For`` parser does NOT validate that the forwarded
      value is an IP address. A peer of ``127.0.0.1`` sending
      ``X-Forwarded-For: testclient`` therefore rewrites ``scope["client"]`` to
      ``("testclient", 0)`` and lands here.
    - That peer does not occur today: the namespace runs in ambient mesh, which
      preserves the real source IP, and ``FORWARDED_ALLOW_IPS`` is unset in the
      pod. It becomes live the moment any one of those changes — a loopback
      sidecar, an added ``FORWARDED_ALLOW_IPS``, or leaving ambient mode. None of
      those is an auth change, so none would be reviewed as one.

    Tests inject a REAL loopback peer instead (``TestClient(app,
    client=("127.0.0.1", 50000))``). ``tests/unit/test_auth.py`` pins that this
    name is no longer trusted, in any casing.
    """
    if not host:
        return False
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(addr in net for net in TRUSTED_NETWORKS)


def _is_public_path(path: str) -> bool:
    """Exact-set public-path check (plus docs/redoc/openapi prefixes).

    Deliberately NOT a suffix match: a suffix match ("ends with /health")
    would let ``/api/formal/anything-health`` skip auth. Kept identical to the
    pre-introspection behaviour.
    """
    clean = path.rstrip("/") or "/"
    if clean in PUBLIC_ENDPOINTS:
        return True
    return clean.startswith("/docs") or clean.startswith("/redoc") or clean.startswith("/openapi.json")


def _is_protected_path(path: str) -> bool:
    clean = path.rstrip("/") or "/"
    return any(clean.startswith(prefix) for prefix in PROTECTED_PREFIXES)


def _extract_license_token(request: Request) -> str | None:
    """Pull the license token out of ``X-Tenant-Id: Bearer <token>`` only."""
    x_tenant_id = request.headers.get("X-Tenant-Id", "")
    if x_tenant_id and x_tenant_id.lower().startswith("bearer "):
        return x_tenant_id[7:].strip() or None
    return None


async def _introspect_license(token: str) -> dict[str, Any] | None:
    """Call the licenses service to validate ``token``.

    ``active`` payloads are cached 5 min; ``active: false`` payloads are cached
    briefly in the negative cache. A cold-cache burst for the same token is
    collapsed into a single upstream POST via a per-token single-flight lock.
    Returns ``None`` (→ 401) when ``LICENSE_INTROSPECT_URL`` is unset, so a
    bearer caller fails closed while raw-uuid pod-to-pod callers keep routing.
    """
    cache_key = hashlib.sha256(token.encode()).hexdigest()[:16]
    hit = _cached_license(cache_key)
    if hit is not None:
        return hit

    url = _license_introspect_url()
    if not url:
        logger.warning("LICENSE_INTROSPECT_URL not configured; rejecting bearer token")
        return None

    # Cache miss: single-flight so a concurrent burst for the same cold token
    # collapses into ONE upstream introspection. Re-check inside the lock — a
    # prior waiter may have populated the cache while we were blocked.
    lock = _introspect_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        hit = _cached_license(cache_key)
        if hit is not None:
            return hit
        return await _introspect_license_uncached(url, token, cache_key)


async def _introspect_license_uncached(url: str, token: str, cache_key: str) -> dict[str, Any] | None:
    """Perform the introspection POST and populate the caches.

    Always called under the per-token single-flight lock held by
    :func:`_introspect_license`.
    """
    try:
        async with httpx.AsyncClient(timeout=LICENSE_INTROSPECT_TIMEOUT) as client:
            resp = await client.post(url, json={"token": token})
    except Exception as e:  # noqa: BLE001 — network failures shouldn't 500 the request
        logger.warning("license_introspect_failed: %s", str(e))
        return None

    if resp.status_code != 200:
        return None
    try:
        data: dict[str, Any] = resp.json()
    except ValueError:
        return None
    if data.get("active"):
        _license_cache[cache_key] = data
    else:
        _license_neg_cache[cache_key] = data
    return data


def _error_payload(code: str, message: str, http_status: int, request_id: str) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "status": http_status,
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    }


def _unauthorized(message: str, request: Request) -> JSONResponse:
    request_id = str(getattr(request.state, "request_id", "") or "")
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content=_error_payload("AUTHENTICATION_REQUIRED", message, 401, request_id),
        headers={"WWW-Authenticate": "Bearer"},
    )


def _error(
    http_status: int, code: str, message: str, request: Request, headers: dict[str, str] | None = None
) -> JSONResponse:
    request_id = str(getattr(request.state, "request_id", "") or "")
    return JSONResponse(
        status_code=http_status,
        content=_error_payload(code, message, http_status, request_id),
        headers=headers,
    )


def _bad_request(code: str, message: str, request: Request) -> JSONResponse:
    request_id = str(getattr(request.state, "request_id", "") or "")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=_error_payload(code, message, 400, request_id),
    )


# The resolved tenant id is interpolated VERBATIM into structured identifiers all
# over this process, each with its own delimiter:
#
#   services/cache.py       ``subgraph:<tenant>:<entity>:<hops>``     delimiter ``:``
#   services/history_service ``history/<tenant>/<entity>/``           delimiter ``/``
#   context.tenant_id_var -> the Postgres RLS GUC ``app.current_tenant``
#   clients/*               outbound ``X-Tenant-Id`` on every downstream call
#
# A tenant id carrying one of those delimiters is a grammar-injection primitive,
# not an exotic name. Reproduced end-to-end before this check existed: tenant
# ``acme:eu`` wrote ``subgraph:acme:eu:e1:2``, which parses back as tenant
# ``acme`` — so ``POST /api/cache/clear`` as ``acme`` bulk-deleted ``acme:eu``'s
# entries (200, previous_size=1), while ``acme:eu`` was 403-ed from invalidating
# its OWN key. A ``/`` does the same to the MinIO history prefix.
#
# Constrain the id ONCE, here, at the single point it enters the process, instead
# of escaping it at each interpolation site: escaping fixes one grammar and every
# future site has to remember to do it again. Fails loud (400, named reason)
# rather than silently producing a well-formed-but-wrong identifier.
#
# The charset is deliberately conservative and covers every tenant id this system
# actually issues: UUIDs (``DEFAULT_TENANT_ID``), and the ``tenant-abc`` /
# ``tenant_from_license`` style ids used by callers and tests. It excludes the two
# delimiters above plus whitespace, control characters and header-injection bytes.
_TENANT_ID_RE = re.compile(r"\A[A-Za-z0-9._-]{1,128}\Z")


def _is_valid_tenant_id(tenant_id: str) -> bool:
    """True if ``tenant_id`` is safe to interpolate into every id grammar above."""
    return bool(_TENANT_ID_RE.match(tenant_id))


@contextlib.contextmanager
def _bind_identity(
    tenant_id: str, tenant_source: str, license_id: str | None, license_token: str | None
) -> Iterator[None]:
    """Bind the request's tenant, its provenance AND its licence for the call.

    They are set and reset TOGETHER on purpose. ``context.tenant_id_var`` drives
    the Postgres RLS GUC (``app.current_tenant``); ``auth.tenant.tenant_source_var``
    decides whether the outbound ML-gateway hop may carry
    ``X-Effective-Tenant-Id`` (which unlocks that tenant's BYO credentials and
    bills it); ``auth.tenant.license_id_var`` is what every outbound client stamps
    as ``X-License-Id`` so the downstream's cost event can be attributed to the
    licence that actually incurred it. A tenant carrying another request's
    provenance — or another request's LICENCE, which would bill a real customer
    for a stranger's tokens — is precisely the forgery this trio exists to
    prevent, so there is deliberately no way to bind the tenant without stating
    both: all three parameters are required.

    ``license_id`` is ``None`` for every caller that presented no licence (the
    whole trusted-network path). That is an honest NULL and it is carried as one:
    the header is then omitted downstream rather than backfilled.

    The ``tenant_id_var`` half is byte-identical to the pre-WS-4 binding, so the
    RLS GUC contract is unchanged.
    """
    tok_tenant = tenant_id_var.set(tenant_id)
    tok_source = tenant_source_var.set(tenant_source)
    tok_license = license_id_var.set(license_id)
    tok_license_token = license_token_var.set(license_token)
    try:
        yield
    finally:
        license_token_var.reset(tok_license_token)
        license_id_var.reset(tok_license)
        tenant_source_var.reset(tok_source)
        tenant_id_var.reset(tok_tenant)


# ---------------------------------------------------------------------------
# Tenant-source measurement (``scientific_processor_tenant_source_total`` +
# the TENANT_SOURCE log line)
# ---------------------------------------------------------------------------
# ⭐ MEASUREMENT ONLY. Nothing below changes which tenant is bound, whether a
# request is served, or what it returns; every helper is wrapped so that a
# failure to RECORD can never become a failure to SERVE. See ``metrics.py``
# (``TENANT_SOURCE``) for the label vocabulary and why each label is bounded.
#
# The point is to enumerate the callers that rely on a credential-less plain
# ``X-Tenant-Id`` (``source="plain_header_trusted"``) before that path is
# closed. The counter gives VOLUME per bounded bucket; the log line names the
# PEER behind each bucket so Loki can attribute it to a workload.

#: The first segment under ``/api`` of every protected prefix. A test pins this
#: against ``PROTECTED_PREFIXES`` and the routers ``app.py`` mounts, so a new
#: router that is not added here fails CI rather than silently reading as
#: ``other``.
SCIENTIFIC_ROUTE_GROUPS: frozenset[str] = frozenset(
    {
        "cache",
        "explain",
        "formal",
        "pipeline",
        "sandbox",
        "stats",
        "topology",
        "verify",
    }
)

#: ``secrets.token_urlsafe(32)`` — the shape the licenses service mints.
_LICENCE_SHAPE = re.compile(r"[A-Za-z0-9_-]{43}")
#: Three base64url segments, header starting ``{"`` → ``eyJ``.
_JWT_SHAPE = re.compile(r"eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*")
#: Anything outside this set is replaced before a caller-supplied value reaches
#: a log line (no CR/LF/space injection, no ``key=value`` forgery).
_AUDIT_UNSAFE = re.compile(r"[^A-Za-z0-9._:/@+()-]")
_AUDIT_MAX_LEN = 128

#: One TENANT_SOURCE line per (source, peer, route_group) per hour.
_TENANT_SOURCE_LOG_INTERVAL_SECONDS = 3600.0
#: Bounded: pod IPs churn on every rollout. Clear-on-full costs one extra line
#: per caller, never an unbounded table.
_TENANT_SOURCE_LOG_MAX_KEYS = 1024
_tenant_source_log_seen: dict[tuple[str, str, str], float] = {}

#: Sources whose detail line is emitted here. The refusal sources are
#: deliberately absent: they are reachable with no credential from outside the
#: cluster, so a per-peer line for them would be a log-amplification surface.
#: Their volume is still counted.
_TENANT_SOURCE_LOGGED = frozenset(
    {
        "licence_x_tenant_id",
        "licence_authorization",
        "licence_authorization_onbehalf",
        "plain_header_vouched",
        "plain_header_trusted",
        "default_fallback",
    }
)


def tenant_route_group(path: str) -> str:
    """Bounded route label: the protected first segment under ``/api``."""
    parts = path.split("/")
    # "/api/x/y" -> ["", "api", "x", "y"]
    if len(parts) < 3 or parts[2] == "":
        return "root"
    return parts[2] if parts[2] in SCIENTIFIC_ROUTE_GROUPS else "other"


def authorization_carrier(value: str | None) -> str:
    """SHAPE of the ``Authorization`` header. Never introspected here, never logged."""
    raw = (value or "").strip()
    if not raw:
        return "none"
    scheme, _, rest = raw.partition(" ")
    if scheme.lower() != "bearer":
        return "non_bearer"
    cred = rest.strip()
    if _JWT_SHAPE.fullmatch(cred):
        return "bearer_jwt"
    if _LICENCE_SHAPE.fullmatch(cred):
        return "bearer_licence_shaped"
    return "bearer_other"


def _audit_field(value: object) -> str:
    """A caller-supplied value made safe and bounded for one log field."""
    if value is None:
        return "-"
    text = str(value)
    if not text:
        return "-"
    return _AUDIT_UNSAFE.sub("_", text[:_AUDIT_MAX_LEN])


def reset_tenant_source_log_gate() -> None:
    """Drop TENANT_SOURCE log dedup state. For tests and nothing else."""
    _tenant_source_log_seen.clear()


def _should_log_tenant_source(source: str, peer: str, route_group: str) -> bool:
    key = (source, peer, route_group)
    now = time.monotonic()
    last = _tenant_source_log_seen.get(key)
    if last is not None and now - last < _TENANT_SOURCE_LOG_INTERVAL_SECONDS:
        return False
    if len(_tenant_source_log_seen) >= _TENANT_SOURCE_LOG_MAX_KEYS:
        _tenant_source_log_seen.clear()
    _tenant_source_log_seen[key] = now
    return True


def _record_tenant_source(
    request: Request,
    source: str,
    *,
    has_licence: bool,
    subject: object = None,
) -> None:
    """Count this request's tenant source once; log the caller at most hourly.

    ⛔ Must never raise into ``dispatch``: measurement that can break serving is
    a behaviour change. Any failure is swallowed and logged at DEBUG.
    """
    try:
        carrier = authorization_carrier(request.headers.get("authorization"))
        route_group = tenant_route_group(request.url.path)
        TENANT_SOURCE.labels(
            source=source,
            has_licence="true" if has_licence else "false",
            carrier=carrier,
            route_group=route_group,
        ).inc()
        if source not in _TENANT_SOURCE_LOGGED:
            return
        peer = _audit_field(request.client.host if request.client else None)
        if not _should_log_tenant_source(source, peer, route_group):
            return
        logger.info(
            "TENANT_SOURCE source=%s has_licence=%s carrier=%s route_group=%s peer=%s "
            "subject=%s user_agent=%s — how this caller's tenant was resolved "
            "(logged at most once per hour per source+peer+route_group; "
            "scientific_processor_tenant_source_total is the volume)",
            source,
            "true" if has_licence else "false",
            carrier,
            route_group,
            peer,
            _audit_field(subject),
            _audit_field(request.headers.get("user-agent")),
        )
    except Exception:  # noqa: BLE001 — measurement must never break serving
        logger.debug("tenant-source measurement failed", exc_info=True)


class AuthMiddleware(BaseHTTPMiddleware):
    """License-token auth with public/trusted-network exits.

    Binds the resolved tenant into ``tenant_id_var`` for the lifetime of the
    downstream call so ``register_tenant_guc`` scopes Postgres RLS
    (``app.current_tenant``) on every transaction, and its WS-4 provenance into
    ``tenant_source_var`` in lockstep.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path

        if _is_public_path(path) or not _is_protected_path(path):
            return await call_next(request)

        # NB: `x-subject-id`, `x-user-id` and `x-user-roles` are deliberately
        # NOT read anywhere below — they are client-supplied on every remaining
        # path, so they can never be identity. `X-Tenant-Name` is display
        # context only.
        x_tenant_name = request.headers.get("X-Tenant-Name", "")

        token = _extract_license_token(request)
        if not token and _is_bearer_scheme(request.headers.get("X-Tenant-Id")):
            # `X-Tenant-Id: Bearer` with no token. Header parsing strips the
            # trailing space, so this used to miss the licence path AND the
            # trusted branch's `startswith("bearer ")` guard and was bound as a
            # plain tenant literally named "Bearer". A credential that is
            # present but empty is malformed — a 401 MALFORMED_CREDENTIAL in
            # every mode, from any peer, never introspected.
            _record_tenant_source(request, "malformed_credential", has_licence=False)
            return _error(
                status.HTTP_401_UNAUTHORIZED,
                "MALFORMED_CREDENTIAL",
                "X-Tenant-Id carries the Bearer scheme with an empty token",
                request,
                headers={"WWW-Authenticate": "Bearer"},
            )
        if token:
            data = await _introspect_license(token)
            if not data or not data.get("active"):
                _record_tenant_source(request, "licence_invalid", has_licence=False)
                return _unauthorized("Invalid or expired license token", request)

            tenant_id = data.get("tenant_id")
            if not tenant_id:
                _record_tenant_source(
                    request, "licence_missing_tenant", has_licence=True, subject=data.get("subject_id")
                )
                return _unauthorized("License token missing tenant context", request)
            if not _is_valid_tenant_id(str(tenant_id)):
                # Same delimiter-injection class as the trusted path below; a
                # license is more trustworthy than a header but not a reason to
                # skip the check that keeps every id grammar unambiguous.
                logger.warning("license_tenant_id_malformed")
                _record_tenant_source(
                    request, "licence_malformed_tenant", has_licence=True, subject=data.get("subject_id")
                )
                return _unauthorized("License token carries a malformed tenant id", request)

            # Subject/roles come from the LICENSE payload or not at all.
            subject_id = data.get("subject_id")
            subject_str: str = str(subject_id) if subject_id else ""
            roles = list(data.get("roles", []) or [])
            user = User(
                sub=subject_str or str(tenant_id),
                preferred_username=str(data.get("subject_username") or data.get("tenant_name") or "license"),
                tenant_id=str(tenant_id),
                subject_id=subject_str or None,
                tenant_name=str(data.get("tenant_name") or x_tenant_name) or None,
                realm_access={"roles": roles},
            )
            # The tenant came out of an INTROSPECTED license — self-authenticating
            # provenance, and the only one that needs no vouch.
            _record_tenant_source(request, "licence_x_tenant_id", has_licence=True, subject=subject_str)
            return await self._call_with_tenant(
                request,
                call_next,
                user,
                data.get("license_id"),
                data,
                tenant_source=SOURCE_LICENSE_TOKEN,
                license_token=token,
            )

        # A licence in `Authorization` (only when X-Tenant-Id is not itself a
        # Bearer — that carrier keeps precedence). ⛔ ENFORCE ONLY: in observe
        # nothing here runs — no introspection, no binding — so every caller
        # gets exactly origin/main's answer.
        if _tokenless_identity_enforced():
            authorization_token = _authorization_bearer(request)
            if authorization_token:
                return await self._dispatch_authorization_licence(request, call_next, authorization_token)

        # No token — RFC1918 / loopback bypass for in-cluster
        # traffic. Callers (discovery-engine, kag proxy) set a raw
        # X-Tenant-Id: <uuid>; accept it for tenant ROUTING only.
        client_host = request.client.host if request.client else None
        if _is_trusted_network(client_host):
            raw_tenant = request.headers.get("X-Tenant-Id", "").strip()
            if raw_tenant.lower().startswith("bearer "):
                # A bearer value that failed the token path above must not be
                # mistaken for a raw uuid tenant.
                raw_tenant = ""
            if not raw_tenant and not _allow_trusted_network_default_tenant():
                # The closed position of the migration switch. Only reachable
                # once gitops has flipped ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT
                # false, which is only safe once TENANT_FALLBACK has gone quiet.
                _record_tenant_source(request, "default_fallback_refused", has_licence=False)
                return _unauthorized(
                    "Missing tenant context: name a tenant in X-Tenant-Id or present a license token",
                    request,
                )
            tenant_id = raw_tenant or DEFAULT_TENANT_ID
            if not _is_valid_tenant_id(tenant_id):
                # Refuse at the door rather than letting a delimiter-bearing id
                # reach the cache-key / MinIO-prefix builders. See _TENANT_ID_RE.
                _record_tenant_source(request, "invalid_tenant_id", has_licence=False)
                return _bad_request(
                    "INVALID_TENANT_ID",
                    "X-Tenant-Id must match [A-Za-z0-9._-]{1,128}",
                    request,
                )

            if raw_tenant and _tokenless_identity_enforced():
                # AUTH_TOKENLESS_IDENTITY_MODE=enforce: a caller that NAMES a
                # tenant must prove it with a licence (X-Tenant-Id: Bearer, or
                # Authorization: Bearer above). Reaching here means it did not.
                _record_tenant_source(request, "plain_header_refused", has_licence=False)
                return _unauthorized(
                    "A plain X-Tenant-Id is not accepted without a licence "
                    "(AUTH_TOKENLESS_IDENTITY_MODE=enforce): send X-Tenant-Id: Bearer <licence> "
                    "or Authorization: Bearer <licence>",
                    request,
                )

            # WS-4: the tenant VALUE is unchanged either way — this only records
            # how much scientific-processor can stand behind it. A bare
            # X-Tenant-Id from a pod-CIDR peer is `unvouched-header` (accepted
            # as a data scope, never delegated to the metered gateway); an
            # allowlisted service that proves its OWN identity in
            # `Authorization: Bearer` promotes the SAME tenant to
            # `vouched-onbehalf`. Vouching can only WIDEN: `authorized_onbehalf`
            # never raises and a failed vouch leaves the request accepted
            # exactly as before.
            if raw_tenant:
                tenant_source = (
                    SOURCE_VOUCHED_ONBEHALF if await authorized_onbehalf(request) else SOURCE_UNVOUCHED_HEADER
                )
                if tenant_source == SOURCE_VOUCHED_ONBEHALF:
                    _record_tenant_source(request, "plain_header_vouched", has_licence=True)
                else:
                    _record_tenant_source(request, "plain_header_trusted", has_licence=False)
            else:
                # This service's OWN default tenant, not an assertion by the caller.
                tenant_source = SOURCE_TRUSTED_NETWORK_DEFAULT
                _record_tenant_source(request, "default_fallback", has_licence=False)
                # THE MIGRATION BURN-DOWN SIGNAL, and the reason this whole
                # branch is survivable.
                #
                # DEFAULT_TENANT_ID is not a placeholder here: measured live,
                # 37 of the 38 cluster workloads that set it point at the SAME
                # REAL customer tenant 0195f48b-d000-7000-8000-000000000001. So
                # every anonymous in-cluster caller that reaches this line has
                # its work — and its downstream token spend — booked against a
                # real paying customer who never made the request. That is
                # mis-billing, silently, with nothing in the response to show it.
                #
                # Substituting a different value is NOT the fix and must not be
                # attempted here: the nil uuid is quarantined by billing_service
                # as malformed (`_cost_schema.py`), which turns mis-billing into
                # UN-billing, and "n8n-dev" resolves to another real tenant. The
                # fix is to credential these callers, and this WARN is what makes
                # that possible — it names the peer and the path, so the set of
                # callers still on the fallback can be ENUMERATED from Loki
                # rather than guessed at. Only when it stops being emitted is
                # ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT safe to close.
                #
                # Emitted ONLY on the header-less branch: a caller that named its
                # own tenant is not on the fallback and logging it would bury the
                # signal in noise it can never burn down.
                #
                # A WARN line because it names the PEER, which a bounded
                # Prometheus label must not. Its volume is counted separately
                # (`scientific_processor_tenant_source_total{source="default_fallback"}`).
                # Keep TENANT_FALLBACK a stable, greppable token.
                logger.warning(
                    "TENANT_FALLBACK peer=%s path=%s tenant=%s — no X-Tenant-Id and no license "
                    "token; served as DEFAULT_TENANT_ID, which is a REAL customer tenant, so this "
                    "caller's downstream spend is billed to someone who did not make the request. "
                    "Credential this caller before ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT is closed.",
                    client_host or "unknown",
                    path,
                    tenant_id,
                )

            # NEVER derive roles/subject/user-id from client-supplied headers:
            # every pod is inside the trusted CIDR, so that would be forgery.
            user = User(
                sub=tenant_id,
                preferred_username="service",
                tenant_id=tenant_id,
                realm_access={"roles": []},
                subject_id=None,
                tenant_name=x_tenant_name or None,
            )
            return await self._call_with_tenant(
                request, call_next, user, None, None, tenant_source=tenant_source, license_token=None
            )

        _record_tenant_source(request, "missing", has_licence=False)
        return _unauthorized("Missing license token", request)

    async def _dispatch_authorization_licence(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
        token: str,
    ) -> Response:
        """Authenticate by a licence in ``Authorization: Bearer`` (procedure_management #116 port).

        ⛔ Called ONLY under ``AUTH_TOKENLESS_IDENTITY_MODE=enforce``. In observe
        this carrier does not exist: a forwarded Keycloak JWT introspected as a
        licence is a 401 (the 2026-04-26 fleet incident) and a network call on
        every JWT-bearing request — observe-mode costs nobody signed up for.

        ===========================  ==========  ===================================
        plain X-Tenant-Id            vouched?    outcome
        ===========================  ==========  ===================================
        absent / blank               --          licence tenant
        equal to licence tenant_id   --          licence tenant
        differs                      yes         the asserted tenant (on-behalf)
        differs                      no          403 TENANT_ASSERTION_NOT_PERMITTED
        fails the id grammar         --          400 (as the trusted path)
        ===========================  ==========  ===================================

        Introspection with NO verdict (unset URL, transport error, non-200,
        unparseable body) is a 503 + ``Retry-After``: not the caller's fault, so
        not a 401. ``active: false`` (a JWT, a revoked licence) is a 401. The
        tenant is the licence's ``tenant_id``, never its ``subject_id``.
        "Vouched" = introspected ``subject_id`` on
        ``SCIENTIFIC_PROCESSOR_ONBEHALF_SUBJECTS`` — the same allowlist
        ``auth/vouch.py`` uses, keyed on what introspection returned.
        """
        try:
            data: Any = await _introspect_license(token)
        except ValueError, AttributeError:
            # A 200 whose body is not a JSON object: no verdict, not a 500.
            data = None
        if not isinstance(data, dict):
            logger.warning("auth.introspection_unavailable carrier=authorization")
            _record_tenant_source(request, "licence_authorization_unavailable", has_licence=False)
            return _error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "LICENSE_INTROSPECTION_UNAVAILABLE",
                "License introspection is unavailable; the token was not checked. "
                "This is not a problem with the request.",
                request,
                headers={"Retry-After": "5"},
            )
        if not data.get("active"):
            _record_tenant_source(request, "licence_authorization_invalid", has_licence=False)
            return _unauthorized(
                "Authorization bearer is not a valid licence token (AUTH_TOKENLESS_IDENTITY_MODE=enforce)",
                request,
            )

        raw_subject = data.get("subject_id")
        subject_str = str(raw_subject).strip() if raw_subject is not None else ""
        licence_tenant = data.get("tenant_id")
        if not licence_tenant or not _is_valid_tenant_id(str(licence_tenant)):
            _record_tenant_source(
                request, "licence_authorization_malformed_tenant", has_licence=True, subject=subject_str
            )
            return _unauthorized("License token carries no usable tenant context", request)
        licence_tenant = str(licence_tenant)

        acting_tenant = licence_tenant
        asserted = request.headers.get("X-Tenant-Id", "").strip()
        if asserted:
            if not _is_valid_tenant_id(asserted):
                _record_tenant_source(request, "invalid_tenant_id", has_licence=True, subject=subject_str)
                return _bad_request(
                    "INVALID_TENANT_ID",
                    "X-Tenant-Id must match [A-Za-z0-9._-]{1,128}",
                    request,
                )
            if asserted != licence_tenant:
                if not (subject_str and subject_str in onbehalf_subjects()):
                    logger.warning(
                        "auth.authorization_licence_tenant_mismatch subject=%s path=%s",
                        _audit_field(subject_str or None),
                        _audit_field(request.url.path),
                    )
                    _record_tenant_source(
                        request, "licence_authorization_foreign_tenant", has_licence=True, subject=subject_str
                    )
                    return _error(
                        status.HTTP_403_FORBIDDEN,
                        "TENANT_ASSERTION_NOT_PERMITTED",
                        "This licence may not assert a tenant other than its own",
                        request,
                    )
                acting_tenant = asserted

        x_tenant_name = request.headers.get("X-Tenant-Name", "")
        if acting_tenant != licence_tenant:
            # Delegation: an allow-listed SERVICE acting for the tenant it named.
            # Same footing as today's trusted-path vouch (`vouched-onbehalf`, no
            # licence id / token bound — the licence is the service's, not the
            # tenant's, so it must not be billed or forwarded as theirs).
            _record_tenant_source(request, "licence_authorization_onbehalf", has_licence=True, subject=subject_str)
            user = User(
                sub=acting_tenant,
                preferred_username="service",
                tenant_id=acting_tenant,
                realm_access={"roles": []},
                subject_id=None,
                tenant_name=x_tenant_name or None,
            )
            return await self._call_with_tenant(
                request, call_next, user, None, None, tenant_source=SOURCE_VOUCHED_ONBEHALF, license_token=None
            )

        roles = list(data.get("roles", []) or [])
        user = User(
            sub=subject_str or licence_tenant,
            preferred_username=str(data.get("subject_username") or data.get("tenant_name") or "license"),
            tenant_id=licence_tenant,
            subject_id=subject_str or None,
            tenant_name=str(data.get("tenant_name") or x_tenant_name) or None,
            realm_access={"roles": roles},
        )
        _record_tenant_source(request, "licence_authorization", has_licence=True, subject=subject_str)
        return await self._call_with_tenant(
            request,
            call_next,
            user,
            data.get("license_id"),
            data,
            tenant_source=SOURCE_LICENSE_TOKEN,
            license_token=token,
        )

    async def _call_with_tenant(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
        user: User,
        license_id: Any,
        license_data: dict[str, Any] | None,
        *,
        tenant_source: str,
        license_token: str | None,
    ) -> Response:
        """Stash identity on request.state and bind the RLS tenant GUC var.

        ``tenant_source`` is keyword-only with NO default so a future branch
        cannot bind a tenant without saying where it came from.
        """
        # Normalise the licence ONCE, here, at the single point it enters the
        # process. The introspection payload is untyped JSON, so `license_id` can
        # arrive as an int, as `""`, or as `"  "` — all of which are "no licence"
        # but only `None` and `""` are falsy. A whitespace-only value would sail
        # through `outbound_headers`' truthiness check and put a blank
        # `X-License-Id` on the wire, which billing_service reads as a present
        # but unresolvable licence: strictly worse than the honest omission,
        # because it is no longer counted as unattributed.
        license_str: str | None = (str(license_id).strip() or None) if license_id is not None else None
        request.state.user = user
        request.state.tenant_id = user.effective_tenant_id
        request.state.subject_id = user.subject_id
        request.state.tenant_name = user.effective_tenant_name
        request.state.license_id = license_str
        request.state.license = license_data
        # Per-request record of the provenance, alongside the tenant it belongs
        # to. `tests/unit/test_ws4_vouched_delegation.py` pins that this and
        # `tenant_source_var` never disagree; the ContextVar is what the outbound
        # ML client actually reads (a fire-and-forget task inherits the
        # `contextvars` snapshot taken at `asyncio.create_task` time, so it does
        # NOT need the value threaded through by hand).
        request.state.tenant_source = tenant_source
        # Bind tenant for RLS (register_tenant_guc reads tenant_id_var per tx)
        # plus its provenance and its licence, in lockstep. The licence half is
        # what makes the outbound `X-License-Id` possible at all: it is read by
        # every client in `src/clients/`, including from the SubgraphX
        # fire-and-forget task, which has no request to read `request.state` from.
        with _bind_identity(user.effective_tenant_id, tenant_source, license_str, license_token):
            return await call_next(request)
