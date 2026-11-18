"""Prometheus metrics for scientific-processor.

The service exposes no ``/metrics`` endpoint (``telemetry.py`` wires OTLP
tracing only), deliberately: an unprotected route on the app port is needless
exposure. The counter below lives in the default in-process registry, ready
for whenever scraping is wired; until then the deduped ``TENANT_SOURCE`` log
line (Loki) is the measurement.

Tenant source — how each protected request's tenant was resolved
---------------------------------------------------------------
⭐ WHY THIS EXISTS. ``AuthMiddleware`` honours a credential-less plain
``X-Tenant-Id`` from any peer inside ``TRUSTED_CIDRS`` (the whole pod CIDR) and
binds it as the tenant (``unvouched-header``). Before this counter the only
signal was the ``TENANT_FALLBACK`` WARN, which covers the header-LESS branch and
says nothing about callers that NAME a tenant without proving it — so there was
no way to measure who relies on that path, and therefore no way to close it
without an untested blast radius. This is MEASUREMENT ONLY: nothing about how a
request is resolved, served or refused changes.

Incremented EXACTLY ONCE per request on a protected (``PROTECTED_PREFIXES``)
path, at the exit that decided it. Public and unprotected paths resolve no
tenant and are not counted.

⛔ BOUNDED LABELS ONLY — no raw path, no tenant id, no peer, no subject. Protected
paths are reachable unauthenticated, so any caller-controlled label would let a
flood mint unbounded series. Peer and subject go in the rate-limited
``TENANT_SOURCE`` log line instead.

source:
  licence_x_tenant_id       licence in ``X-Tenant-Id: Bearer``, introspected active;
                            tenant = the licence's own tenant.
  plain_header_vouched      no licence in ``X-Tenant-Id``, trusted peer, bare tenant
                            promoted to ``vouched-onbehalf`` by an allow-listed
                            service licence in ``Authorization``.
  plain_header_trusted      ⛔ THE HOLE: no licence, trusted-CIDR peer, bare
                            ``X-Tenant-Id`` accepted verbatim (``unvouched-header``).
  default_fallback          no licence, no tenant header, trusted peer; served as
                            DEFAULT_TENANT_ID.
  malformed_credential      ``X-Tenant-Id: Bearer`` with an EMPTY token → 401,
                            every mode (was served as a tenant named "Bearer").
  default_fallback_refused  as above with ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT
                            closed → 401.
  invalid_tenant_id         trusted peer, tenant fails the id grammar → 400.
  licence_invalid           licence presented, introspection not active → 401.
  licence_missing_tenant    licence active but tenant-less → 401.
  licence_malformed_tenant  licence active, tenant fails the id grammar → 401.
  missing                   no licence, untrusted peer → 401.

  Reachable ONLY under ``AUTH_TOKENLESS_IDENTITY_MODE=enforce``:
  plain_header_refused      trusted peer, plain ``X-Tenant-Id``, no licence → 401.
  licence_authorization     licence in ``Authorization``, active; tenant = its own
                            (a plain ``X-Tenant-Id`` beside it, if any, equal).
  licence_authorization_onbehalf
                            as above, plain ``X-Tenant-Id`` names ANOTHER tenant
                            and the licence subject is on the on-behalf allowlist.
  licence_authorization_foreign_tenant
                            as above, subject NOT allow-listed → 403.
  licence_authorization_invalid
                            ``Authorization`` bearer introspected not active
                            (a Keycloak JWT, a revoked licence) → 401.
  licence_authorization_malformed_tenant
                            active licence, tenant missing / fails grammar → 401.
  licence_authorization_unavailable
                            introspection gave no verdict → 503 + Retry-After.

has_licence: "true" iff a licence presented on this request was introspected
  ACTIVE (in ``X-Tenant-Id``, or the vouching licence in ``Authorization``).
  Shape is never enough to say "true".

carrier: the SHAPE of the ``Authorization`` header. Shape only — this module
  never introspects it and never logs it:
    none | bearer_licence_shaped | bearer_jwt | bearer_other | non_bearer
  A ``plain_header_trusted`` series with ``carrier="bearer_licence_shaped"`` is a
  caller that already HOLDS a licence and sends it in the wrong header — the
  cheapest caller to fix before the hole closes.

route_group: the first path segment under ``/api`` when it is one of the
  service's protected prefixes, else ``other`` (``root`` for a bare ``/api``).
"""

from __future__ import annotations

from prometheus_client import Counter

TENANT_SOURCES: tuple[str, ...] = (
    "licence_x_tenant_id",
    "plain_header_vouched",
    "plain_header_trusted",
    "default_fallback",
    "default_fallback_refused",
    "invalid_tenant_id",
    "licence_invalid",
    "licence_missing_tenant",
    "licence_malformed_tenant",
    "missing",
    "malformed_credential",
    # AUTH_TOKENLESS_IDENTITY_MODE=enforce only
    "plain_header_refused",
    "licence_authorization",
    "licence_authorization_onbehalf",
    "licence_authorization_foreign_tenant",
    "licence_authorization_invalid",
    "licence_authorization_malformed_tenant",
    "licence_authorization_unavailable",
)

TENANT_SOURCE_CARRIERS: tuple[str, ...] = (
    "none",
    "bearer_licence_shaped",
    "bearer_jwt",
    "bearer_other",
    "non_bearer",
)

TENANT_SOURCE = Counter(
    "scientific_processor_tenant_source_total",
    "Protected requests by how the tenant was resolved (measurement only).",
    ["source", "has_licence", "carrier", "route_group"],
)
