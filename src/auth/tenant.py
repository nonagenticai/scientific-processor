"""Tenant PROVENANCE — how the request's tenant was established, and whether
scientific-processor may DELEGATE it to the metered modality-gateway.

WHY THIS MODULE EXISTS
----------------------
scientific-processor's service-account subject
``019f83f2-ddd1-779f-9c70-7a0c6de1dacb`` (svc-scientific-processor) is on the
modality-gateway's ``AUTH_DELEGATING_SUBJECTS``. When this service sends
``X-Effective-Tenant-Id: <T>`` the gateway ACTS AS tenant T: it resolves T's own
BYO LLM provider credentials and BILLS T.

``AuthMiddleware``'s trusted-network branch accepts a BARE, uncredentialed
``X-Tenant-Id`` from any peer inside ``TRUSTED_CIDRS`` — in production the whole
pod CIDR ``10.0.0.0/8`` — and ``clients.pytorch_backend`` forwarded that value
straight into ``X-Effective-Tenant-Id`` on both ``/api/ml/embeddings`` and
``/api/ml/gnn/inference``. Any unauthenticated pod in the cluster could
therefore name a victim tenant and spend the victim's LLM credentials.

The tenant VALUE is not what changes here — only how much this service can stand
behind it. A bare ``X-Tenant-Id`` remains a fully accepted DATA SCOPE on every
route; nothing is newly rejected. Only the DELEGATION is gated.

Ported from ``hirag_service`` (commit 65a4c7c, the "omit" variant) and ``kag``
(commit 04d7885, which introduced the vocabulary).
"""

from __future__ import annotations

import os
from contextvars import ContextVar

# ---------------------------------------------------------------------------
# This service's own default tenant
# ---------------------------------------------------------------------------
#: THE SINGLE DEFINITION. ``auth.middleware`` and ``auth.dependencies`` both used
#: to call ``os.getenv("DEFAULT_TENANT_ID", ...)`` themselves, with the literal
#: fallback uuid spelled out three times across the repo. Three independent
#: readers of one env var is three chances for them to disagree: a typo in one
#: literal, or one module reading the env at a different point in startup, and
#: the tenant a header-less request is SERVED as stops matching the tenant
#: ``_ANONYMOUS_USER`` and ``resolve_tenant_id`` fall back to — which silently
#: splits one caller's cache namespace, MinIO prefix and RLS scope in two.
#: Imported by both modules, so ``mw.DEFAULT_TENANT_ID`` /
#: ``dependencies.DEFAULT_TENANT_ID`` still resolve (and are still individually
#: monkeypatchable) exactly as before.
#:
#: ``config.Settings.default_tenant_id`` is deliberately NOT folded in here: it
#: is read through an ``lru_cache``d ``get_settings()``, so moving it would change
#: WHEN the env is frozen, which is a behaviour change unrelated to billing
#: attribution. Both must name the same env var and the same fallback.
DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "0195f48b-d000-7000-8000-000000000001")

# ---------------------------------------------------------------------------
# Provenance vocabulary
# ---------------------------------------------------------------------------
#: NB there is deliberately NO ``auth-disabled`` member. ``AuthMiddleware`` is
#: not installed at all when ``ENABLE_AUTH=false`` (see ``app.create_app``), so
#: nothing binds this ContextVar and every read returns ``SOURCE_NONE`` — which
#: is already non-delegatable. An explicit constant would have to be PRODUCED by
#: something, and the only candidate was a router dependency nobody called; a
#: named-but-unreachable provenance is how a vocabulary drifts away from what the
#: code does. Dev mode therefore cannot spend anyone's credentials, by the same
#: fail-closed default that covers background threads and startup probes.
#: An INTROSPECTED ``X-Tenant-Id: Bearer <license-token>``. Self-authenticating:
#: the licenses service already vouched for it before we ever saw the tenant.
SOURCE_LICENSE_TOKEN = "license-token"
#: THE HOLE. A bare ``X-Tenant-Id`` from a peer inside ``TRUSTED_CIDRS``, i.e.
#: any pod in the cluster, with no credential of any kind.
SOURCE_UNVOUCHED_HEADER = "unvouched-header"
#: A bare ``X-Tenant-Id`` whose caller ALSO proved its own identity in
#: ``Authorization: Bearer <service-token>`` and whose introspected subject is on
#: ``SCIENTIFIC_PROCESSOR_ONBEHALF_SUBJECTS``. See ``auth.vouch``.
SOURCE_VOUCHED_ONBEHALF = "vouched-onbehalf"
#: No ``X-Tenant-Id`` at all from a trusted peer — this service's OWN
#: ``DEFAULT_TENANT_ID``. Not attacker-chosen, but still nobody's proven
#: assertion about a customer.
SOURCE_TRUSTED_NETWORK_DEFAULT = "trusted-network-default"
#: No request bound at all — a background task, a startup probe, or a code path
#: that never went through ``AuthMiddleware``. The ContextVar default and the
#: fallback everywhere else, so an UNSET provenance can never be mistaken for a
#: proven one. Fail closed.
SOURCE_NONE = "none"

#: The provenances this service may delegate. Everything absent from this set is
#: a tenant scientific-processor ACCEPTS (so no caller is newly 401'd and no data
#: scope narrows) but will not vouch for on the metered hop.
#:
#: Deliberately EXCLUDED: ``unvouched-header`` (the forged-tenant hole),
#: ``trusted-network-default`` (this service's own default, nobody's assertion
#: about a customer), ``auth-disabled`` and ``none``.
VOUCHED_TENANT_SOURCES: frozenset[str] = frozenset({SOURCE_LICENSE_TOKEN, SOURCE_VOUCHED_ONBEHALF})

#: Request-scoped provenance, the parallel of ``context.tenant_id_var``.
#:
#: Bound and reset TOGETHER with the tenant by ``AuthMiddleware._bind_identity``
#: — the two must never drift, or a tenant would carry another request's
#: provenance, which is precisely the forgery this pair exists to prevent.
#:
#: BACKGROUND TASKS: this var DOES reach them, and that is measured, not assumed.
#: ``asyncio.create_task`` snapshots ``contextvars.copy_context()`` at CREATION
#: time and the task runs in that independent copy, so the ``reset()`` performed
#: by ``AuthMiddleware`` when the request ends cannot disturb it. The SubgraphX
#: explanation job (``services.subgraphx_service._spawn``, a fire-and-forget task
#: spawned while ``POST /api/explain/subgraphx`` is still in scope) therefore
#: observes the provenance its originating request was bound with.
#:
#: This is the same mechanism ``context.tenant_id_var`` already relies on for the
#: RLS GUC in those jobs, so provenance and tenant stay consistent by
#: construction. An EARLIER draft of this module asserted the opposite — that a
#: background read would yield ``SOURCE_NONE`` — and proposed threading the value
#: through every ``subgraphx_service`` frame. That claim was falsified directly
#: (a task spawned inside a scope that is reset immediately afterwards still
#: reads the bound value), and the plumbing it motivated was dropped.
#:
#: A path that genuinely does NOT inherit the request context — a worker started
#: at import time, a thread, or a task spawned from a different scope — reads the
#: ``SOURCE_NONE`` default and is therefore NOT delegated. Fail closed.
tenant_source_var: ContextVar[str] = ContextVar("tenant_source", default=SOURCE_NONE)

#: The LICENCE the current request was authenticated with, for cost attribution
#: on every outbound hop. ``None`` when the caller presented no licence.
#:
#: WHY THIS EXISTS
#: ---------------
#: ``AuthMiddleware`` has always resolved ``license_id`` out of the introspected
#: payload and stashed it on ``request.state.license_id`` — and that is where it
#: DIED. Grepping the repo, ``license_id`` appeared in ``auth/middleware.py`` and
#: nowhere else: no client emitted ``X-License-Id``, so every downstream hop
#: (inference-router above all, the metered LLM call) received a tenant and no
#: licence. ``billing_service`` counts exactly that shape in
#: ``billing_license_unattributed_total``: a resolved tenant whose spend cannot be
#: attributed to a licence. Fleet-wide that is what holds
#: ``billing_attribution_rate`` at 0.9212.
#:
#: WHY A ContextVar AND NOT A FUNCTION ARGUMENT
#: -------------------------------------------
#: The tenant reaches the clients as an explicit ``tenant_id=`` argument, but the
#: tenant is a DATA SCOPE that the routers already hold. The licence is
#: request IDENTITY, and this service already carries request identity to the
#: outbound hop exactly this way: ``tenant_source_var`` directly above, bound in
#: lockstep by ``AuthMiddleware._bind_identity`` and read by
#: ``clients.pytorch_backend``. Threading a licence argument instead would mean a
#: new parameter on every router, every service method and every job-manager
#: frame — and it would still not reach ``services.subgraphx_service``'s
#: fire-and-forget ``asyncio.create_task`` child, which makes two ML-gateway calls
#: with no request in scope. That plumbing was already proposed once for the
#: provenance and dropped for the same reason (see ``tenant_source_var`` above).
#: So this is the SAME mechanism extended, not a second one invented alongside it.
#:
#: HONEST NULL, NEVER A SUBSTITUTE
#: ------------------------------
#: The default is ``None`` and a ``None`` licence causes the header to be OMITTED
#: entirely (``kag_common.auth.outbound_headers`` emits ``X-License-Id`` only for
#: a truthy value). It is never backfilled with this service's own service
#: licence, with the tenant, or with a placeholder: substituting any of those
#: would move the spend onto a real licence that did not incur it, which is worse
#: than leaving it unattributed — an unattributed event is still counted and
#: still visible in ``billing_license_unattributed_total``, and that counter is
#: how anyone knows which callers still need credentialing.
license_id_var: ContextVar[str | None] = ContextVar("license_id", default=None)


def may_delegate(tenant_source: str) -> bool:
    """True iff a tenant with this provenance may be NAMED to the metered gateway.

    ``tenant_source`` is required on purpose — see
    ``clients.pytorch_backend.PyTorchBackendClient._embedding_headers``.
    """
    return tenant_source in VOUCHED_TENANT_SOURCES


#: The RAW licence token the current request was authenticated with — the end
#: tenant's own credential from ``X-Tenant-Id: Bearer <token>``. ``None`` for every
#: caller that presented no token (the trusted-network path).
#:
#: WHY: GPU backends (lean4-prover first) must receive a licence on every call so
#: the work is introspected and attributed. When the end tenant presented one it
#: is the right credential to forward — the downstream then attributes the work to
#: THAT tenant. Only when there is none does a client fall back to this service's
#: own service licence (``SCIENTIFIC_PROCESSOR_DOWNSTREAM_LICENSE_TOKEN``).
#:
#: Bound and reset in lockstep with ``license_id_var`` by
#: ``AuthMiddleware._bind_identity``; never logged.
license_token_var: ContextVar[str | None] = ContextVar("license_token", default=None)
