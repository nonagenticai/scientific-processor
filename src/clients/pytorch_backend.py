"""
HTTP client for the PyTorch Backend service (in production: the modality-gateway).

Provides access to embedding generation and GNN inference.

WS-4: WHY THE DELEGATED HOP IS GATED
------------------------------------
``X-Effective-Tenant-Id`` is not attribution metadata at the modality-gateway —
it is an IDENTITY ASSERTION. scientific-processor's SA subject
``019f83f2-ddd1-779f-9c70-7a0c6de1dacb`` (svc-scientific-processor) is on the
gateway's ``AUTH_DELEGATING_SUBJECTS``, so the gateway ACTS AS the named tenant:
it resolves THAT TENANT's BYO LLM provider credentials and BILLS that tenant.

``auth.middleware.AuthMiddleware``'s trusted-network branch accepts a bare,
uncredentialed ``X-Tenant-Id`` from any peer inside ``TRUSTED_CIDRS`` — in
production the whole pod CIDR — and ``_embedding_headers`` forwarded that value
straight into ``X-Effective-Tenant-Id`` on both ``/api/ml/embeddings`` and
``/api/ml/gnn/inference``. Any unauthenticated pod in the cluster could
therefore name a victim tenant and spend the victim's LLM credentials.

So the header is now gated on the tenant's PROVENANCE
(``auth.tenant.tenant_source_var``, bound in lockstep with the tenant by
``AuthMiddleware``): only a tenant that arrived as a license token, or was
vouched for by an allowlisted service (``auth/vouch.py``,
``SCIENTIFIC_PROCESSOR_ONBEHALF_SUBJECTS``), is delegated.

OMIT, NOT REFUSE
----------------
An unvouched tenant does NOT fail the call — the header is withheld and a
``WS4_DELEGATION omitted`` warning is logged. Two reasons:

1. This runs inside an HTTP client shared by a REQUEST-scoped caller
   (``RagasService.evaluate``) and a FIRE-AND-FORGET background job
   (``SubgraphXService.explain`` spawns ``_run_explanation`` via
   ``asyncio.create_task``, which reaches ``gnn_inference`` twice). Raising here
   would surface in the background job as an unhandled task exception that
   strands the job, not as a clean 403 anybody can act on.
2. Every in-cluster caller today reaches this service on exactly the
   bare-``X-Tenant-Id`` trusted-network path, and
   ``SCIENTIFIC_PROCESSOR_ONBEHALF_SUBJECTS`` is UNSET until the gitops roll.
   Refusing would take those callers down on deploy.

Omitting is fail-safe, not fail-open: the gateway then resolves the SA's OWN
(scientific-processor platform) tenant for BYO + billing. The blast radius
becomes "scientific-processor pays and no customer BYO provider is selected" — a
loud, self-inflicted cost signal — instead of "a stranger's credentials are
spent".

BURN-DOWN SIGNAL — deliberately a LOG LINE, not a Prometheus counter. The sibling
ports (hirag ``hirag_effective_tenant_delegation_total``) increment a counter,
but scientific-processor exposes no ``/metrics`` endpoint at all (``telemetry.py``
wires OTLP *tracing* only, and no ServiceMonitor scrapes this service in
``gitops-production/clusters/strato/services/scientific-processor/``). A counter
here would be unscrapable decoration. ``WS4_DELEGATION omitted`` is a stable,
greppable token in Loki instead; wiring a real metrics endpoint is a separate
change.

Nothing is newly rejected. A bare ``X-Tenant-Id`` remains a fully accepted DATA
SCOPE on every route.

Ported from ``hirag_service`` #81 (``clients/ml_identity.delegation_headers``)
and ``kag`` #123 (``services/proxy.build_downstream_tenant_headers``).
"""

from __future__ import annotations

import httpx
from kag_common.attribution import attribution_event_hook
from kag_common.auth import outbound_headers

from auth.tenant import license_id_var, may_delegate, tenant_source_var
from config import get_settings
from log_config import get_logger

logger = get_logger(__name__)

EFFECTIVE_TENANT_HEADER = "X-Effective-Tenant-Id"


def delegation_headers(*, tenant_id: str, tenant_source: str) -> dict[str, str]:
    """The delegation header for the metered-gateway hop, or ``{}`` when unproven.

    Both parameters are keyword-only and have NO DEFAULT on purpose. Delegation
    is a privilege escalation — it converts a header into the authority to spend
    another tenant's LLM credentials — so a future call site must be forced to
    state where its tenant came from. A default would let the next caller
    silently re-open the hole; without one, omitting it is a ``TypeError`` at
    call time and a type-check error before that.

    This is the ONLY place in the service that may emit
    ``X-Effective-Tenant-Id``; ``tests/unit/test_ws4_vouched_delegation.py``
    pins that structurally.

    Returns:
        ``{"X-Effective-Tenant-Id": tenant_id}`` when the provenance is one this
        service can stand behind (``auth.tenant.VOUCHED_TENANT_SOURCES``), else
        ``{}``.
    """
    if not tenant_id:
        return {}
    if not may_delegate(tenant_source):
        logger.warning(
            "WS4_DELEGATION omitted: tenant=%s source=%s — a tenant may only be delegated to "
            "the metered ML gateway when it arrives as a license token or is vouched by an "
            "allowlisted service token (SCIENTIFIC_PROCESSOR_ONBEHALF_SUBJECTS). Falling back "
            "to scientific-processor's own service-account tenant for BYO + billing.",
            tenant_id,
            tenant_source,
        )
        return {}
    return {EFFECTIVE_TENANT_HEADER: tenant_id}


class PyTorchBackendClient:
    """Async HTTP client for pytorch-backend service."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        downstream_license_token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Modality-gateway migration (see config.downstream_license_token). When
        # set, embedding calls carry `X-Tenant-Id: Bearer <token>` so the
        # license-introspecting gateway accepts them. Empty (default) => raw
        # X-Tenant-Id => today's behaviour.
        self.downstream_license_token = downstream_license_token or None
        self._client: httpx.AsyncClient | None = None

    def _embedding_headers(self, tenant_id: str | None, tenant_source: str) -> dict[str, str]:
        """Build the outbound tenant headers for an ML-gateway call.

        Mirrors the kag `build_gateway_tenant_header` pattern:
        - token set  => `X-Tenant-Id: Bearer <token>` (gateway introspects it),
          PLUS — **only when `tenant_source` is a provenance this service can
          stand behind** — `X-Effective-Tenant-Id: <tenant_id>` so the gateway
          delegates BYO/cost resolution to the real customer tenant instead of
          svc-scientific-processor's own service tenant (requires
          svc-scientific-processor in the gateway's AUTH_DELEGATING_SUBJECTS).
          An unvouched tenant is omitted, not refused — see the module docstring;
        - tenant_id present (Bearer-form or bare) => X-Tenant-Id verbatim
          (today's behaviour: the two branches were byte-identical);
        - neither => no X-Tenant-Id.

        A bare X-Tenant-Id to the direct (non-gateway) backend is a DATA SCOPE,
        not an identity to spend as: no BYO credential is selected and no third
        party is billed, so provenance does not gate that path.

        ``tenant_source`` is required (no default) so a new caller cannot bind a
        tenant to this hop without saying where it came from.

        LICENCE. ``X-License-Id`` rides EVERY branch, gateway or not, because it
        is attribution data rather than authority — unlike
        ``X-Effective-Tenant-Id``, it grants nothing and selects no credential, so
        it is not gated on provenance. It is also NOT the same thing as
        ``downstream_license_token``: that is scientific-processor's OWN service
        licence, used to authenticate this hop, and putting it in ``X-License-Id``
        would book a customer's embedding spend against this service's licence.
        No licence in scope => no header.
        """
        license_id = license_id_var.get()
        if self.downstream_license_token:
            # `service_token` is the ladder rung that always Bearer-prefixes —
            # the opaque service licence must NOT go through the JWT-shape
            # heuristic on `tenant_token`, which would send it bare and 401.
            headers = outbound_headers(service_token=self.downstream_license_token, license_id=license_id)
            # Delegate to the real customer tenant when we have a plain UUID
            # (never a Bearer value, which would be a token, not a tenant) AND
            # the tenant's provenance permits being spent as.
            if tenant_id and not tenant_id.startswith("Bearer "):
                headers.update(delegation_headers(tenant_id=tenant_id, tenant_source=tenant_source))
            return headers
        return outbound_headers(tenant_id=tenant_id, license_id=license_id)

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            s = get_settings()
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                # Forward the caller's X-Project-Id / X-Job-Id (kag_common 0.33): read per
                # request from the bound context, so this cached client never leaks one
                # request's project onto another. Unbound (background work) adds nothing.
                event_hooks={"request": [attribution_event_hook]},
                transport=httpx.AsyncHTTPTransport(retries=s.http_transport_retries),
                timeout=httpx.Timeout(
                    connect=s.http_connect_timeout_s,
                    read=self.timeout,
                    write=s.http_write_timeout_s,
                    pool=s.http_pool_timeout_s,
                ),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def get_embeddings(
        self,
        texts: list[str],
        model: str = "default",
        normalize: bool = True,
        tenant_id: str | None = None,
    ) -> dict:
        """Generate embeddings for a list of texts.

        The tenant's PROVENANCE is read from the request-scoped
        ``tenant_source_var`` rather than passed in, exactly as the RLS tenant is:
        it is bound by ``AuthMiddleware`` and inherited by
        ``asyncio.create_task`` children, and its ``SOURCE_NONE`` default makes an
        unbound context (startup probe, import-time worker) non-delegatable —
        fail closed.
        """
        client = self._ensure_client()
        headers = self._embedding_headers(tenant_id, tenant_source_var.get())
        response = await client.post(
            "/api/ml/embeddings",
            json={"texts": texts, "model": model, "normalize": normalize},
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    async def gnn_inference(
        self,
        node_features: list[list[float]],
        edge_index: list[list[int]],
        model: str = "default",
        tenant_id: str | None = None,
    ) -> dict:
        """Run GNN inference on graph data."""
        client = self._ensure_client()
        # Route through the shared header builder so GNN calls carry the same
        # effective-tenant + Bearer license attribution as embedding calls
        # (was a bare X-Tenant-Id that dropped the customer tenant in gateway mode)
        # AND the same WS-4 provenance gate. This path is reached from the
        # SubgraphX fire-and-forget task, which inherits the ContextVar snapshot
        # taken at `asyncio.create_task` time inside the originating request.
        headers = self._embedding_headers(tenant_id, tenant_source_var.get())
        response = await client.post(
            "/api/ml/gnn/inference",
            json={
                "node_features": node_features,
                "edge_index": edge_index,
                "model": model,
            },
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> dict:
        """Check pytorch-backend health."""
        client = self._ensure_client()
        response = await client.get("/health")
        response.raise_for_status()
        return response.json()
