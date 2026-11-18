"""Request-scoped context vars.

``tenant_id_var`` carries the authenticated tenant for the lifetime of a
request so the database layer can bind the Postgres RLS GUC
(``app.current_tenant``) on every transaction without threading the tenant
through every call site. Set by the auth middleware from
``request.state.tenant_id`` (never a client-asserted body value); read by
``register_tenant_guc`` in the engine ``begin`` hook.
"""

from __future__ import annotations

from contextvars import ContextVar

tenant_id_var: ContextVar[str | None] = ContextVar("tenant_id", default=None)
"""The current request's tenant id, or ``None`` outside a request."""
