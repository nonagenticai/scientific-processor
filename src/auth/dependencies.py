"""FastAPI dependencies for authentication."""

import os
import uuid

from fastapi import HTTPException, Request, status

from config import parse_auth_flag

from .models import User
from .tenant import DEFAULT_TENANT_ID

# Same strict parser as ``config.Settings.enable_auth``: the two used to disagree
# (``ENABLE_AUTH=1`` was auth-ON in Settings but anonymous here). An unknown
# spelling raises at import, i.e. the process refuses to start.
ENABLE_AUTH = parse_auth_flag("ENABLE_AUTH", os.getenv("ENABLE_AUTH"), True)

# ``DEFAULT_TENANT_ID`` is imported, not re-read from the env: this module and
# ``auth.middleware`` used to call ``os.getenv`` independently with the fallback
# uuid spelled out twice. If the two ever disagreed, a header-less caller would be
# SERVED as one tenant by the middleware and READ BACK as another by
# ``resolve_tenant_id`` — splitting its cache namespace and RLS scope in half.
# ``auth.tenant`` is the single definition; see the comment there.

_ANONYMOUS_USER = User(
    sub=DEFAULT_TENANT_ID,
    preferred_username="anonymous",
    tenant_id=DEFAULT_TENANT_ID,
)


def get_current_user(request: Request) -> User:
    """Get authenticated user from request."""
    if not ENABLE_AUTH:
        return _ANONYMOUS_USER
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    user: User = request.state.user
    return user


def get_current_user_id(request: Request) -> str:
    """Get current user's ID."""
    user = get_current_user(request)
    return user.sub


def require_role(role: str):
    """Require specific role."""

    def checker(request: Request) -> User:
        user = get_current_user(request)
        if not user.has_realm_role(role):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Role '{role}' required")
        return user

    return checker


def require_any_role(*roles: str):
    """Require any of specified roles."""

    def checker(request: Request) -> User:
        user = get_current_user(request)
        if not any(user.has_realm_role(r) for r in roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of roles required: {', '.join(roles)}",
            )
        return user

    return checker


def require_all_roles(*roles: str):
    """Require all specified roles."""

    def checker(request: Request) -> User:
        user = get_current_user(request)
        missing = [r for r in roles if not user.has_realm_role(r)]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required roles: {', '.join(missing)}",
            )
        return user

    return checker


def _require_uuid_tenant(tenant_id: str) -> str:
    """Return ``tenant_id`` as a ``str``, or raise 400 if it is not a uuid.

    WHY THIS EXISTS. ``jobs.tenant_id`` and ``verification_history.tenant_id`` are
    real Postgres ``uuid`` columns (migration ``005``). ``AuthMiddleware`` admits
    anything matching ``_TENANT_ID_RE`` = ``[A-Za-z0-9._-]{1,128}``, which happily
    passes legacy slugs like ``n8n-dev`` and ``anonymous``. Before the type flip
    those merely matched no rows (0 rows / 404). After it, Postgres raises
    ``InvalidTextRepresentation: invalid input syntax for type uuid`` and the
    caller gets a **500** — malformed client input turned into a server error.
    So the check lives here, at the single choke point every router reads the
    tenant through, not at ~40 individual call sites.

    400, not 404: the request is malformed, it is not a missing resource. 400 is
    also what ``middleware._bad_request("INVALID_TENANT_ID", …)`` already returns
    for a header that fails the coarser regex, so this is the same answer for the
    same class of defect, just a stricter grammar.

    ⚠️ Returns a ``str``, never a ``uuid.UUID``. The value flows into
    ``set_config('app.current_tenant', <value>, true)``, whose signature is
    ``(text, text, bool)`` — there is no uuid overload, so a ``uuid.UUID`` here
    dies with ``UndefinedFunction`` and takes down every transaction under RLS.
    It also flows verbatim into the cache-key and MinIO-prefix grammars, which
    are string builders.

    ⚠️ Returns the CANONICAL spelling — ``str(uuid.UUID(x))`` — not the original
    string. One tenant has unlimited valid spellings, because ``uuid.UUID()``
    accepts dashes wherever they fall (and braces, and a ``urn:uuid:`` prefix)::

        019efaa0-512b-733e-9e98-b1ac82a6f7ee     canonical
        019efaa0-512b-733e-9e98-b1ac82a6f7ee-    same uuid, different string
        019efaa0-512b733e9e98b1ac82a6f7ee        same uuid, different string

    Postgres does not care: every bind and every RLS predicate casts to ``uuid``,
    so all three are one tenant there. But wherever the validated string is used
    as a KEY rather than cast — the ``CacheService`` key grammar is this repo's
    exposure, MinIO prefixes the same shape — the spellings partition. A caller
    that sent the dashed variant would get its own cache namespace and could not
    see entries it had just written under the canonical one. Canonicalising here
    collapses every spelling onto one key and removes the whole class, instead of
    leaving each keyed consumer to guard against it.

    This is still exactly what the pod-to-pod ``X-Tenant-Id: <uuid>`` path wants:
    that path already sends canonical uuids, for which this is the identity.
    """
    try:
        canonical = uuid.UUID(tenant_id)
    except (ValueError, AttributeError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"tenant id must be a uuid (got {tenant_id!r})",
        ) from exc
    return str(canonical)


def resolve_tenant_id(request: Request) -> str:
    """Authoritative per-request tenant, for inline use inside routers.

    The tenant is set ONLY by :class:`~auth.middleware.AuthMiddleware`
    (introspected-license payload, or a raw pod-to-pod ``X-Tenant-Id: <uuid>``
    on the trusted-network path). Routers must read it from here rather than
    re-reading the ``X-Tenant-Id`` header themselves — that header is now
    ``Bearer <token>`` on the introspection path and would poison tenant
    resolution. Falls back to ``DEFAULT_TENANT_ID`` only when auth is disabled
    (middleware absent); an authed protected request always has it set, and an
    unauthenticated one is rejected upstream before reaching a router.
    """
    return _require_uuid_tenant(str(getattr(request.state, "tenant_id", None) or DEFAULT_TENANT_ID))


async def get_tenant_id(request: Request) -> str:
    """Get tenant id from request state, falling back to DEFAULT_TENANT_ID."""
    return resolve_tenant_id(request)


async def get_subject_id(request: Request) -> str:
    """Get subject (user within tenant) from request state.

    Falls back to empty string — never raises. Empty string is treated as
    "tenant-only caller" downstream.
    """
    return str(getattr(request.state, "subject_id", "") or "")


async def get_tenant_name(request: Request) -> str:
    """Get tenant display name from request state. Falls back to empty string."""
    return str(getattr(request.state, "tenant_name", "") or "")
