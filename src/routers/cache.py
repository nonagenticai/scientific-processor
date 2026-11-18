"""Cache management endpoints — stats, invalidation, and clear."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from auth.dependencies import resolve_tenant_id
from dependencies import get_cache_service
from services.cache import CacheService

router = APIRouter()


class CacheStatsResponse(BaseModel):
    size: int
    max_entries: int
    ttl_seconds: float
    hits: int
    misses: int
    hit_rate: float


class CacheInvalidateRequest(BaseModel):
    key: str


class CacheClearResponse(BaseModel):
    cleared: bool
    previous_size: int


def _assert_tenant_owns_key(key: str, tenant_id: str) -> None:
    """Reject a cache key that belongs to another tenant.

    Ownership is decided by ``CacheService.key_tenant`` — the single definition of
    the ``<namespace>:<tenant_id>:<rest>`` grammar, kept beside the ``make_*_key``
    builders that produce it. A key with no tenant segment belongs to nobody and is
    refused (fail closed).

    403, not 404: the check reads the caller-supplied string ONLY and never touches
    the cache, so it discloses nothing about whether the entry exists. A 404 would
    assert non-existence — a claim this check cannot make without probing the shared
    cache, which would itself be the cross-tenant disclosure we are closing.
    """
    if CacheService.key_tenant(key) != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cache key is not scoped to the calling tenant",
        )


@router.get("/stats", response_model=CacheStatsResponse, operation_id="cache_stats")
async def cache_stats(
    cache: CacheService = Depends(get_cache_service),
) -> CacheStatsResponse:
    """Return cache hit/miss statistics."""
    stats = cache.stats()
    return CacheStatsResponse(**stats)


@router.post("/invalidate", operation_id="cache_invalidate")
async def cache_invalidate(
    body: CacheInvalidateRequest,
    request: Request,
    cache: CacheService = Depends(get_cache_service),
) -> dict[str, str]:
    """Invalidate a specific cache entry owned by the calling tenant."""
    _assert_tenant_owns_key(body.key, resolve_tenant_id(request))
    cache.invalidate(body.key)
    return {"status": "invalidated", "key": body.key}


# Why /clear is tenant-scoped rather than admin-gated:
#
# ``CacheService`` is a process-global singleton, so a blanket clear would wipe every
# other tenant's entries on behalf of one caller. The obvious guard — an admin realm
# role — is UNREACHABLE here: ``AuthMiddleware`` hard-codes ``realm_access={"roles": []}``
# on the in-cluster path (roles must never be derived from client headers, since every
# pod sits inside the trusted CIDR), and the licenses introspection payload carries no
# ``roles`` field at all. A role gate therefore 403s every real caller, permanently and
# silently. Scoping the clear by the same ``CacheService.key_tenant`` grammar the
# ownership check already uses closes the cross-tenant wipe AND keeps the endpoint
# usable.
#
# ``CacheClearResponse`` keeps its exact shape (an orval client baseline depends on it);
# ``previous_size`` now means "this tenant's entries removed", not the global size.
@router.post("/clear", response_model=CacheClearResponse, operation_id="cache_clear")
async def cache_clear(
    request: Request,
    cache: CacheService = Depends(get_cache_service),
) -> CacheClearResponse:
    """Clear the calling tenant's cached entries.

    Scoped to the caller: other tenants' entries, and keys with no tenant segment,
    are left untouched. `previous_size` reports how many of the caller's entries
    were removed.
    """
    removed = cache.clear_tenant(resolve_tenant_id(request))
    return CacheClearResponse(cleared=True, previous_size=removed)
