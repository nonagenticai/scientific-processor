"""FastAPI dependency injection factories for clients and services."""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from clients.discovery_engine import DiscoveryEngineClient
from clients.graph_adapter import GraphAdapterClient
from clients.inference_router import InferenceRouterClient
from clients.lean4_prover import Lean4ProverClient
from clients.minio_client import MinIOClient
from clients.pytorch_backend import PyTorchBackendClient
from config import Settings, get_settings
from services.cache import CacheService
from services.centrality_service import CentralityService
from services.community_service import CommunityService
from services.contradiction_service import ContradictionService
from services.cove_service import CoVeService
from services.formal_service import FormalService
from services.integrity_service import IntegrityService
from services.job_manager import JobManager
from services.ragas_service import RagasService
from services.sandbox_service import SandboxService
from services.stats_service import StatsService
from services.subgraph_service import SubgraphService
from services.subgraphx_service import SubgraphXService
from services.sympy_verifier import SymPyVerifier
from services.z3_verifier import Z3Verifier

# Singletons — persist across requests for connection reuse
_job_manager: JobManager | None = None
_minio_client: MinIOClient | None = None
_graph_client: GraphAdapterClient | None = None
_pytorch_client: PyTorchBackendClient | None = None
_inference_client: InferenceRouterClient | None = None
_discovery_client: DiscoveryEngineClient | None = None
_lean4_client: Lean4ProverClient | None = None
_cache_service: CacheService | None = None

# Database engine + session factory (set during lifespan if DATABASE_URL is configured)
_db_engine: AsyncEngine | None = None
_db_session_factory: async_sessionmaker[AsyncSession] | None = None


def set_db_session_factory(engine: AsyncEngine, factory: async_sessionmaker[AsyncSession]) -> None:
    """Called from app lifespan to wire up the database layer."""
    global _db_engine, _db_session_factory
    _db_engine = engine
    _db_session_factory = factory
    # HIPAA G4: bind app.current_tenant on every transaction so Postgres RLS
    # scopes to the request tenant. No-op unless TENANT_RLS_MODE=enforce;
    # fail-closed to 0 rows when no tenant is bound. Single chokepoint → covers
    # every session produced by this factory by construction.
    from kag_common.database import register_tenant_guc

    from context import tenant_id_var

    register_tenant_guc(engine, lambda: tenant_id_var.get(None))


async def dispose_db_engine() -> None:
    """Called from app lifespan shutdown to close the database engine."""
    global _db_engine, _db_session_factory
    if _db_engine is not None:
        await _db_engine.dispose()
    _db_engine = None
    _db_session_factory = None


def get_graph_client(settings: Settings = Depends(get_settings)) -> GraphAdapterClient:
    global _graph_client
    if _graph_client is None:
        _graph_client = GraphAdapterClient(
            base_url=settings.graph_adapter_url,
            timeout=settings.request_timeout,
            service_token=settings.graph_service_token,
        )
    return _graph_client


def get_pytorch_client(settings: Settings = Depends(get_settings)) -> PyTorchBackendClient:
    global _pytorch_client
    if _pytorch_client is None:
        _pytorch_client = PyTorchBackendClient(
            base_url=settings.pytorch_backend_url,
            timeout=settings.request_timeout,
            downstream_license_token=settings.downstream_license_token,
        )
    return _pytorch_client


def get_inference_client(settings: Settings = Depends(get_settings)) -> InferenceRouterClient:
    global _inference_client
    if _inference_client is None:
        _inference_client = InferenceRouterClient(
            base_url=settings.inference_router_url, timeout=settings.request_timeout
        )
    return _inference_client


def get_discovery_client(settings: Settings = Depends(get_settings)) -> DiscoveryEngineClient:
    global _discovery_client
    if _discovery_client is None:
        _discovery_client = DiscoveryEngineClient(
            base_url=settings.discovery_engine_url, timeout=settings.request_timeout
        )
    return _discovery_client


def get_lean4_client(settings: Settings = Depends(get_settings)) -> Lean4ProverClient:
    global _lean4_client
    if _lean4_client is None:
        _lean4_client = Lean4ProverClient(
            base_url=settings.lean4_prover_url,
            timeout=settings.request_timeout,
            service_license_token=settings.downstream_license_token,
            max_concurrency=settings.lean4_prover_max_concurrency,
            busy_deadline_s=settings.lean4_prover_busy_deadline_s,
        )
    return _lean4_client


def get_cache_service(settings: Settings = Depends(get_settings)) -> CacheService:
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService(ttl_seconds=settings.cache_ttl_seconds, max_entries=settings.cache_max_entries)
    return _cache_service


def clear_cache() -> None:
    """Clear and reset the cache singleton. Call during app shutdown."""
    global _cache_service
    if _cache_service is not None:
        _cache_service.clear()
    _cache_service = None


async def close_clients() -> None:
    """Close all persistent HTTP clients. Call during app shutdown."""
    global _graph_client, _pytorch_client, _inference_client, _discovery_client, _lean4_client
    global _minio_client
    for client in (_graph_client, _pytorch_client, _inference_client, _discovery_client, _lean4_client):
        if client is not None:
            await client.close()
    _graph_client = None
    _pytorch_client = None
    _inference_client = None
    _discovery_client = None
    _lean4_client = None
    # MinIOClient now holds one long-lived S3 client (and its connection pool)
    # instead of opening one per operation, so shutdown has to release it.
    if _minio_client is not None:
        await _minio_client.aclose()
        _minio_client = None


def get_subgraph_service(
    graph_client: GraphAdapterClient = Depends(get_graph_client),
    settings: Settings = Depends(get_settings),
    cache: CacheService = Depends(get_cache_service),
) -> SubgraphService:
    return SubgraphService(graph_client=graph_client, settings=settings, cache=cache)


def get_contradiction_service(
    subgraph_service: SubgraphService = Depends(get_subgraph_service),
) -> ContradictionService:
    return ContradictionService(subgraph_service=subgraph_service)


def get_centrality_service(
    subgraph_service: SubgraphService = Depends(get_subgraph_service),
    graph_client: GraphAdapterClient = Depends(get_graph_client),
    cache: CacheService = Depends(get_cache_service),
) -> CentralityService:
    return CentralityService(subgraph_service=subgraph_service, graph_client=graph_client, cache=cache)


def get_community_service(
    graph_client: GraphAdapterClient = Depends(get_graph_client),
    cache: CacheService = Depends(get_cache_service),
) -> CommunityService:
    return CommunityService(graph_client=graph_client, cache=cache)


def get_stats_service() -> StatsService:
    return StatsService()


def get_integrity_service(
    subgraph_service: SubgraphService = Depends(get_subgraph_service),
    contradiction_service: ContradictionService = Depends(get_contradiction_service),
    centrality_service: CentralityService = Depends(get_centrality_service),
    community_service: CommunityService = Depends(get_community_service),
    graph_client: GraphAdapterClient = Depends(get_graph_client),
    settings: Settings = Depends(get_settings),
) -> IntegrityService:
    return IntegrityService(
        subgraph_service=subgraph_service,
        contradiction_service=contradiction_service,
        centrality_service=centrality_service,
        community_service=community_service,
        settings=settings,
        graph_client=graph_client,
    )


def get_cove_service(
    inference_client: InferenceRouterClient = Depends(get_inference_client),
    graph_client: GraphAdapterClient = Depends(get_graph_client),
    discovery_client: DiscoveryEngineClient = Depends(get_discovery_client),
) -> CoVeService:
    return CoVeService(inference_client=inference_client, graph_client=graph_client, discovery_client=discovery_client)


def get_ragas_service(
    inference_client: InferenceRouterClient = Depends(get_inference_client),
    pytorch_client: PyTorchBackendClient = Depends(get_pytorch_client),
) -> RagasService:
    return RagasService(inference_client=inference_client, embedding_client=pytorch_client)


def get_job_manager() -> JobManager:
    """Return a persistent job manager if DATABASE_URL is set, else in-memory."""
    global _job_manager
    if _job_manager is None:
        if _db_session_factory is not None:
            from services.persistent_job_manager import PersistentJobManager

            _job_manager = PersistentJobManager(session_factory=_db_session_factory)
        else:
            _job_manager = JobManager()
    assert _job_manager is not None
    return _job_manager


def get_minio_client(settings: Settings = Depends(get_settings)) -> MinIOClient | None:
    global _minio_client
    if _minio_client is None and settings.minio_secret_key:
        _minio_client = MinIOClient(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
        )
    return _minio_client


def get_subgraphx_service(
    subgraph_service: SubgraphService = Depends(get_subgraph_service),
    pytorch_client: PyTorchBackendClient = Depends(get_pytorch_client),
    job_manager: JobManager = Depends(get_job_manager),
    minio_client: MinIOClient | None = Depends(get_minio_client),
) -> SubgraphXService:
    return SubgraphXService(
        subgraph_service=subgraph_service,
        pytorch_client=pytorch_client,
        job_manager=job_manager,
        minio_client=minio_client,
    )


def get_sandbox_service(
    job_manager: JobManager = Depends(get_job_manager),
    minio_client: MinIOClient | None = Depends(get_minio_client),
    settings: Settings = Depends(get_settings),
) -> SandboxService:
    return SandboxService(job_manager=job_manager, minio_client=minio_client, settings=settings)


def get_z3_verifier() -> Z3Verifier:
    return Z3Verifier()


def get_sympy_verifier() -> SymPyVerifier:
    return SymPyVerifier()


def get_formal_service(
    inference_client: InferenceRouterClient = Depends(get_inference_client),
    z3_verifier: Z3Verifier = Depends(get_z3_verifier),
    sympy_verifier: SymPyVerifier = Depends(get_sympy_verifier),
    minio_client: MinIOClient | None = Depends(get_minio_client),
    lean4_client: Lean4ProverClient = Depends(get_lean4_client),
) -> FormalService:
    return FormalService(
        inference_client=inference_client,
        z3_verifier=z3_verifier,
        sympy_verifier=sympy_verifier,
        minio_client=minio_client,
        lean4_client=lean4_client,
    )
