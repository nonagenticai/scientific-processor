"""
Configuration helpers for the scientific processor service.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import ClassVar

#: ``AUTH_TOKENLESS_IDENTITY_MODE`` (estate name: procedure_management #116,
#: requirements_management, tool_registry). Governs ONLY a caller that asserts a
#: tenant without a licence binding it, plus the ``Authorization: Bearer
#: <licence>`` carrier:
#:   observe (default) — served exactly as before; ``Authorization`` is never
#:                       introspected as a licence (only its SHAPE is counted).
#:   enforce           — a trusted-network plain ``X-Tenant-Id`` with no licence
#:                       is a 401, and the ``Authorization`` licence carrier
#:                       switches on (see ``auth/middleware.py``).
#: Any other value FAILS STARTUP: a typo must not silently leave the hole open.
TOKENLESS_IDENTITY_MODE_ENV = "AUTH_TOKENLESS_IDENTITY_MODE"
TOKENLESS_IDENTITY_MODE_OBSERVE = "observe"
TOKENLESS_IDENTITY_MODE_ENFORCE = "enforce"
TOKENLESS_IDENTITY_MODES: frozenset[str] = frozenset({TOKENLESS_IDENTITY_MODE_OBSERVE, TOKENLESS_IDENTITY_MODE_ENFORCE})

#: Boolean auth-gate flags (``ENABLE_AUTH``, ``ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT``).
#: Same rule as the mode above: an unrecognised spelling FAILS STARTUP. These
#: used to fall back silently — ``ENABLE_AUTH=1`` disabled auth in
#: ``auth.dependencies`` (``== "true"``) while ``Settings`` read it as enabled,
#: and ``ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT=flase`` left the default-tenant
#: fallback OPEN. Unset or blank keeps the default.
ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT_ENV = "ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT"
_TRUE_FLAGS: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_FALSE_FLAGS: frozenset[str] = frozenset({"0", "false", "no", "off"})


def parse_auth_flag(name: str, raw: str | None, default: bool) -> bool:
    """Strictly parse a boolean auth-gate flag; unset/blank = ``default``.

    Raises ``ValueError`` (naming the env var) on any other spelling, so a typo
    refuses boot instead of silently picking a side.
    """
    value = (raw or "").strip().lower()
    if not value:
        return default
    if value in _TRUE_FLAGS:
        return True
    if value in _FALSE_FLAGS:
        return False
    raise ValueError(
        f"{name}={raw!r} is not a recognised boolean; use one of "
        f"{sorted(_TRUE_FLAGS | _FALSE_FLAGS)} (refusing to start rather than guess)"
    )


def parse_tokenless_identity_mode(raw: str | None) -> str:
    """Normalise ``AUTH_TOKENLESS_IDENTITY_MODE``; unset/blank = observe.

    Raises:
        ValueError: on any value other than observe / enforce (case and
            surrounding whitespace ignored).
    """
    value = (raw or "").strip().lower() or TOKENLESS_IDENTITY_MODE_OBSERVE
    if value not in TOKENLESS_IDENTITY_MODES:
        raise ValueError(
            f"{TOKENLESS_IDENTITY_MODE_ENV}={raw!r} is not one of {sorted(TOKENLESS_IDENTITY_MODES)}; "
            "refusing to start rather than guess"
        )
    return value


def _get_env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass
class Settings:
    # Database (optional — if unset, falls back to in-memory job store)
    database_url: str | None = field(default_factory=lambda: os.getenv("DATABASE_URL"))

    # Core
    version: str = field(default_factory=lambda: _get_env("SCIENTIFIC_PROCESSOR_VERSION", "0.1.0"))
    service_name: str = field(
        default_factory=lambda: _get_env("SCIENTIFIC_PROCESSOR_SERVICE_NAME", "scientific-processor")
    )
    environment: str = field(default_factory=lambda: _get_env("ENVIRONMENT", "production"))
    enable_docs: bool = field(default_factory=lambda: _get_bool("ENABLE_DOCS", False))
    # ⚠️ MUST stay in step with ``auth.tenant.DEFAULT_TENANT_ID`` — same env var,
    # same fallback uuid. Not imported from there because this field is read
    # through an ``lru_cache``d ``get_settings()``, so folding the two together
    # would change WHEN the env is frozen for every consumer of Settings.
    default_tenant_id: str = field(
        default_factory=lambda: _get_env("DEFAULT_TENANT_ID", "0195f48b-d000-7000-8000-000000000001")
    )
    enable_auth: bool = field(default_factory=lambda: parse_auth_flag("ENABLE_AUTH", os.getenv("ENABLE_AUTH"), True))
    # Validated at boot only; ``auth.middleware`` re-reads it per request so a
    # gitops flip needs no re-import.
    allow_trusted_network_default_tenant: bool = field(
        default_factory=lambda: parse_auth_flag(
            ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT_ENV, os.getenv(ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT_ENV), True
        )
    )
    # Raw value; validated in ``__post_init__`` in EVERY environment (boot
    # refusal). The middleware re-reads the env at call time, like its other
    # switches, so this field exists to make an unknown value fail the boot.
    auth_tokenless_identity_mode: str = field(
        default_factory=lambda: os.getenv(TOKENLESS_IDENTITY_MODE_ENV, TOKENLESS_IDENTITY_MODE_OBSERVE)
    )

    # Server
    docs_url: str | None = field(default_factory=lambda: os.getenv("SCIENTIFIC_PROCESSOR_DOCS_URL", "/docs"))
    redoc_url: str | None = field(default_factory=lambda: os.getenv("SCIENTIFIC_PROCESSOR_REDOC_URL", "/redoc"))
    openapi_url: str | None = field(
        default_factory=lambda: os.getenv("SCIENTIFIC_PROCESSOR_OPENAPI_URL", "/openapi.json")
    )
    bind_host: str = field(
        default_factory=lambda: _get_env("SCIENTIFIC_PROCESSOR_BIND_HOST", "0.0.0.0")  # nosec B104
    )
    bind_port: int = field(default_factory=lambda: _get_int("SCIENTIFIC_PROCESSOR_BIND_PORT", 8808))
    reload: bool = field(default_factory=lambda: _get_bool("SCIENTIFIC_PROCESSOR_RELOAD", False))

    # Upstream service URLs.
    # Empty-string placeholders — real values come from the
    # scientific-processor Deployment env block in gitops-production. The
    # `__post_init__` validator below fails fast if any are missing in
    # production. Per gitops-production/docs/config-externalization-policy.md.
    graph_adapter_url: str = field(default_factory=lambda: _get_env("SCIENTIFIC_PROCESSOR_GRAPH_ADAPTER_URL", ""))
    # WS-4 on-behalf-of: this service's own identity token, presented to
    # graph-adapter as `Authorization: Bearer <token>` so its asserted plain
    # X-Tenant-Id is trusted (subject allowlisted in GRAPH_ADAPTER_ONBEHALF_SUBJECTS).
    # Empty (default) => no Authorization header => today's behaviour.
    graph_service_token: str = field(default_factory=lambda: _get_env("SCIENTIFIC_PROCESSOR_GRAPH_SERVICE_TOKEN", ""))
    pytorch_backend_url: str = field(default_factory=lambda: _get_env("SCIENTIFIC_PROCESSOR_PYTORCH_BACKEND_URL", ""))
    # This service's OWN service licence token. Two consumers:
    # - ML-backend embedding calls via the license-introspecting modality-gateway
    #   (`X-Tenant-Id: Bearer <token>`; empty => raw X-Tenant-Id).
    # - lean4-prover (GPU backend): every call must carry a licence. The end
    #   tenant's own token is forwarded when the request had one; this token is
    #   the fallback. REQUIRED in production (boot fails without it).
    downstream_license_token: str = field(
        default_factory=lambda: _get_env("SCIENTIFIC_PROCESSOR_DOWNSTREAM_LICENSE_TOKEN", "")
    )
    inference_router_url: str = field(default_factory=lambda: _get_env("SCIENTIFIC_PROCESSOR_INFERENCE_ROUTER_URL", ""))
    discovery_engine_url: str = field(default_factory=lambda: _get_env("SCIENTIFIC_PROCESSOR_DISCOVERY_ENGINE_URL", ""))
    lean4_prover_url: str = field(default_factory=lambda: _get_env("SCIENTIFIC_PROCESSOR_LEAN4_PROVER_URL", ""))
    # GPU admission control (lean4-prover#15): the server's `prove` lane admits
    # 2 active + 8 waiting and answers 503 gpu_busy + Retry-After beyond that.
    # Cap this pod's in-flight lean4 calls, and bound how long one call keeps
    # retrying a busy server before it surfaces 503 + Retry-After upward.
    lean4_prover_max_concurrency: int = field(
        default_factory=lambda: _get_int("SCIENTIFIC_PROCESSOR_LEAN4_PROVER_MAX_CONCURRENCY", 4)
    )
    lean4_prover_busy_deadline_s: float = field(
        default_factory=lambda: _get_float("SCIENTIFIC_PROCESSOR_LEAN4_PROVER_BUSY_DEADLINE_S", 120.0)
    )

    # Sandbox image (private registry path; environment-specific).
    sandbox_image: str = field(default_factory=lambda: _get_env("SCIENTIFIC_PROCESSOR_SANDBOX_IMAGE", ""))

    # OTLP traces exporter endpoint (cluster observability).
    otlp_traces_endpoint: str = field(default_factory=lambda: _get_env("OTLP_TRACES_ENDPOINT", ""))

    # MinIO (artifact storage)
    minio_endpoint: str = field(default_factory=lambda: _get_env("SCIENTIFIC_PROCESSOR_MINIO_ENDPOINT", ""))
    minio_access_key: str = field(default_factory=lambda: _get_env("SCIENTIFIC_PROCESSOR_MINIO_ACCESS_KEY", ""))
    minio_secret_key: str | None = field(default_factory=lambda: os.getenv("SCIENTIFIC_PROCESSOR_MINIO_SECRET_KEY"))
    minio_bucket: str = field(
        default_factory=lambda: _get_env("SCIENTIFIC_PROCESSOR_MINIO_BUCKET", "scientific-processor")
    )

    # httpx transport hardening for upstream clients (inference-router,
    # graph-adapter, pytorch-backend, discovery-engine, lean4-prover). A
    # bare AsyncClient with no retries surfaced transient ambient-mesh
    # handshake blips as terminal "Server disconnected" failures during
    # the 2026-05-16 trust.pdf ingest. Mirror the worker_management F3
    # pattern (PR #46): AsyncHTTPTransport(retries=N) covers idempotent
    # connect-level failures; explicit Timeout covers cold-cache latency.
    http_connect_timeout_s: float = field(
        default_factory=lambda: _get_float("SCIENTIFIC_PROCESSOR_HTTP_CONNECT_TIMEOUT_S", 10.0)
    )
    http_write_timeout_s: float = field(
        default_factory=lambda: _get_float("SCIENTIFIC_PROCESSOR_HTTP_WRITE_TIMEOUT_S", 10.0)
    )
    http_pool_timeout_s: float = field(
        default_factory=lambda: _get_float("SCIENTIFIC_PROCESSOR_HTTP_POOL_TIMEOUT_S", 5.0)
    )
    http_transport_retries: int = field(
        default_factory=lambda: _get_int("SCIENTIFIC_PROCESSOR_HTTP_TRANSPORT_RETRIES", 3)
    )

    # Request limits
    request_timeout: float = field(default_factory=lambda: _get_float("SCIENTIFIC_PROCESSOR_REQUEST_TIMEOUT", 300.0))
    subgraph_max_hops: int = field(default_factory=lambda: _get_int("SCIENTIFIC_PROCESSOR_SUBGRAPH_MAX_HOPS", 3))
    subgraph_max_nodes: int = field(default_factory=lambda: _get_int("SCIENTIFIC_PROCESSOR_SUBGRAPH_MAX_NODES", 5000))

    # Sandbox limits
    sandbox_timeout: int = field(default_factory=lambda: _get_int("SCIENTIFIC_PROCESSOR_SANDBOX_TIMEOUT", 120))
    sandbox_memory_limit: str = field(
        default_factory=lambda: _get_env("SCIENTIFIC_PROCESSOR_SANDBOX_MEMORY_LIMIT", "512Mi")
    )
    sandbox_cpu_limit: str = field(default_factory=lambda: _get_env("SCIENTIFIC_PROCESSOR_SANDBOX_CPU_LIMIT", "500m"))

    # Pipeline settings
    pipeline_auto_cove_threshold: float = field(
        default_factory=lambda: _get_float("SCIENTIFIC_PROCESSOR_PIPELINE_AUTO_COVE_THRESHOLD", 0.6)
    )
    pipeline_auto_formal_threshold: float = field(
        default_factory=lambda: _get_float("SCIENTIFIC_PROCESSOR_PIPELINE_AUTO_FORMAL_THRESHOLD", 0.3)
    )
    pipeline_batch_size: int = field(default_factory=lambda: _get_int("SCIENTIFIC_PROCESSOR_PIPELINE_BATCH_SIZE", 50))
    pipeline_concurrency: int = field(default_factory=lambda: _get_int("SCIENTIFIC_PROCESSOR_PIPELINE_CONCURRENCY", 10))

    # Cache settings
    cache_ttl_seconds: int = field(default_factory=lambda: _get_int("SCIENTIFIC_PROCESSOR_CACHE_TTL_SECONDS", 300))
    cache_max_entries: int = field(default_factory=lambda: _get_int("SCIENTIFIC_PROCESSOR_CACHE_MAX_ENTRIES", 1000))

    # Sandbox toggle (disable for local dev without K8s)
    sandbox_enabled: bool = field(default_factory=lambda: _get_bool("SCIENTIFIC_PROCESSOR_SANDBOX_ENABLED", False))
    # Explicit opt-in to run submitted code IN-PROCESS when the K8s sandbox is
    # off. Local execution runs caller code in the service container with all
    # its secrets — never enable outside local dev. Absent this flag, the local
    # path refuses rather than silently downgrading to arbitrary execution.
    sandbox_allow_local: bool = field(
        default_factory=lambda: _get_bool("SCIENTIFIC_PROCESSOR_SANDBOX_ALLOW_LOCAL", False)
    )

    # Integrity score weights
    weight_citation: float = field(default_factory=lambda: _get_float("SCIENTIFIC_PROCESSOR_WEIGHT_CITATION", 0.25))
    weight_crossref: float = field(default_factory=lambda: _get_float("SCIENTIFIC_PROCESSOR_WEIGHT_CROSSREF", 0.20))
    weight_community: float = field(default_factory=lambda: _get_float("SCIENTIFIC_PROCESSOR_WEIGHT_COMMUNITY", 0.15))
    weight_authority: float = field(default_factory=lambda: _get_float("SCIENTIFIC_PROCESSOR_WEIGHT_AUTHORITY", 0.25))
    weight_contradiction: float = field(
        default_factory=lambda: _get_float("SCIENTIFIC_PROCESSOR_WEIGHT_CONTRADICTION", 0.15)
    )

    # Fields whose value MUST be supplied via env in production. The
    # `__post_init__` hook below raises if any are empty when
    # ENVIRONMENT=production. Source-of-truth for the env var names lives in
    # the scientific-processor Deployment manifest in gitops-production.
    _REQUIRED_IN_PRODUCTION: ClassVar[tuple[str, ...]] = (
        "graph_adapter_url",
        "pytorch_backend_url",
        "inference_router_url",
        "discovery_engine_url",
        "lean4_prover_url",
        # lean4-prover rejects tokenless calls; without a service licence every
        # trusted-network formal request would 401 at runtime. Fail at boot.
        "downstream_license_token",
        "minio_endpoint",
        "minio_access_key",
        "otlp_traces_endpoint",
        # sandbox_image is only required when sandbox_enabled; checked below.
    )

    def __post_init__(self) -> None:
        """Fail fast on an unknown identity mode, or missing production-only fields."""
        # Every environment, before the production-only early return: an
        # unknown spelling must never boot as a silent ``observe``.
        self.auth_tokenless_identity_mode = parse_tokenless_identity_mode(self.auth_tokenless_identity_mode)
        if self.environment != "production":
            return
        missing = [n for n in self._REQUIRED_IN_PRODUCTION if not getattr(self, n)]
        if self.sandbox_enabled and not self.sandbox_image:
            missing.append("sandbox_image")
        if missing:
            raise RuntimeError(
                f"scientific-processor settings missing required fields in "
                f"production: {missing}. Set them via env vars on the "
                f"Deployment (see gitops-production)."
            )


@lru_cache
def get_settings() -> Settings:
    """Cached accessor for scientific processor settings."""
    return Settings()
