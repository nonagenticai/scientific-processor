# OPERATION_ID_PLAN

Canonical operation IDs for every route in scientific_processor.
Format: `<tag>_<verb>` snake_case, globally unique within the service.

## Why this matters

The `kag` gateway merges this service's `/openapi.json` under the `/scientific`
prefix (`kag/src/services/openapi_merge.py`) and prefixes each `operationId`
with `scientific_`. The `n8n-nodes-kag` generator then camelCases it into the
node operation name. So `cache_clear` → `scientific_cache_clear` →
`scientificCacheClear`. Without an explicit `operation_id`, FastAPI auto-derives
`cache_clear_api_cache_clear_post` → `scientificCacheClearApiCacheClearPost`
(ugly). Every route therefore declares an explicit `operation_id`.

## Conventions

- Every route declares an explicit `operation_id=`.
- The router's first tag (set in `src/app.py` `include_router(..., tags=[...])`)
  is the operation_id prefix: `health`, `cache`, `topology`, `verification`,
  `statistics`, `explainability`, `formal`, `sandbox`, `pipeline`.
- A fallback `generate_unique_id_function=_route_operation_id` in `src/app.py`
  produces `{tag}_{route.name}` if a future route forgets to declare one — the
  `tests/unit/test_openapi.py` guardrail still rejects literally auto-generated ids.

## Routes

### Health (`src/routers/health.py`, no prefix, tag `health`)

| Method | Path | Function | operation_id |
|---|---|---|---|
| GET | / (redirect to /docs, `include_in_schema=False`) | `root` | `redirect_root_to_docs` |
| GET | /health | `health` | `health_check` |
| GET | /health/ready | `readiness` | `health_readiness` |
| GET | /health/live | `liveness` | `health_liveness` |
| GET | /status | `status_check` | `health_status` |

### Cache (`src/routers/cache.py`, prefix `/api/cache`, tag `cache`)

| Method | Path | Function | operation_id |
|---|---|---|---|
| GET | /api/cache/stats | `cache_stats` | `cache_stats` |
| POST | /api/cache/invalidate | `cache_invalidate` | `cache_invalidate` |
| POST | /api/cache/clear | `cache_clear` | `cache_clear` |

### Topology (`src/routers/topology.py`, prefix `/api/topology`, tag `topology`)

| Method | Path | Function | operation_id |
|---|---|---|---|
| POST | /api/topology/subgraph | `extract_subgraph` | `topology_extract_subgraph` |
| POST | /api/topology/contradictions | `detect_contradictions` | `topology_detect_contradictions` |
| POST | /api/topology/authority | `compute_authority` | `topology_compute_authority` |
| POST | /api/topology/communities | `detect_communities` | `topology_detect_communities` |

### Verification (`src/routers/verification.py`, prefix `/api/verify`, tag `verification`)

| Method | Path | Function | operation_id |
|---|---|---|---|
| POST | /api/verify/cove | `verify_cove` | `verification_cove` |
| POST | /api/verify/ragas | `verify_ragas` | `verification_ragas` |
| POST | /api/verify/integrity | `compute_integrity` | `verification_integrity` |

### Statistics (`src/routers/statistics.py`, prefix `/api/stats`, tag `statistics`)

| Method | Path | Function | operation_id |
|---|---|---|---|
| POST | /api/stats/meta-analysis | `run_meta_analysis` | `statistics_meta_analysis` |
| POST | /api/stats/confidence-interval | `compute_confidence_interval` | `statistics_confidence_interval` |
| POST | /api/stats/trend | `detect_trend` | `statistics_trend` |

### Explainability (`src/routers/explainability.py`, prefix `/api/explain`, tag `explainability`)

| Method | Path | Function | operation_id |
|---|---|---|---|
| POST | /api/explain/subgraphx | `run_subgraphx` | `explainability_subgraphx` |
| GET | /api/explain/jobs/{job_id} | `get_explanation_job` | `explainability_get_job` |

### Formal (`src/routers/formal.py`, prefix `/api/formal`, tag `formal`)

| Method | Path | Function | operation_id |
|---|---|---|---|
| POST | /api/formal/translate | `translate_to_formal` | `formal_translate` |
| POST | /api/formal/verify | `verify_formal` | `formal_verify` |
| POST | /api/formal/proofs/{claim_id} | `create_proof` | `formal_create_proof` |
| GET | /api/formal/proofs/{claim_id} | `get_proofs` | `formal_get_proofs` |

### Sandbox (`src/routers/sandbox.py`, prefix `/api/sandbox`, tag `sandbox`)

| Method | Path | Function | operation_id |
|---|---|---|---|
| POST | /api/sandbox/execute | `execute_sandbox` | `sandbox_execute` |
| GET | /api/sandbox/jobs/{job_id} | `get_sandbox_job` | `sandbox_get_job` |
| GET | /api/sandbox/results/{job_id} | `get_sandbox_results` | `sandbox_get_results` |

### Pipeline (`src/routers/pipeline.py`, prefix `/api/pipeline`, tag `pipeline`)

| Method | Path | Function | operation_id |
|---|---|---|---|
| POST | /api/pipeline/trigger | `trigger_verification` | `pipeline_trigger` |
| POST | /api/pipeline/webhook | `webhook_notification` | `pipeline_webhook` |
| POST | /api/pipeline/batch | `batch_verification` | `pipeline_batch` |
| GET | /api/pipeline/status | `pipeline_status` | `pipeline_status` |
| GET | /api/pipeline/history/{entity_id} | `entity_history` | `pipeline_history` |
| GET | /api/pipeline/anomalies | `get_anomalies` | `pipeline_anomalies` |

**Total: 32 in-schema operations** (the `/` root redirect is `include_in_schema=False`).
