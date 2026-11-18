# Scientific Processor

Scientific data analysis and verification engine for KAG: graph topology analysis, Chain-of-Verification, RAGAS metrics, formal verification (Z3/SymPy/Lean4), SubgraphX explainability, statistical analysis, and sandboxed code execution. Port 8808.

## What Scientific Processor Does

When the KAG platform ingests documents and builds a knowledge graph, the raw data needs to be verified before anyone can trust it. Scientific Processor is the verification and quality engine that answers the question "how much should we trust this piece of knowledge?" It receives requests from other KAG services -- primarily through the KAG gateway -- and returns quantitative confidence scores, contradiction reports, and formal proofs that tell the platform whether a knowledge graph node is well-supported, internally consistent, and logically sound.

The service runs several independent verification strategies and combines their results. It analyzes graph topology to find contradictions between SUPPORTS and REFUTES edges, ranks node authority using PageRank, and detects communities of related knowledge. It runs Chain-of-Verification workflows where an LLM generates probe questions about a claim, searches the graph for answers, and aggregates the verdicts into a confidence score. It computes RAGAS metrics (faithfulness, relevancy, precision) to measure how well a retrieval system's answers hold up against their source context. For claims that can be expressed mathematically, it translates them into formal logic and runs them through Z3, SymPy, or Lean4 theorem provers for definitive verification.

All of these capabilities feed into a composite integrity score -- a single weighted number that combines citation support, cross-references, community membership, authority, and contradiction penalties. The full verification pipeline orchestrates these checks in sequence (integrity, then Chain-of-Verification, then formal verification), so a single API call can assess a claim from multiple angles. Results are persisted to MinIO for audit trails, and batch processing with webhook triggers supports high-throughput verification of large knowledge graphs.

| | |
|---|---|
| **Stack** | Python 3.14 · FastAPI · NetworkX 3.4 · SciPy · NumPy · Z3 · SymPy · httpx · structlog |
| **Database** | PostgreSQL (job persistence) or SQLite (tests) |
| **Runs on** | mf1 (K8s) |

## Capabilities

**Topology** -- Subgraph extraction (k-hop neighborhoods via Cypher), contradiction detection (SUPPORTS/REFUTES net scoring), authority scoring (PageRank/eigenvector centrality), community analysis (Louvain/Leiden/label propagation).

**Verification** -- Chain-of-Verification (LLM question generation + graph search + verdict aggregation), RAGAS (faithfulness, answer relevancy, context precision), composite integrity scoring (5-signal weighted: citation support, cross-reference, community membership, authority, contradiction penalty).

**Statistics** -- Fixed/random-effects meta-analysis with heterogeneity tests, bootstrap and t-distribution confidence intervals, Mann-Kendall trend detection with Theil-Sen slope.

**Explainability** -- SubgraphX via MCTS with Shapley value approximation for GNN prediction explanations. Async jobs with progress tracking.

**Formal Verification** -- LLM translates claims to formal logic. Z3 SAT/SMT verification, SymPy symbolic simplification, Lean4 theorem proving.

**Sandbox** -- Resource-limited code execution with timeout enforcement for hypothesis testing and custom computations.

**Pipeline** -- Full verification orchestration (integrity -> CoVe -> formal), batch processing, webhook triggers, MinIO-backed history tracking.

## API Endpoints

| Group | Prefix | Description |
|-------|--------|-------------|
| Topology | `/api/topology` | Subgraph extraction, contradiction detection, authority scoring, community detection |
| Verification | `/api/verify` | Chain-of-Verification, RAGAS metrics, composite integrity scoring |
| Explainability | `/api/explainability` | SubgraphX job launch and status polling |
| Formal | `/api/formal` | Translate claims to formal logic, verify, store proofs |
| Sandbox | `/api/sandbox` | Submit code for execution, poll job status |
| Pipeline | `/api/pipeline` | Full verification pipeline, batch, webhook trigger, history |
| Statistics | `/api/stats` | Meta-analysis, confidence intervals, trend detection |
| Health | `/health` | Liveness, readiness, health status |

## Quick Start

```bash
uv sync --all-extras
uv run uvicorn src.app:create_app --factory --host 0.0.0.0 --port 8808
```

## Configuration

All configuration via environment variables. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `SCIENTIFIC_PROCESSOR_GRAPH_ADAPTER_URL` | `http://graph-adapter...:8804` | Graph Adapter (Cypher, centrality, communities) |
| `SCIENTIFIC_PROCESSOR_PYTORCH_BACKEND_URL` | `http://pytorch-backend...:9100` | Embeddings and GNN inference |
| `SCIENTIFIC_PROCESSOR_INFERENCE_ROUTER_URL` | `http://inference-router...:8150` | LLM calls (CoVe, RAGAS, formal translation) |
| `SCIENTIFIC_PROCESSOR_DISCOVERY_ENGINE_URL` | `http://discovery-engine...:8807` | Academic paper cross-checks |
| `SCIENTIFIC_PROCESSOR_MINIO_ENDPOINT` | `localhost:9000` | Verification history storage |
| `SCIENTIFIC_PROCESSOR_WEIGHT_*` | varies | Integrity score component weights |
| `SCIENTIFIC_PROCESSOR_SUBGRAPH_MAX_HOPS` | `3` | Max traversal depth (capped at 5) |
| `SCIENTIFIC_PROCESSOR_SANDBOX_TIMEOUT` | `120` | Sandbox execution timeout (seconds) |
| `ENABLE_AUTH` | `true` | Istio header auth + trusted network bypass |
| `LOG_LEVEL` | `INFO` | Log level |

## Integration

Accessed via KAG gateway at `/scientific/*`. Upstream dependencies:
- **graph-adapter** (8804) -- knowledge graph queries, centrality, community detection
- **pytorch-backend** (9100) -- embeddings for RAGAS, GNN inference for SubgraphX
- **inference-router** (8150) -- LLM chat completions for CoVe, RAGAS, formal translation
- **discovery-engine** (8807) -- academic paper discovery and cross-checks

## Architecture

Service-oriented with HTTP clients to upstream KAG services. All graph state lives in Neo4j via Graph Adapter; this service is stateless except for async job tracking (PostgreSQL or in-memory).

```
src/
├── clients/        # HTTP clients (graph adapter, inference router, pytorch backend, discovery engine)
├── auth/           # Istio header auth middleware
├── models/         # Pydantic request/response models
├── services/       # Business logic (subgraph, contradiction, centrality, community, integrity,
│                   #   cove, ragas, stats, subgraphx, formal, sandbox, pipeline, history, cache)
├── routers/        # FastAPI route handlers (7 groups)
├── app.py          # FastAPI app factory
└── config.py       # Settings from env vars
```

## Tests

```bash
uv run pytest tests/unit -v        # 621 tests, 97% coverage
uv run ruff check src/ tests/      # Lint
uv run ruff format --check src/    # Format check
uv run ty check .                  # Type checking
```

Coverage threshold: 85% (enforced in pyproject.toml).

## CI/CD

GitHub Actions: `ruff + ty + unit tests --> build amd64 --> deploy to test namespace --> smoke test --> promote to :latest`. Image pushed to `192.168.1.114:5000/scientific_processor:sha-<commit>`. Flux auto-deploys to K3s.

## Licensing

This service verifies service licence tokens minted by the platform licence
service; it never mints them and ships no licence server. Nothing in the
application, the migrations or the test suite requires a licence key to run:
licence introspection is reached only through the upstream calls a deployment
configures.

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
