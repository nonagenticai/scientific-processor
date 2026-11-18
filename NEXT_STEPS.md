# Scientific Processor -- Status

## Implemented

All planned phases are complete:

- **Phase C (Graph Topology + Statistics)** -- Done. Subgraph extraction (k-hop via Cypher + NetworkX), contradiction detection, PageRank/eigenvector centrality (local + Graph Adapter), community detection (Louvain/Leiden/label propagation), meta-analysis (fixed/random effects), confidence intervals (bootstrap/t-dist), Mann-Kendall trend detection. Composite 5-signal integrity scoring.
- **Phase D (Verification: CoVe + RAGAS)** -- Done. Chain-of-Verification with LLM question generation + graph search + verdict aggregation. RAGAS faithfulness, answer relevancy, context precision.
- **Phase E (Sandbox Code Execution)** -- Done. Resource-limited code execution with timeout enforcement and safe expression evaluation.
- **Phase F (SubgraphX Explainability)** -- Done. MCTS + Shapley value GNN explainability with async job tracking. Edge masking, UCB1 exploration, progress reporting.
- **Phase G (Formal Verification)** -- Done. Z3 SAT/SMT, SymPy symbolic simplification, Lean4 theorem proving. LLM-guided translation to formal logic. Proof storage.
- **Phase H (Gateway + GitOps)** -- Done. KAG gateway proxies `/scientific/*`. GitOps manifests deployed.

Additional features beyond original plan:
- MinIO-backed verification history (HistoryService)
- Persistent job manager (PostgreSQL-backed for production)
- LRU caching with TTL for subgraph queries
- Pipeline orchestration with batch processing and webhook callbacks

621 unit tests, 97% coverage.

## Remaining Work

- **Lean4 binary in image** -- Current Lean4 prover client calls an external Lean4 binary; needs to be bundled in the Docker image or run as a sidecar
- **Sandbox K8s Jobs** -- SandboxService implementation uses local execution; K8s Job-based ephemeral containers for true isolation is not yet wired up
- **GNN model hosting** -- SubgraphX calls pytorch-backend for GNN inference; assumes a model is deployed there
- **MinIO bucket auto-creation** -- HistoryService assumes the bucket exists; could add auto-create on startup
