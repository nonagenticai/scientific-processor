"""SubgraphX explainability service — MCTS + Shapley values for GNN explanations."""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from typing import Any

from clients.minio_client import MinIOClient
from clients.pytorch_backend import PyTorchBackendClient
from log_config import get_logger
from services.job_manager import JobManager, job_terminal_state
from services.subgraph_service import SubgraphService

logger = get_logger(__name__)

# Retain references to fire-and-forget background tasks so the event loop cannot
# garbage-collect them mid-flight.
_background_tasks: set[asyncio.Task[Any]] = set()


def _spawn(coro: Any) -> asyncio.Task[Any]:
    """Schedule a background task and keep a strong reference until it finishes."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


class SubgraphXService:
    """Explain GNN predictions via SubgraphX (MCTS + Shapley values)."""

    def __init__(
        self,
        subgraph_service: SubgraphService,
        pytorch_client: PyTorchBackendClient,
        job_manager: JobManager,
        minio_client: MinIOClient | None,
    ) -> None:
        self.subgraph_service = subgraph_service
        self.pytorch_client = pytorch_client
        self.job_manager = job_manager
        self.minio_client = minio_client

    async def explain(
        self,
        node_id: str,
        model_type: str,
        num_iterations: int,
        subgraph_size: int,
        tenant_id: str | None,
    ) -> str:
        """Launch async SubgraphX explanation job. Returns job_id."""
        job_id = await self.job_manager.create_job(job_type="subgraphx", tenant_id=tenant_id)
        _spawn(self._run_explanation(job_id, node_id, model_type, num_iterations, subgraph_size, tenant_id))
        return job_id

    @staticmethod
    def _identity_features(n: int) -> list[list[float]]:
        """The n x n identity feature matrix, as the list-of-lists the GNN wire wants.

        Deliberately NOT ``np.eye(n).tolist()``. ``tolist()`` materialises n**2
        DISTINCT ``PyFloat`` objects — 40 000 of them at n=200 — one allocation per
        cell, in a C loop that holds the GIL from start to finish. Every cell is either
        0.0 or 1.0, so allocating a separate object for each is pure waste: copying a
        shared zero row and poking one 1.0 into it produces a byte-identical result
        with exactly 2 float objects total, because ``[0.0] * n`` and ``list.copy()``
        copy POINTERS.

        Measured in this service's venv (py3.14, median of 15, gc between reps):

            n      np.eye().tolist()   this        speedup
            500          7.7 ms        0.6 ms       12.7x   <- the real ceiling
            1000        30.6 ms        3.5 ms        8.8x
            2000       274.8 ms       33.8 ms        8.1x

        ``n`` is bounded at 500, not 5000: ``_run_explanation`` calls
        ``extract_subgraph()`` WITHOUT ``max_nodes``, so it takes that method's own
        default of 500, and ``SCIENTIFIC_PROCESSOR_SUBGRAPH_MAX_NODES`` (5000) is only
        the ceiling that default is clamped against.

        At 0.6 ms this needs no ``asyncio.to_thread``, and an earlier attempt to add
        one was actively counter-productive: ``tolist()`` never releases the GIL, so a
        worker thread cannot free the loop. Measured at n=500 — inline: 15.7 ms wall,
        0 heartbeat ticks; ``to_thread``: 18.4 ms wall, 3 ticks, max stall STILL
        15.7 ms; a genuinely GIL-releasing ``to_thread(time.sleep)`` over the same
        window: 405 ticks, max stall 10.2 ms. The offload bought 3 ticks out of 405
        and made the wall clock worse. Removing the n**2 allocation is what actually
        fixes it.

        Rows are independent lists (not aliases of one shared row): ``_evaluate_subgraph``
        stores them by reference into its masked payload.
        """
        zeros = [0.0] * n
        features: list[list[float]] = []
        for i in range(n):
            row = zeros.copy()
            row[i] = 1.0
            features.append(row)
        return features

    async def _run_explanation(
        self,
        job_id: str,
        node_id: str,
        model_type: str,
        num_iterations: int,
        subgraph_size: int,
        tenant_id: str | None,
    ) -> None:
        """Background task running MCTS + Shapley.

        The whole body runs under ``job_terminal_state``: whatever goes wrong —
        listed or not — the job ends up ``failed`` with the error recorded, never
        stranded at ``running``.
        """
        start = time.monotonic()
        async with job_terminal_state(self.job_manager, job_id, label="SubgraphX explanation"):
            await self.job_manager.update_job(job_id, status="running")

            # Step 1: Extract subgraph around target node
            subgraph_data = await self.subgraph_service.extract_subgraph(
                entity_id=node_id, max_hops=2, tenant_id=tenant_id
            )
            graph = subgraph_data.graph

            # Step 2: Convert to matrices
            nodes = list(graph.nodes())
            node_to_idx: dict[str, int] = {n: i for i, n in enumerate(nodes)}
            n = len(nodes)

            # Adjacency: edge_index as [[src_indices], [tgt_indices]]
            edge_index: list[list[int]] = [[], []]
            for u, v in graph.edges():
                edge_index[0].append(node_to_idx[u])
                edge_index[1].append(node_to_idx[v])

            # Node features: identity matrix (no per-node properties are wired yet).
            # O(n^2) cells but only 2 float objects and n list copies — ~0.6 ms at the
            # n=500 ceiling this path can actually reach. See _identity_features for why
            # it is not np.eye().tolist() and why it is not offloaded to a thread.
            node_features: list[list[float]] = self._identity_features(n)

            # Step 3: Baseline GNN score
            baseline = await self.pytorch_client.gnn_inference(
                node_features=node_features,
                edge_index=edge_index,
                model=model_type,
                tenant_id=tenant_id,
            )
            baseline_scores = baseline.get("scores", [])
            target_idx = node_to_idx.get(node_id, 0)
            if not baseline_scores or target_idx >= len(baseline_scores):
                # Backend returned an empty/short score vector — fail the job loudly
                # instead of letting an IndexError kill the task and leave the job
                # stuck "running" forever.
                raise ValueError(
                    f"GNN baseline returned no score for node {node_id!r} "
                    f"(scores length={len(baseline_scores)}, expected index {target_idx})"
                )
            baseline_score = baseline_scores[target_idx]

            await self.job_manager.update_job(job_id, progress=0.2)

            # Step 4: MCTS exploration
            best_subgraphs = await self._mcts_search(
                nodes,
                node_to_idx,
                node_features,
                edge_index,
                graph,
                node_id,
                model_type,
                num_iterations,
                subgraph_size,
                baseline_score,
                tenant_id,
                job_id,
            )

            await self.job_manager.update_job(job_id, progress=0.7)

            # Step 5: Shapley values for top subgraphs
            shapley_results = await self._compute_shapley_values(
                best_subgraphs,
                nodes,
                node_to_idx,
                node_features,
                edge_index,
                graph,
                node_id,
                model_type,
                baseline_score,
                tenant_id,
            )

            elapsed_ms = (time.monotonic() - start) * 1000
            result: dict[str, Any] = {
                "node_id": node_id,
                "model_type": model_type,
                "baseline_score": baseline_score,
                "influential_subgraphs": shapley_results,
                "num_iterations": num_iterations,
                "computation_time_ms": round(elapsed_ms, 2),
            }

            # Store to MinIO if available
            result_url = None
            if self.minio_client:
                key = f"subgraphx/{job_id}/result.json"
                await self.minio_client.upload(key, json.dumps(result).encode(), "application/json")
                result_url = key

            await self.job_manager.update_job(job_id, status="completed", result=result, result_url=result_url)

    async def _mcts_search(
        self,
        nodes: list[str],
        node_to_idx: dict[str, int],
        node_features: list[list[float]],
        edge_index: list[list[int]],
        graph: Any,
        target_node: str,
        model_type: str,
        num_iterations: int,
        subgraph_size: int,
        baseline_score: float,
        tenant_id: str | None,
        job_id: str,
    ) -> list[tuple[float, frozenset[int]]]:
        """Monte Carlo Tree Search for influential subgraphs."""
        target_idx = node_to_idx.get(target_node, 0)
        all_indices = list(range(len(nodes)))

        # Always include target node
        visits: dict[frozenset[int], int] = {}
        values: dict[frozenset[int], float] = {}
        best_subgraphs: list[tuple[float, frozenset[int]]] = []

        for iteration in range(num_iterations):
            # Selection: start from target node, expand
            current = frozenset([target_idx])

            # Expansion: add nodes greedily with UCB1
            while len(current) < subgraph_size:
                # Get neighbors of current set
                candidates: set[int] = set()
                for idx in current:
                    node = nodes[idx]
                    for neighbor in graph.neighbors(node):
                        n_idx = node_to_idx[neighbor]
                        if n_idx not in current:
                            candidates.add(n_idx)

                if not candidates:
                    # Try random nodes if no neighbors available
                    remaining = set(all_indices) - current
                    if not remaining:
                        break
                    candidates = remaining

                # UCB1 selection
                best_candidate = None
                best_ucb = -float("inf")
                total_parent_visits = visits.get(current, 1)

                for c in candidates:
                    child = current | {c}
                    child_visits = visits.get(child, 0)
                    if child_visits == 0:
                        # Unvisited — prioritize exploration
                        best_candidate = c
                        break
                    child_value = values.get(child, 0.0) / child_visits
                    ucb = child_value + math.sqrt(2 * math.log(total_parent_visits) / child_visits)
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_candidate = c

                if best_candidate is None:
                    best_candidate = random.choice(list(candidates))

                current = current | {best_candidate}

            # Rollout: evaluate this subgraph
            score = await self._evaluate_subgraph(
                current, nodes, node_to_idx, node_features, edge_index, graph, target_node, model_type, tenant_id
            )

            # Score delta — how much does this subgraph explain the prediction
            reward = abs(baseline_score - score) if baseline_score != 0 else score

            # Backpropagation
            visits[current] = visits.get(current, 0) + 1
            values[current] = values.get(current, 0.0) + reward

            best_subgraphs.append((reward, current))

            # Update progress
            progress = 0.2 + 0.5 * (iteration + 1) / num_iterations
            await self.job_manager.update_job(job_id, progress=round(progress, 2))

        # Return top unique subgraphs by reward
        best_subgraphs.sort(key=lambda x: x[0], reverse=True)
        seen: set[frozenset[int]] = set()
        unique_best: list[tuple[float, frozenset[int]]] = []
        for reward, sg in best_subgraphs:
            if sg not in seen:
                seen.add(sg)
                unique_best.append((reward, sg))
            if len(unique_best) >= 5:
                break

        return unique_best

    async def _compute_shapley_values(
        self,
        best_subgraphs: list[tuple[float, frozenset[int]]],
        nodes: list[str],
        node_to_idx: dict[str, int],
        node_features: list[list[float]],
        edge_index: list[list[int]],
        graph: Any,
        target_node: str,
        model_type: str,
        baseline_score: float,
        tenant_id: str | None,
    ) -> list[dict[str, Any]]:
        """Approximate Shapley values for nodes in each top subgraph."""
        results: list[dict[str, Any]] = []
        for reward, subgraph_indices in best_subgraphs:
            sg_nodes = [nodes[i] for i in subgraph_indices]
            shapley: dict[str, float] = {}

            # For each node in subgraph, estimate marginal contribution
            for node_idx in subgraph_indices:
                # Score without this node
                without = subgraph_indices - {node_idx}
                if not without:
                    shapley[nodes[node_idx]] = reward
                    continue

                score_without = await self._evaluate_subgraph(
                    without, nodes, node_to_idx, node_features, edge_index, graph, target_node, model_type, tenant_id
                )
                score_with = await self._evaluate_subgraph(
                    subgraph_indices,
                    nodes,
                    node_to_idx,
                    node_features,
                    edge_index,
                    graph,
                    target_node,
                    model_type,
                    tenant_id,
                )

                marginal = abs(score_with - score_without)
                shapley[nodes[node_idx]] = round(marginal, 6)

            results.append(
                {
                    "nodes": sg_nodes,
                    "reward": round(reward, 6),
                    "shapley_values": shapley,
                }
            )

        return results

    async def _evaluate_subgraph(
        self,
        node_indices: frozenset[int],
        nodes: list[str],
        node_to_idx: dict[str, int],
        node_features: list[list[float]],
        edge_index: list[list[int]],
        graph: Any,
        target_node: str,
        model_type: str,
        tenant_id: str | None,
    ) -> float:
        """Evaluate GNN on a subgraph (mask out non-included nodes)."""
        n = len(nodes)
        target_idx = node_to_idx.get(target_node, 0)
        included = set(node_indices)

        # Mask node features — zero out non-included nodes
        masked_features: list[list[float]] = []
        for i in range(n):
            if i in included:
                masked_features.append(node_features[i])
            else:
                masked_features.append([0.0] * len(node_features[i]))

        # Filter edges to only those between included nodes
        masked_edge_index: list[list[int]] = [[], []]
        for j in range(len(edge_index[0])):
            if edge_index[0][j] in included and edge_index[1][j] in included:
                masked_edge_index[0].append(edge_index[0][j])
                masked_edge_index[1].append(edge_index[1][j])

        result = await self.pytorch_client.gnn_inference(
            node_features=masked_features,
            edge_index=masked_edge_index,
            model=model_type,
            tenant_id=tenant_id,
        )

        scores = result.get("scores", [0.0] * n)
        if target_idx < len(scores):
            return scores[target_idx]
        return 0.0
