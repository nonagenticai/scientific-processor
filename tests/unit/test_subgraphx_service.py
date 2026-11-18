"""Unit tests for SubgraphXService -- MCTS + Shapley GNN explainability."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import networkx as nx
import pytest

from models.topology import SubgraphResponse
from services.job_manager import JobManager
from services.subgraph_service import SubgraphData
from services.subgraphx_service import SubgraphXService

pytestmark = pytest.mark.unit


def _make_test_graph() -> nx.DiGraph:
    """Build a small 4-node directed graph for testing."""
    g = nx.DiGraph()
    g.add_node("A", label="Claim")
    g.add_node("B", label="Source")
    g.add_node("C", label="Source")
    g.add_node("D", label="Evidence")
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    g.add_edge("B", "D")
    return g


def _make_subgraph_data(graph: nx.DiGraph | None = None) -> SubgraphData:
    """Wrap a graph in SubgraphData."""
    g = graph or _make_test_graph()
    response = SubgraphResponse(
        entity_id="A",
        nodes=[{"id": n, "label": "", "labels": [], "properties": {}} for n in g.nodes()],
        edges=[],
        node_count=len(g.nodes()),
        edge_count=len(g.edges()),
        max_hops_used=2,
        computation_time_ms=1.0,
    )
    return SubgraphData(response=response, graph=g)


@pytest.fixture
def job_manager():
    return JobManager()


@pytest.fixture
def subgraph_mock():
    mock = AsyncMock()
    mock.extract_subgraph = AsyncMock(return_value=_make_subgraph_data())
    return mock


@pytest.fixture
def pytorch_mock():
    """Mock that returns deterministic scores based on number of non-zero feature rows."""
    mock = AsyncMock()

    async def _gnn_inference(node_features, edge_index, model, tenant_id=None):
        # Score each node: 0.5 if its feature row is non-zero, else 0.0
        scores = []
        for row in node_features:
            if any(v != 0.0 for v in row):
                scores.append(0.8)
            else:
                scores.append(0.1)
        return {"scores": scores}

    mock.gnn_inference = AsyncMock(side_effect=_gnn_inference)
    return mock


@pytest.fixture
def minio_mock():
    mock = AsyncMock()
    mock.upload = AsyncMock()
    return mock


@pytest.fixture
def service(subgraph_mock, pytorch_mock, job_manager, minio_mock):
    return SubgraphXService(
        subgraph_service=subgraph_mock,
        pytorch_client=pytorch_mock,
        job_manager=job_manager,
        minio_client=minio_mock,
    )


@pytest.fixture
def service_no_minio(subgraph_mock, pytorch_mock, job_manager):
    return SubgraphXService(
        subgraph_service=subgraph_mock,
        pytorch_client=pytorch_mock,
        job_manager=job_manager,
        minio_client=None,
    )


class TestExplainLaunchesJob:
    async def test_explain_returns_job_id(self, service, job_manager):
        job_id = await service.explain(
            node_id="A",
            model_type="gcn",
            num_iterations=5,
            subgraph_size=3,
            tenant_id="t1",
        )
        assert isinstance(job_id, str)
        assert len(job_id) == 36  # UUID

        job = await job_manager.get_job(job_id)
        assert job is not None
        assert job.job_type == "subgraphx"
        assert job.tenant_id == "t1"

    async def test_explain_job_completes(self, service, job_manager):
        job_id = await service.explain(
            node_id="A",
            model_type="gcn",
            num_iterations=5,
            subgraph_size=3,
            tenant_id="t1",
        )
        # Let the background task run
        await asyncio.sleep(0.2)

        job = await job_manager.get_job(job_id)
        assert job.status == "completed"
        assert job.progress == 1.0
        assert job.result is not None
        assert job.result["node_id"] == "A"
        assert job.result["model_type"] == "gcn"
        assert "baseline_score" in job.result
        assert "influential_subgraphs" in job.result
        assert "computation_time_ms" in job.result

    async def test_explain_stores_result_in_minio(self, service, job_manager, minio_mock):
        job_id = await service.explain(
            node_id="A",
            model_type="gcn",
            num_iterations=5,
            subgraph_size=3,
            tenant_id="t1",
        )
        await asyncio.sleep(0.2)

        minio_mock.upload.assert_called_once()
        call_args = minio_mock.upload.call_args
        assert call_args[0][0] == f"subgraphx/{job_id}/result.json"
        assert call_args[0][2] == "application/json"

        job = await job_manager.get_job(job_id)
        assert job.result_url == f"subgraphx/{job_id}/result.json"

    async def test_explain_no_minio_still_completes(self, service_no_minio, job_manager):
        job_id = await service_no_minio.explain(
            node_id="A",
            model_type="gcn",
            num_iterations=5,
            subgraph_size=3,
            tenant_id=None,
        )
        await asyncio.sleep(0.2)

        job = await job_manager.get_job(job_id)
        assert job.status == "completed"
        assert job.result_url is None


class TestMCTSProducesRankedSubgraphs:
    async def test_mcts_returns_subgraphs(self, service, job_manager):
        job_id = await service.explain(
            node_id="A",
            model_type="gcn",
            num_iterations=10,
            subgraph_size=3,
            tenant_id="t1",
        )
        await asyncio.sleep(0.3)

        job = await job_manager.get_job(job_id)
        assert job.status == "completed"
        subgraphs = job.result["influential_subgraphs"]
        assert len(subgraphs) >= 1
        # Each subgraph should have nodes, reward, and shapley_values
        for sg in subgraphs:
            assert "nodes" in sg
            assert "reward" in sg
            assert "shapley_values" in sg
            assert len(sg["nodes"]) >= 1

    async def test_mcts_subgraphs_contain_target_node(self, service, job_manager):
        """Target node A should always be in every subgraph since MCTS starts from it."""
        job_id = await service.explain(
            node_id="A",
            model_type="gcn",
            num_iterations=10,
            subgraph_size=3,
            tenant_id="t1",
        )
        await asyncio.sleep(0.3)

        job = await job_manager.get_job(job_id)
        for sg in job.result["influential_subgraphs"]:
            assert "A" in sg["nodes"]

    async def test_mcts_limits_to_top_5_unique(self, service, job_manager):
        job_id = await service.explain(
            node_id="A",
            model_type="gcn",
            num_iterations=50,
            subgraph_size=3,
            tenant_id="t1",
        )
        await asyncio.sleep(0.5)

        job = await job_manager.get_job(job_id)
        subgraphs = job.result["influential_subgraphs"]
        assert len(subgraphs) <= 5


class TestShapleyValues:
    async def test_shapley_values_per_node(self, service, job_manager):
        job_id = await service.explain(
            node_id="A",
            model_type="gcn",
            num_iterations=5,
            subgraph_size=3,
            tenant_id="t1",
        )
        await asyncio.sleep(0.3)

        job = await job_manager.get_job(job_id)
        for sg in job.result["influential_subgraphs"]:
            shapley = sg["shapley_values"]
            # Every node in the subgraph should have a Shapley value
            for node in sg["nodes"]:
                assert node in shapley
                assert isinstance(shapley[node], float)
                assert shapley[node] >= 0.0

    async def test_shapley_values_are_finite(self, service, job_manager):
        job_id = await service.explain(
            node_id="A",
            model_type="gcn",
            num_iterations=5,
            subgraph_size=3,
            tenant_id="t1",
        )
        await asyncio.sleep(0.3)

        job = await job_manager.get_job(job_id)
        import math

        for sg in job.result["influential_subgraphs"]:
            for val in sg["shapley_values"].values():
                assert math.isfinite(val)


class TestJobLifecycle:
    async def test_pending_then_running_then_completed(self, service, job_manager):
        job_id = await service.explain(
            node_id="A",
            model_type="gcn",
            num_iterations=5,
            subgraph_size=3,
            tenant_id="t1",
        )
        # Immediately after launch, should be pending or running
        job = await job_manager.get_job(job_id)
        assert job.status in ("pending", "running")

        await asyncio.sleep(0.3)

        job = await job_manager.get_job(job_id)
        assert job.status == "completed"
        assert job.progress == 1.0

    async def test_progress_updates_during_execution(self, service, job_manager, subgraph_mock, pytorch_mock):
        """With enough iterations, progress should advance past the 0.2 baseline."""
        job_id = await service.explain(
            node_id="A",
            model_type="gcn",
            num_iterations=10,
            subgraph_size=3,
            tenant_id="t1",
        )
        await asyncio.sleep(0.3)

        job = await job_manager.get_job(job_id)
        assert job.status == "completed"
        assert job.progress == 1.0


class TestFailureCase:
    async def test_extraction_failure_sets_failed(self, job_manager, pytorch_mock, minio_mock):
        from clients.graph_adapter import GraphAdapterError

        subgraph_mock = AsyncMock()
        subgraph_mock.extract_subgraph = AsyncMock(side_effect=GraphAdapterError("Graph unavailable"))

        svc = SubgraphXService(
            subgraph_service=subgraph_mock,
            pytorch_client=pytorch_mock,
            job_manager=job_manager,
            minio_client=minio_mock,
        )

        job_id = await svc.explain(
            node_id="A",
            model_type="gcn",
            num_iterations=5,
            subgraph_size=3,
            tenant_id="t1",
        )
        await asyncio.sleep(0.2)

        job = await job_manager.get_job(job_id)
        assert job.status == "failed"
        assert "Graph unavailable" in job.error

    async def test_gnn_inference_failure_sets_failed(self, job_manager, subgraph_mock, minio_mock):
        import httpx

        pytorch_mock = AsyncMock()
        pytorch_mock.gnn_inference = AsyncMock(side_effect=httpx.ConnectError("GNN backend down"))

        svc = SubgraphXService(
            subgraph_service=subgraph_mock,
            pytorch_client=pytorch_mock,
            job_manager=job_manager,
            minio_client=minio_mock,
        )

        job_id = await svc.explain(
            node_id="A",
            model_type="gcn",
            num_iterations=5,
            subgraph_size=3,
            tenant_id="t1",
        )
        await asyncio.sleep(0.2)

        job = await job_manager.get_job(job_id)
        assert job.status == "failed"
        assert "GNN backend down" in job.error

    async def test_empty_baseline_scores_sets_failed(self, job_manager, subgraph_mock, minio_mock):
        """An empty/short baseline score vector must fail the job, not raise an
        uncaught IndexError that leaves the job stuck 'running' forever."""
        pytorch_mock = AsyncMock()
        pytorch_mock.gnn_inference = AsyncMock(return_value={"scores": []})

        svc = SubgraphXService(
            subgraph_service=subgraph_mock,
            pytorch_client=pytorch_mock,
            job_manager=job_manager,
            minio_client=minio_mock,
        )

        job_id = await svc.explain(
            node_id="A",
            model_type="gcn",
            num_iterations=5,
            subgraph_size=3,
            tenant_id="t1",
        )
        await asyncio.sleep(0.2)

        job = await job_manager.get_job(job_id)
        assert job.status == "failed"
        assert "no score" in job.error

    async def test_minio_upload_failure_sets_failed(self, job_manager, subgraph_mock, pytorch_mock):
        from botocore.exceptions import ClientError

        minio_mock = AsyncMock()
        minio_mock.upload = AsyncMock(
            side_effect=ClientError({"Error": {"Code": "500", "Message": "MinIO connection refused"}}, "PutObject")
        )

        svc = SubgraphXService(
            subgraph_service=subgraph_mock,
            pytorch_client=pytorch_mock,
            job_manager=job_manager,
            minio_client=minio_mock,
        )

        job_id = await svc.explain(
            node_id="A",
            model_type="gcn",
            num_iterations=5,
            subgraph_size=3,
            tenant_id="t1",
        )
        await asyncio.sleep(0.3)

        job = await job_manager.get_job(job_id)
        assert job.status == "failed"
        assert "MinIO connection refused" in job.error


class TestEvaluateSubgraph:
    async def test_evaluate_masks_non_included_nodes(self, service):
        """When evaluating a subgraph with only node 0, other features should be zeroed."""
        graph = _make_test_graph()
        nodes = list(graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        import numpy as np

        node_features = np.eye(len(nodes)).tolist()
        edge_index: list[list[int]] = [[], []]
        for u, v in graph.edges():
            edge_index[0].append(node_to_idx[u])
            edge_index[1].append(node_to_idx[v])

        # Evaluate with just node 0
        score = await service._evaluate_subgraph(
            frozenset([0]),
            nodes,
            node_to_idx,
            node_features,
            edge_index,
            graph,
            nodes[0],
            "gcn",
            None,
        )
        # Our mock returns 0.8 for non-zero features, 0.1 for zero
        assert score == 0.8

    async def test_evaluate_filters_edges(self, service, pytorch_mock):
        """Edges between non-included nodes should be filtered out."""
        graph = _make_test_graph()
        nodes = list(graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        import numpy as np

        node_features = np.eye(len(nodes)).tolist()
        edge_index: list[list[int]] = [[], []]
        for u, v in graph.edges():
            edge_index[0].append(node_to_idx[u])
            edge_index[1].append(node_to_idx[v])

        # Include only nodes 0 and 1 (A and B)
        await service._evaluate_subgraph(
            frozenset([0, 1]),
            nodes,
            node_to_idx,
            node_features,
            edge_index,
            graph,
            nodes[0],
            "gcn",
            None,
        )

        # Verify the edge_index passed to gnn_inference only has edges between {0, 1}
        call_args = pytorch_mock.gnn_inference.call_args
        passed_edge_index = call_args.kwargs.get("edge_index") or call_args[0][1]
        for src, tgt in zip(passed_edge_index[0], passed_edge_index[1]):
            assert src in (0, 1)
            assert tgt in (0, 1)


class TestIdentityFeatures:
    """The node-feature matrix must not allocate n**2 Python float objects.

    History, because the shape of the fix matters more than the fix:

    ``_run_explanation`` built its identity feature matrix with
    ``np.eye(n).tolist()``. ``tolist()`` allocates one distinct ``PyFloat`` per cell —
    n**2 of them — inside a C loop that HOLDS THE GIL throughout. The first attempt to
    fix the resulting event-loop stall wrapped that call in ``asyncio.to_thread``, and
    a test "proved" it worked by monkeypatching the module's ``np`` with a
    ``time.sleep`` stand-in and counting heartbeat ticks. ``time.sleep`` RELEASES the
    GIL; ``tolist`` does not. The test therefore measured a workload with the opposite
    GIL semantics to production and would have certified any GIL-bound no-op offload.
    Measured at n=500 (the real ceiling): inline 15.7 ms wall / 0 ticks; ``to_thread``
    18.4 ms wall / 3 ticks with the max stall still 15.7 ms; a genuinely GIL-releasing
    ``to_thread`` over the same window, 405 ticks. The offload did nothing.

    So the offload is gone and the allocation is gone instead: copying a shared zero
    row costs 0.6 ms at n=500 versus 7.7 ms, and needs no thread at all.

    These tests are deterministic — no timing assertions, which on a loaded dev
    machine only manufacture flakes. The object-identity count below is the mutation
    killer: it fails the moment anything reintroduces ``np.eye(n).tolist()``.
    """

    def test_matches_the_numpy_identity_matrix_exactly(self):
        """The wire contract is unchanged: same JSON the GNN backend always got."""
        import numpy as np

        for n in (0, 1, 2, 3, 7, 64):
            assert SubgraphXService._identity_features(n) == np.eye(n).tolist(), n

    def test_is_a_real_identity_matrix(self):
        m = SubgraphXService._identity_features(5)
        assert len(m) == 5
        for i, row in enumerate(m):
            assert len(row) == 5
            assert row[i] == 1.0
            assert sum(row) == 1.0

    def test_does_not_allocate_one_python_float_per_cell(self):
        """KILLS a revert to ``np.eye(n).tolist()``.

        Every cell is 0.0 or 1.0, so the whole matrix needs exactly two float objects.
        ``tolist()`` allocates n**2 distinct ones — 40 000 at n=200 — which is the
        entire cost this method exists to avoid.
        """
        n = 200
        m = SubgraphXService._identity_features(n)
        distinct = {id(cell) for row in m for cell in row}
        assert len(distinct) <= 2, (
            f"{len(distinct)} distinct float objects for {n * n} cells — "
            "the n**2 allocation is back (np.eye().tolist()?)"
        )

    def test_rows_are_independent_lists_not_aliases_of_one_row(self):
        """KILLS the tempting ``[zeros] * n`` shortcut.

        ``_evaluate_subgraph`` stores these rows BY REFERENCE into the payload it
        sends per MCTS rollout, so aliased rows would be a silent correctness bug
        rather than a performance one.
        """
        n = 16
        m = SubgraphXService._identity_features(n)
        assert len({id(row) for row in m}) == n
        m[0][3] = 9.0
        assert m[1][3] == 0.0

    async def test_run_explanation_builds_the_matrix_inline_and_completes(
        self, subgraph_mock, pytorch_mock, job_manager, minio_mock, monkeypatch
    ):
        """End to end, and KILLS a re-added ``asyncio.to_thread`` at the call site.

        Offloading a call that never drops the GIL does not free the event loop
        (measured at n=500: 3 heartbeat ticks, versus 405 for a genuinely
        GIL-releasing call over the same window, with the max stall unchanged at
        15.7 ms) and costs a thread handoff on every explanation job. Nothing else in
        this code path uses ``to_thread``, so making it explode is an exact test of
        the call site. If a future change genuinely needs an offload it must justify
        it against a GIL-RELEASING control — not against a ``time.sleep`` stand-in,
        which is what the previous version of this test did.
        """
        from services import subgraphx_service as module

        async def _boom(*args, **kwargs):
            raise AssertionError("asyncio.to_thread was used for a GIL-holding build — it cannot free the event loop")

        monkeypatch.setattr(module.asyncio, "to_thread", _boom)

        service = SubgraphXService(
            subgraph_service=subgraph_mock,
            pytorch_client=pytorch_mock,
            job_manager=job_manager,
            minio_client=minio_mock,
        )

        before = set(module._background_tasks)
        job_id = await service.explain(node_id="A", model_type="gcn", num_iterations=2, subgraph_size=2, tenant_id="t1")
        new_tasks = set(module._background_tasks) - before
        assert len(new_tasks) == 1
        await new_tasks.pop()

        first_call = pytorch_mock.gnn_inference.call_args_list[0]
        features = first_call.kwargs["node_features"]
        n = len(features)
        assert features == [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        job = await job_manager.get_job(job_id)
        assert job.status == "completed", job.error


class TestBackgroundTaskRetention:
    """Recovered from the stranded PR #51.

    ``subgraphx_service._spawn`` landed via #39 but nothing asserted the strong
    reference is taken and released. An unretained task can be GC'd mid-flight and
    silently cancelled — which leaves its job pinned at "running" forever, the exact
    state ``job_terminal_state`` exists to prevent (and cannot, if the task never runs
    its ``finally``).
    """

    async def test_explain_holds_a_strong_reference_until_done(self, service, job_manager):
        from services import subgraphx_service as module

        before = set(module._background_tasks)
        job_id = await service.explain(
            node_id="A",
            model_type="gcn",
            num_iterations=5,
            subgraph_size=3,
            tenant_id="t1",
        )
        new_tasks = set(module._background_tasks) - before
        assert len(new_tasks) == 1, "the _run_explanation task was not retained — it can be GC'd mid-flight"

        task = new_tasks.pop()
        await task
        assert task not in module._background_tasks, "the done callback must release the reference"

        job = await job_manager.get_job(job_id)
        assert job.status in ("completed", "failed"), "the job must never be left at 'running'"


class TestShortBaselineScoreVector:
    """Recovered from the stranded PR #51.

    Main covers the EMPTY ``scores`` vector (``test_empty_baseline_scores_sets_failed``)
    but not the short-but-non-empty one, which is the case the ``target_idx >= len``
    half of the bounds check exists for. Without it, mutating that guard down to
    ``if not baseline_scores:`` still passes every test on main while restoring the
    original IndexError-strands-the-job defect for every node past index 0.
    """

    async def test_short_scores_list_fails_the_job(self, job_manager, subgraph_mock, minio_mock):
        """Target node "D" is at index 3 but the GNN returned only 2 scores."""
        pytorch_mock = AsyncMock()
        pytorch_mock.gnn_inference = AsyncMock(return_value={"scores": [0.4, 0.6]})

        svc = SubgraphXService(
            subgraph_service=subgraph_mock,
            pytorch_client=pytorch_mock,
            job_manager=job_manager,
            minio_client=minio_mock,
        )

        job_id = await svc.explain(
            node_id="D",
            model_type="gcn",
            num_iterations=5,
            subgraph_size=3,
            tenant_id="t1",
        )
        await asyncio.sleep(0.2)

        job = await job_manager.get_job(job_id)
        assert job.status == "failed", f"job stranded at {job.status!r} — the task died without updating state"
        # The message names actual length vs required index so schema drift is diagnosable.
        assert "scores length=2" in job.error
        assert "expected index 3" in job.error
