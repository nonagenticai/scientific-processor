"""A job must never remain ``running`` after its background task has ended.

This is an *invariant*, not an exception taxonomy. #39 added ``IndexError`` to the
``except`` tuple in ``SubgraphXService._run_explanation``; that closed one hole and
left every other unlisted exception type with exactly the same behaviour — the task
dies, nothing marks the job failed, and the row stays ``running`` forever.

These tests deliberately raise exception types that are NOT enumerated in any
handler. If the fix is a taxonomy patch they fail; only a terminal-state guarantee
(try/finally or equivalent) makes them pass.
"""

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

TERMINAL = ("completed", "failed", "timeout")


def _subgraph_data() -> SubgraphData:
    g = nx.DiGraph()
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    return SubgraphData(
        response=SubgraphResponse(
            entity_id="A",
            nodes=[{"id": n, "label": "", "labels": [], "properties": {}} for n in g.nodes()],
            edges=[],
            node_count=g.number_of_nodes(),
            edge_count=g.number_of_edges(),
            max_hops_used=2,
            computation_time_ms=1.0,
        ),
        graph=g,
    )


@pytest.fixture
def job_manager():
    return JobManager()


def _service(job_manager, *, subgraph_side_effect=None, gnn_side_effect=None) -> SubgraphXService:
    subgraph_mock = AsyncMock()
    if subgraph_side_effect is not None:
        subgraph_mock.extract_subgraph = AsyncMock(side_effect=subgraph_side_effect)
    else:
        subgraph_mock.extract_subgraph = AsyncMock(return_value=_subgraph_data())

    pytorch_mock = AsyncMock()
    if gnn_side_effect is not None:
        pytorch_mock.gnn_inference = AsyncMock(side_effect=gnn_side_effect)
    else:
        pytorch_mock.gnn_inference = AsyncMock(return_value={"scores": [0.8, 0.1, 0.1]})

    return SubgraphXService(
        subgraph_service=subgraph_mock,
        pytorch_client=pytorch_mock,
        job_manager=job_manager,
        minio_client=AsyncMock(),
    )


async def _run_and_get(service: SubgraphXService, job_manager: JobManager):
    job_id = await service.explain(node_id="A", model_type="gcn", num_iterations=3, subgraph_size=2, tenant_id="t1")
    for _ in range(50):
        await asyncio.sleep(0.02)
        job = await job_manager.get_job(job_id)
        if job is not None and job.status in TERMINAL:
            return job
    return await job_manager.get_job(job_id)


# ---------------------------------------------------------------------------
# RED PROOF 2 — an exception type nobody enumerated strands the job at `running`.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("event loop closed mid-inference"),
        TypeError("graph adapter returned a str where a dict was expected"),
        AttributeError("'NoneType' object has no attribute 'nodes'"),
        ZeroDivisionError("division by zero"),
    ],
    ids=["RuntimeError", "TypeError", "AttributeError", "ZeroDivisionError"],
)
async def test_unenumerated_exception_still_terminates_the_job(job_manager, exc):
    """None of these are in the handler's tuple. The job must still end up failed."""
    service = _service(job_manager, subgraph_side_effect=exc)

    job = await _run_and_get(service, job_manager)

    assert job is not None
    assert job.status in TERMINAL, f"job stranded at {job.status!r} after the task raised {type(exc).__name__}"
    assert job.status == "failed"
    assert job.error, "a failed job must record why it failed"
    assert type(exc).__name__ in job.error or str(exc) in job.error


async def test_unenumerated_exception_deep_in_the_pipeline_terminates_the_job(job_manager):
    """The crash happens after the job is already ``running`` and past step 3."""
    service = _service(job_manager, gnn_side_effect=MemoryError("cannot allocate feature matrix"))

    job = await _run_and_get(service, job_manager)

    assert job is not None
    assert job.status == "failed", f"job stranded at {job.status!r}"
    assert job.error


async def test_index_error_terminates_the_job(job_manager):
    """IndexError is a LookupError, not a KeyError.

    NOTE: #39 already enumerated ``IndexError`` in the handler tuple, so this
    passes on pristine main. It is kept as a regression guard, not as a RED proof.
    """
    service = _service(job_manager, subgraph_side_effect=IndexError("list index out of range"))

    job = await _run_and_get(service, job_manager)

    assert job is not None
    assert job.status == "failed"


async def test_happy_path_still_completes(job_manager):
    """The terminal-state guard must not mask success."""
    service = _service(job_manager)

    job = await _run_and_get(service, job_manager)

    assert job is not None
    assert job.status == "completed"
    assert job.progress == 1.0
