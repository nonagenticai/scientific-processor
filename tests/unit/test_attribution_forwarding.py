"""Caller project/job attribution is forwarded to every downstream backend.

Owner ruling 2026-10-07: a cost row's ``project_id`` is the CALLING app's own
project. scientific-processor emits no cost events itself, but every backend it
calls (inference-router, pytorch-backend, lean4-prover, graph-adapter,
discovery-engine) does — so it must forward the ``X-Project-Id`` / ``X-Job-Id``
its caller named, or that spend falls back to the tenant-level project.

These go through the REAL app (``create_app`` -> ``AttributionContextMiddleware``)
and the REAL cached ``httpx.AsyncClient`` each client builds, with only the
network transport swapped for an ``httpx.MockTransport`` that records what
would have left the pod. Deleting the middleware, or the ``event_hooks`` on any
one client, turns the matching case red.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from kag_common.attribution import clear_attribution_context

from app import create_app
from clients.discovery_engine import DiscoveryEngineClient
from clients.graph_adapter import GraphAdapterClient
from clients.inference_router import InferenceRouterClient
from clients.lean4_prover import Lean4ProverClient
from clients.pytorch_backend import PyTorchBackendClient
from config import Settings

pytestmark = pytest.mark.unit

BASE_URL = "http://downstream.test:9999"
PROJECT = "0195f48b-d000-7000-8000-0000000000aa"
JOB = "job-attr-42"

#: name -> factory for each outbound backend client.
CLIENTS: dict[str, Callable[[], Any]] = {
    "inference_router": lambda: InferenceRouterClient(BASE_URL),
    "pytorch_backend": lambda: PyTorchBackendClient(BASE_URL),
    "lean4_prover": lambda: Lean4ProverClient(BASE_URL, service_license_token="svc-token"),
    "graph_adapter": lambda: GraphAdapterClient(BASE_URL),
    "discovery_engine": lambda: DiscoveryEngineClient(BASE_URL),
}


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Swap the network transport every client builds for a recording mock."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", lambda **_kw: httpx.MockTransport(handler))
    return seen


def _app_calling(client_name: str) -> FastAPI:
    """The real app plus one probe route that makes a call through ``client_name``."""
    app = create_app(dataclasses.replace(Settings(), enable_auth=False))
    backend = CLIENTS[client_name]()

    async def _probe() -> dict[str, bool]:
        resp = await backend._ensure_client().get("/probe")
        await backend.close()
        return {"ok": resp.status_code == 200}

    app.add_api_route("/attribution-probe", _probe, methods=["GET"])
    return app


@pytest.mark.parametrize("client_name", sorted(CLIENTS))
def test_inbound_project_and_job_are_forwarded(client_name: str, recorded: list[httpx.Request]) -> None:
    with TestClient(_app_calling(client_name)) as tc:
        resp = tc.get("/attribution-probe", headers={"X-Project-Id": PROJECT, "X-Job-Id": JOB})

    assert resp.status_code == 200, resp.text
    assert len(recorded) == 1
    assert recorded[0].headers.get("X-Project-Id") == PROJECT
    assert recorded[0].headers.get("X-Job-Id") == JOB


@pytest.mark.parametrize("client_name", sorted(CLIENTS))
def test_w2v_spelling_wins_and_is_forwarded_generically(client_name: str, recorded: list[httpx.Request]) -> None:
    w2v_project = "0195f48b-d000-7000-8000-0000000000bb"
    with TestClient(_app_calling(client_name)) as tc:
        tc.get(
            "/attribution-probe",
            headers={"X-Project-Id": PROJECT, "X-W2V-Project-Id": w2v_project, "X-W2V-Job-Id": JOB},
        )

    assert recorded[0].headers.get("X-Project-Id") == w2v_project
    assert recorded[0].headers.get("X-Job-Id") == JOB


@pytest.mark.parametrize("client_name", sorted(CLIENTS))
def test_unbound_request_adds_no_attribution_header(client_name: str, recorded: list[httpx.Request]) -> None:
    with TestClient(_app_calling(client_name)) as tc:
        tc.get("/attribution-probe")

    assert len(recorded) == 1
    assert "X-Project-Id" not in recorded[0].headers
    assert "X-Job-Id" not in recorded[0].headers


def test_malformed_project_is_dropped_not_forwarded(recorded: list[httpx.Request]) -> None:
    with TestClient(_app_calling("inference_router")) as tc:
        tc.get("/attribution-probe", headers={"X-Project-Id": "not-a-uuid", "X-Job-Id": JOB})

    assert "X-Project-Id" not in recorded[0].headers
    assert recorded[0].headers.get("X-Job-Id") == JOB


async def test_background_work_outside_a_request_stays_unbound(recorded: list[httpx.Request]) -> None:
    """A job with no inbound request carries nothing -> billing's tenant-level fallback."""
    clear_attribution_context()
    backend = InferenceRouterClient(BASE_URL)
    await backend._ensure_client().get("/probe")
    await backend.close()

    assert "X-Project-Id" not in recorded[0].headers
    assert "X-Job-Id" not in recorded[0].headers
