"""Tests for CoVe integration with Discovery Engine — external source searching."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from models.verification import CoVeRequest
from services.cove_service import CoVeService

pytestmark = pytest.mark.unit


def _llm_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


@pytest.fixture
def inference_mock():
    mock = AsyncMock()
    mock.chat_completion = AsyncMock()
    return mock


@pytest.fixture
def graph_mock():
    mock = AsyncMock()
    mock.fulltext_search = AsyncMock(return_value={"results": []})
    return mock


@pytest.fixture
def discovery_mock():
    mock = AsyncMock()
    mock.search = AsyncMock(return_value={"results": [], "total": 0})
    return mock


@pytest.fixture
def service_with_discovery(inference_mock, graph_mock, discovery_mock):
    return CoVeService(
        inference_client=inference_mock,
        graph_client=graph_mock,
        discovery_client=discovery_mock,
    )


@pytest.fixture
def service_without_discovery(inference_mock, graph_mock):
    return CoVeService(
        inference_client=inference_mock,
        graph_client=graph_mock,
    )


class TestCoVeWithDiscovery:
    async def test_find_sources_uses_discovery_engine(self, service_with_discovery, discovery_mock, graph_mock):
        """When discovery client is available, it should be queried for sources."""
        graph_mock.fulltext_search.return_value = {"results": []}
        discovery_mock.search.return_value = {
            "results": [
                {"title": "Paper A", "doi": "10.1234/a", "confidence": 0.9},
            ],
            "total": 1,
        }

        sources = await service_with_discovery._find_sources("Is X true?", "t1", 0.5)
        discovery_mock.search.assert_called_once()
        # Should have external sources
        assert len(sources) >= 1

    async def test_find_sources_merges_graph_and_external(self, service_with_discovery, discovery_mock, graph_mock):
        """Sources from graph and discovery engine should be merged."""
        graph_mock.fulltext_search.return_value = {
            "results": [{"id": "n1", "name": "Graph Source", "labels": ["Claim"], "confidence": 0.8, "score": 0.9}],
        }
        discovery_mock.search.return_value = {
            "results": [
                {"title": "External Paper", "doi": "10.1234/x", "confidence": 0.85},
            ],
            "total": 1,
        }

        sources = await service_with_discovery._find_sources("test query", "t1", 0.5)
        assert len(sources) >= 2

    async def test_find_sources_graceful_without_discovery(self, service_without_discovery, graph_mock):
        """Without discovery client, only graph sources should be returned."""
        graph_mock.fulltext_search.return_value = {
            "results": [{"id": "n1", "name": "Source", "labels": [], "confidence": 0.7, "score": 0.8}],
        }

        sources = await service_without_discovery._find_sources("query", "t1", 0.5)
        assert len(sources) >= 1

    async def test_discovery_failure_degrades_gracefully(self, service_with_discovery, discovery_mock, graph_mock):
        """If discovery engine fails, should still return graph sources."""
        graph_mock.fulltext_search.return_value = {
            "results": [{"id": "n1", "name": "Graph Source", "labels": [], "confidence": 0.8, "score": 0.9}],
        }
        discovery_mock.search.side_effect = httpx.ConnectError("Discovery engine down")

        sources = await service_with_discovery._find_sources("query", "t1", 0.5)
        # Should still have graph sources despite discovery failure
        assert len(sources) >= 1

    async def test_verify_claim_with_discovery(
        self, service_with_discovery, inference_mock, graph_mock, discovery_mock
    ):
        """Full verify_claim flow with discovery engine providing external evidence."""
        inference_mock.chat_completion.side_effect = [
            _llm_response("1. Is X supported by evidence?"),
            _llm_response("SUPPORTS\nBased on external evidence"),
        ]
        graph_mock.fulltext_search.return_value = {"results": []}
        discovery_mock.search.return_value = {
            "results": [{"title": "Evidence Paper", "doi": "10.1/x", "confidence": 0.9}],
            "total": 1,
        }

        req = CoVeRequest(claim_text="X is true", num_questions=1)
        result = await service_with_discovery.verify_claim(req, "t1")
        assert result.overall_verdict in ("verified", "disputed", "inconclusive")
        assert result.computation_time_ms > 0
