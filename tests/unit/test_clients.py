"""
Unit tests for HTTP clients: GraphAdapter, PyTorchBackend, InferenceRouter, DiscoveryEngine.

All external HTTP calls are mocked by injecting a mock httpx.AsyncClient via _client.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from auth.tenant import SOURCE_LICENSE_TOKEN, SOURCE_UNVOUCHED_HEADER, tenant_source_var
from clients.discovery_engine import DiscoveryEngineClient
from clients.graph_adapter import GraphAdapterClient
from clients.inference_router import InferenceRouterClient
from clients.pytorch_backend import PyTorchBackendClient

pytestmark = pytest.mark.unit

BASE_URL = "http://test-service:8000"
TENANT = "tenant-abc"

#: Every client now builds its outbound headers with
#: ``kag_common.auth.outbound_headers`` — the fleet's single definition of the
#: inter-service header contract — instead of hand-rolling a dict each. That
#: builder always sets ``Content-Type``, so it appears in the dicts asserted
#: below. The WIRE is unchanged: httpx already set exactly this header for every
#: ``json=`` POST/PUT these tests cover; these assertions compare the dict the
#: client hands to httpx, which is one layer earlier.
JSON_CT = {"Content-Type": "application/json"}
#: A licence id, as it arrives from an introspected license payload.
LICENSE_ID = "lic-cust-0195f48b"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx.Response with .json() and .raise_for_status()."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


def _inject_mock_client(client_obj: Any, response: MagicMock) -> AsyncMock:
    """Inject a mock httpx.AsyncClient into a client's _client attribute."""
    mock = AsyncMock()
    mock.get.return_value = response
    mock.post.return_value = response
    mock.put.return_value = response
    mock.is_closed = False
    client_obj._client = mock
    return mock


# ===========================================================================
# GraphAdapterClient
# ===========================================================================


class TestGraphAdapterClient:
    """Tests for GraphAdapterClient."""

    def test_init_strips_trailing_slash(self) -> None:
        c = GraphAdapterClient("http://host:9000/")
        assert c.base_url == "http://host:9000"

    def test_init_default_timeout(self) -> None:
        c = GraphAdapterClient(BASE_URL)
        assert c.timeout == 60.0

    def test_init_custom_timeout(self) -> None:
        c = GraphAdapterClient(BASE_URL, timeout=30.0)
        assert c.timeout == 30.0

    def test_headers_with_tenant(self) -> None:
        c = GraphAdapterClient(BASE_URL)
        assert c._headers(TENANT) == {**JSON_CT, "X-Tenant-Id": TENANT}

    def test_headers_without_tenant(self) -> None:
        c = GraphAdapterClient(BASE_URL)
        assert c._headers(None) == JSON_CT

    def test_headers_no_service_token_by_default(self) -> None:
        # WS-4: additive — no service token => no Authorization (today's behaviour).
        c = GraphAdapterClient(BASE_URL)
        assert "Authorization" not in c._headers(TENANT)

    def test_headers_with_service_token_adds_authorization(self) -> None:
        # WS-4 on-behalf-of: service identity rides Authorization; the data
        # tenant stays in the (unchanged) plain X-Tenant-Id.
        c = GraphAdapterClient(BASE_URL, service_token="svctok")
        assert c._headers(TENANT) == {
            **JSON_CT,
            "X-Tenant-Id": TENANT,
            "Authorization": "Bearer svctok",
        }

    def test_headers_service_token_without_tenant(self) -> None:
        c = GraphAdapterClient(BASE_URL, service_token="svctok")
        assert c._headers(None) == {**JSON_CT, "Authorization": "Bearer svctok"}

    def test_empty_service_token_is_normalized_to_none(self) -> None:
        c = GraphAdapterClient(BASE_URL, service_token="")
        assert c.service_token is None
        assert "Authorization" not in c._headers(TENANT)

    # -- execute_cypher ------------------------------------------------------

    async def test_execute_cypher_url_and_body(self) -> None:
        resp = _mock_response({"records": [{"n": 1}]})
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        result = await client.execute_cypher("MATCH (n) RETURN n", tenant_id=TENANT)

        mock.post.assert_called_once_with(
            "/query",
            json={"query": "MATCH (n) RETURN n", "parameters": {}, "query_type": "cypher"},
            headers={**JSON_CT, "X-Tenant-Id": TENANT},
        )
        assert result == {"records": [{"n": 1}]}

    async def test_execute_cypher_with_parameters(self) -> None:
        resp = _mock_response({"records": []})
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        await client.execute_cypher("MATCH (n {id: $id})", parameters={"id": "x"})

        call_json = mock.post.call_args[1]["json"]
        assert call_json["parameters"] == {"id": "x"}

    async def test_execute_cypher_no_tenant(self) -> None:
        resp = _mock_response({"records": []})
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        await client.execute_cypher("RETURN 1")

        call_headers = mock.post.call_args[1]["headers"]
        assert call_headers == JSON_CT

    async def test_execute_cypher_raises_on_http_error(self) -> None:
        from clients.graph_adapter import GraphAdapterError

        resp = _mock_response({}, status_code=500)
        resp.text = "Internal Server Error"
        resp.raise_for_status.side_effect = httpx.HTTPStatusError("Server Error", request=MagicMock(), response=resp)
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)
        mock.post.return_value = resp

        with pytest.raises(GraphAdapterError) as exc_info:
            await client.execute_cypher("BAD")
        assert exc_info.value.status_code == 500

    async def test_execute_cypher_raises_unavailable_on_connect_error(self) -> None:
        from clients.graph_adapter import GraphServiceUnavailableError

        client = GraphAdapterClient(BASE_URL)
        mock = AsyncMock()
        mock.is_closed = False
        mock.post.side_effect = httpx.ConnectError("Connection refused")
        client._client = mock

        with pytest.raises(GraphServiceUnavailableError):
            await client.execute_cypher("MATCH (n) RETURN n")

    # -- get_node ------------------------------------------------------------

    async def test_get_node_url(self) -> None:
        resp = _mock_response({"id": "n1", "label": "Concept"})
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        result = await client.get_node("n1", tenant_id=TENANT)

        mock.get.assert_called_once_with(
            "/nodes/n1",
            headers={**JSON_CT, "X-Tenant-Id": TENANT},
        )
        assert result["id"] == "n1"

    async def test_get_node_no_tenant(self) -> None:
        resp = _mock_response({"id": "n2"})
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        await client.get_node("n2")

        call_headers = mock.get.call_args[1]["headers"]
        assert call_headers == JSON_CT

    # -- get_centrality ------------------------------------------------------

    async def test_get_centrality_defaults(self) -> None:
        resp = _mock_response({"scores": []})
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        result = await client.get_centrality()

        mock.post.assert_called_once_with(
            "/analytics/centrality",
            json={"algorithm": "pagerank", "parameters": None},
            headers=JSON_CT,
        )
        assert result == {"scores": []}

    async def test_get_centrality_custom_algorithm(self) -> None:
        resp = _mock_response({"scores": [0.5]})
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        await client.get_centrality(algorithm="betweenness", parameters={"k": 10}, tenant_id=TENANT)

        call_json = mock.post.call_args[1]["json"]
        assert call_json["algorithm"] == "betweenness"
        assert call_json["parameters"] == {"k": 10}
        call_headers = mock.post.call_args[1]["headers"]
        assert call_headers == {**JSON_CT, "X-Tenant-Id": TENANT}

    # -- get_communities -----------------------------------------------------

    async def test_get_communities_defaults(self) -> None:
        resp = _mock_response({"communities": []})
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        result = await client.get_communities()

        call_json = mock.post.call_args[1]["json"]
        assert call_json == {
            "algorithm": "louvain",
            "min_community_size": 3,
            "parameters": None,
        }
        assert result == {"communities": []}

    async def test_get_communities_custom_params(self) -> None:
        resp = _mock_response({"communities": [{"id": 1}]})
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        await client.get_communities(
            algorithm="label_propagation",
            min_community_size=5,
            parameters={"resolution": 1.5},
            tenant_id=TENANT,
        )

        call_json = mock.post.call_args[1]["json"]
        assert call_json["algorithm"] == "label_propagation"
        assert call_json["min_community_size"] == 5
        assert call_json["parameters"] == {"resolution": 1.5}

    # -- get_paths -----------------------------------------------------------

    async def test_get_paths_url_and_body(self) -> None:
        resp = _mock_response({"paths": [["a", "b"]]})
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        result = await client.get_paths("src1", "tgt1", tenant_id=TENANT)

        mock.post.assert_called_once_with(
            "/analytics/path",
            json={
                "source_entity_id": "src1",
                "target_entity_id": "tgt1",
                "path_type": "shortest",
                "max_paths": 5,
                "min_confidence": 0.0,
            },
            headers={**JSON_CT, "X-Tenant-Id": TENANT},
        )
        assert result == {"paths": [["a", "b"]]}

    async def test_get_paths_custom_params(self) -> None:
        resp = _mock_response({"paths": []})
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        await client.get_paths("a", "b", path_type="all", max_paths=10, min_confidence=0.5)

        call_json = mock.post.call_args[1]["json"]
        assert call_json["path_type"] == "all"
        assert call_json["max_paths"] == 10
        assert call_json["min_confidence"] == 0.5

    # -- get_subgraph --------------------------------------------------------

    async def test_get_subgraph_url_and_body(self) -> None:
        resp = _mock_response({"results": [], "result_count": 0})
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        result = await client.get_subgraph("ent1", tenant_id=TENANT)

        mock.post.assert_called_once_with(
            "/nodes/subgraph",
            json={
                "entity_id": "ent1",
                "max_hops": 2,
                "max_nodes": 500,
                "relationship_types": [],
            },
            headers={**JSON_CT, "X-Tenant-Id": TENANT},
        )
        assert result == {"results": [], "result_count": 0}

    async def test_get_subgraph_custom_params(self) -> None:
        resp = _mock_response({"results": []})
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        await client.get_subgraph("ent1", max_hops=4, max_nodes=50, relationship_types=["SUPPORTS"], tenant_id=TENANT)

        call_json = mock.post.call_args[1]["json"]
        assert call_json["max_hops"] == 4
        assert call_json["max_nodes"] == 50
        assert call_json["relationship_types"] == ["SUPPORTS"]

    async def test_get_subgraph_none_relationship_types_sends_empty_list(self) -> None:
        resp = _mock_response({"results": []})
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        await client.get_subgraph("ent1", relationship_types=None)

        assert mock.post.call_args[1]["json"]["relationship_types"] == []

    # -- fulltext_search -----------------------------------------------------

    async def test_fulltext_search_url_and_body(self) -> None:
        resp = _mock_response({"results": [], "result_count": 0})
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        result = await client.fulltext_search("trust", min_score=0.5, limit=5, tenant_id=TENANT)

        mock.post.assert_called_once_with(
            "/nodes/fulltext-search",
            json={"query": "trust", "min_score": 0.5, "limit": 5},
            headers={**JSON_CT, "X-Tenant-Id": TENANT},
        )
        assert result == {"results": [], "result_count": 0}

    async def test_fulltext_search_defaults(self) -> None:
        resp = _mock_response({"results": []})
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        await client.fulltext_search("q")

        call_json = mock.post.call_args[1]["json"]
        assert call_json == {"query": "q", "min_score": 0.0, "limit": 10}

    # -- find_nodes_missing_property -----------------------------------------

    async def test_find_nodes_missing_property_url_and_body(self) -> None:
        resp = _mock_response({"results": [{"id": "n1"}], "result_count": 1})
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        result = await client.find_nodes_missing_property("integrity_score", limit=25, tenant_id=TENANT)

        mock.post.assert_called_once_with(
            "/nodes/missing-property",
            json={"property": "integrity_score", "limit": 25},
            headers={**JSON_CT, "X-Tenant-Id": TENANT},
        )
        assert result["results"] == [{"id": "n1"}]

    # -- update_node (PUT) ---------------------------------------------------

    async def test_update_node_uses_put(self) -> None:
        resp = _mock_response({"success": True, "node_id": "n1"})
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        result = await client.update_node("n1", {"integrity_score": 0.9}, tenant_id=TENANT)

        mock.put.assert_called_once_with(
            "/nodes/n1",
            json={"properties": {"integrity_score": 0.9}},
            headers={**JSON_CT, "X-Tenant-Id": TENANT},
        )
        mock.post.assert_not_called()
        assert result["success"] is True

    # -- health_check --------------------------------------------------------

    async def test_health_check(self) -> None:
        resp = _mock_response({"status": "healthy"})
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        result = await client.health_check()

        mock.get.assert_called_once_with("/health", headers=JSON_CT)
        assert result == {"status": "healthy"}

    async def test_health_check_raises_on_error(self) -> None:
        from clients.graph_adapter import GraphAdapterError

        resp = _mock_response({}, status_code=503)
        resp.text = "Service Unavailable"
        resp.raise_for_status.side_effect = httpx.HTTPStatusError("Unavailable", request=MagicMock(), response=resp)
        client = GraphAdapterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)
        mock.get.return_value = resp

        with pytest.raises(GraphAdapterError) as exc_info:
            await client.health_check()
        assert exc_info.value.status_code == 503

    # -- close ---------------------------------------------------------------

    async def test_close_closes_client(self) -> None:
        client = GraphAdapterClient(BASE_URL)
        mock = AsyncMock()
        mock.is_closed = False
        client._client = mock

        await client.close()

        mock.aclose.assert_called_once()
        assert client._client is None

    async def test_close_noop_when_no_client(self) -> None:
        client = GraphAdapterClient(BASE_URL)
        await client.close()  # should not raise


# ===========================================================================
# PyTorchBackendClient
# ===========================================================================


class TestPyTorchBackendClient:
    """Tests for PyTorchBackendClient."""

    def test_init_strips_trailing_slash(self) -> None:
        c = PyTorchBackendClient("http://pytorch:9000/")
        assert c.base_url == "http://pytorch:9000"

    def test_init_default_timeout(self) -> None:
        c = PyTorchBackendClient(BASE_URL)
        assert c.timeout == 120.0

    # -- get_embeddings ------------------------------------------------------

    async def test_get_embeddings_url_and_body(self) -> None:
        resp = _mock_response({"embeddings": [[0.1, 0.2]]})
        client = PyTorchBackendClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        result = await client.get_embeddings(["hello world"], tenant_id=TENANT)

        mock.post.assert_called_once_with(
            "/api/ml/embeddings",
            json={"texts": ["hello world"], "model": "default", "normalize": True},
            headers={**JSON_CT, "X-Tenant-Id": TENANT},
        )
        assert result == {"embeddings": [[0.1, 0.2]]}

    async def test_get_embeddings_custom_params(self) -> None:
        resp = _mock_response({"embeddings": []})
        client = PyTorchBackendClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        await client.get_embeddings(["a", "b"], model="scibert", normalize=False)

        call_json = mock.post.call_args[1]["json"]
        assert call_json["model"] == "scibert"
        assert call_json["normalize"] is False

    async def test_get_embeddings_no_tenant(self) -> None:
        resp = _mock_response({"embeddings": []})
        client = PyTorchBackendClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        await client.get_embeddings(["text"])

        call_headers = mock.post.call_args[1]["headers"]
        assert call_headers == JSON_CT

    # -- modality-gateway Bearer stamping (no-op until token set) -------------

    async def test_get_embeddings_token_stamps_bearer_and_delegates_a_proven_tenant(self) -> None:
        """Token set => X-Tenant-Id carries `Bearer <token>` for the gateway AND
        X-Effective-Tenant-Id delegates BYO/cost to the real customer tenant
        (svc-scientific-processor is a gateway delegating subject). Previously the
        tenant was silently dropped in gateway mode.

        WS-4 REWRITE — this test previously bound NO provenance (the exact shape a
        forged, uncredentialed X-Tenant-Id produces) and asserted the tenant WAS
        delegated. It asserted the defect verbatim, which is why CI stayed green
        over a cross-tenant credential-theft path. It is rewritten, not deleted:
        the property it was reaching for — the SA Bearer is never clobbered and a
        real customer tenant does reach the gateway — is kept, but the request now
        has to EARN the delegation. The unvouched half is pinned in
        tests/unit/test_ws4_vouched_delegation.py.
        """
        resp = _mock_response({"embeddings": []})
        client = PyTorchBackendClient(BASE_URL, downstream_license_token="lic-xyz")
        mock = _inject_mock_client(client, resp)

        token = tenant_source_var.set(SOURCE_LICENSE_TOKEN)
        try:
            await client.get_embeddings(["text"], tenant_id=TENANT)
        finally:
            tenant_source_var.reset(token)

        call_headers = mock.post.call_args[1]["headers"]
        assert call_headers == {
            **JSON_CT,
            "X-Tenant-Id": "Bearer lic-xyz",
            "X-Effective-Tenant-Id": TENANT,
        }

    async def test_get_embeddings_token_does_not_delegate_an_unvouched_tenant(self) -> None:
        """The forged-header case at this layer: SA Bearer still stamped, but the
        customer tenant is NOT named, so the gateway bills our own SA tenant."""
        resp = _mock_response({"embeddings": []})
        client = PyTorchBackendClient(BASE_URL, downstream_license_token="lic-xyz")
        mock = _inject_mock_client(client, resp)

        token = tenant_source_var.set(SOURCE_UNVOUCHED_HEADER)
        try:
            await client.get_embeddings(["text"], tenant_id=TENANT)
        finally:
            tenant_source_var.reset(token)

        call_headers = mock.post.call_args[1]["headers"]
        assert call_headers == {**JSON_CT, "X-Tenant-Id": "Bearer lic-xyz"}

    async def test_get_embeddings_token_does_not_delegate_bearer_tenant(self) -> None:
        """A Bearer-form tenant_id is a token, not a tenant → never delegated."""
        resp = _mock_response({"embeddings": []})
        client = PyTorchBackendClient(BASE_URL, downstream_license_token="lic-xyz")
        mock = _inject_mock_client(client, resp)

        await client.get_embeddings(["text"], tenant_id="Bearer other-tok")

        call_headers = mock.post.call_args[1]["headers"]
        assert call_headers == {**JSON_CT, "X-Tenant-Id": "Bearer lic-xyz"}

    async def test_get_embeddings_token_stamps_bearer_without_tenant(self) -> None:
        """Token set with no tenant_id still stamps Bearer (gateway introspects it)."""
        resp = _mock_response({"embeddings": []})
        client = PyTorchBackendClient(BASE_URL, downstream_license_token="lic-xyz")
        mock = _inject_mock_client(client, resp)

        await client.get_embeddings(["text"])

        call_headers = mock.post.call_args[1]["headers"]
        assert call_headers == {**JSON_CT, "X-Tenant-Id": "Bearer lic-xyz"}

    async def test_get_embeddings_unset_token_is_byte_identical(self) -> None:
        """Regression: token unset => raw X-Tenant-Id, byte-identical to before."""
        resp = _mock_response({"embeddings": []})
        client = PyTorchBackendClient(BASE_URL)  # no token (default None)
        mock = _inject_mock_client(client, resp)

        await client.get_embeddings(["text"], tenant_id=TENANT)

        call_headers = mock.post.call_args[1]["headers"]
        assert call_headers == {**JSON_CT, "X-Tenant-Id": TENANT}

    async def test_get_embeddings_empty_token_is_byte_identical(self) -> None:
        """Regression: empty-string token (config default) also behaves as unset."""
        resp = _mock_response({"embeddings": []})
        client = PyTorchBackendClient(BASE_URL, downstream_license_token="")
        mock = _inject_mock_client(client, resp)

        await client.get_embeddings(["text"], tenant_id=TENANT)

        call_headers = mock.post.call_args[1]["headers"]
        assert call_headers == {**JSON_CT, "X-Tenant-Id": TENANT}

    async def test_get_embeddings_unset_token_bearer_tenant_forwarded(self) -> None:
        """Token unset, caller passes a Bearer-form tenant_id => forwarded verbatim."""
        resp = _mock_response({"embeddings": []})
        client = PyTorchBackendClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        await client.get_embeddings(["text"], tenant_id="Bearer caller-tok")

        call_headers = mock.post.call_args[1]["headers"]
        assert call_headers == {**JSON_CT, "X-Tenant-Id": "Bearer caller-tok"}

    async def test_get_embeddings_raises_on_error(self) -> None:
        resp = _mock_response({}, status_code=500)
        resp.raise_for_status.side_effect = httpx.HTTPStatusError("Error", request=MagicMock(), response=resp)
        client = PyTorchBackendClient(BASE_URL)
        mock = _inject_mock_client(client, resp)
        mock.post.return_value = resp

        with pytest.raises(httpx.HTTPStatusError):
            await client.get_embeddings(["test"])

    # -- gnn_inference -------------------------------------------------------

    async def test_gnn_inference_url_and_body(self) -> None:
        resp = _mock_response({"predictions": [0, 1]})
        client = PyTorchBackendClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        features = [[1.0, 2.0], [3.0, 4.0]]
        edges = [[0, 1], [1, 0]]
        result = await client.gnn_inference(features, edges, tenant_id=TENANT)

        mock.post.assert_called_once_with(
            "/api/ml/gnn/inference",
            json={
                "node_features": [[1.0, 2.0], [3.0, 4.0]],
                "edge_index": [[0, 1], [1, 0]],
                "model": "default",
            },
            headers={**JSON_CT, "X-Tenant-Id": TENANT},
        )
        assert result == {"predictions": [0, 1]}

    async def test_gnn_inference_custom_model(self) -> None:
        resp = _mock_response({"predictions": []})
        client = PyTorchBackendClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        await client.gnn_inference([[1.0]], [[0]], model="gat_v2")

        call_json = mock.post.call_args[1]["json"]
        assert call_json["model"] == "gat_v2"

    async def test_gnn_inference_no_tenant(self) -> None:
        resp = _mock_response({"predictions": []})
        client = PyTorchBackendClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        await client.gnn_inference([[1.0]], [[0]])

        call_headers = mock.post.call_args[1]["headers"]
        assert call_headers == JSON_CT

    # -- health_check --------------------------------------------------------

    async def test_health_check(self) -> None:
        resp = _mock_response({"status": "healthy"})
        client = PyTorchBackendClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        result = await client.health_check()

        mock.get.assert_called_once_with("/health")
        assert result == {"status": "healthy"}

    # -- close ---------------------------------------------------------------

    async def test_close_closes_client(self) -> None:
        client = PyTorchBackendClient(BASE_URL)
        mock = AsyncMock()
        mock.is_closed = False
        client._client = mock

        await client.close()

        mock.aclose.assert_called_once()
        assert client._client is None


# ===========================================================================
# InferenceRouterClient
# ===========================================================================


class TestInferenceRouterClient:
    """Tests for InferenceRouterClient."""

    def test_init_strips_trailing_slash(self) -> None:
        c = InferenceRouterClient("http://router:8150/")
        assert c.base_url == "http://router:8150"

    def test_init_default_timeout(self) -> None:
        c = InferenceRouterClient(BASE_URL)
        assert c.timeout == 120.0

    # -- chat_completion -----------------------------------------------------

    async def test_chat_completion_url_and_body(self) -> None:
        resp = _mock_response({"choices": [{"message": {"content": "OK"}}]})
        client = InferenceRouterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        messages = [{"role": "user", "content": "Hello"}]
        result = await client.chat_completion(messages, tenant_id=TENANT)

        mock.post.assert_called_once_with(
            "/api/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.0,
                "max_tokens": 2048,
            },
            headers={**JSON_CT, "X-Tenant-Id": TENANT},
        )
        assert result["choices"][0]["message"]["content"] == "OK"

    async def test_chat_completion_custom_params(self) -> None:
        resp = _mock_response({"choices": []})
        client = InferenceRouterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        await client.chat_completion(
            [{"role": "system", "content": "Be precise"}],
            model="gpt-4",
            temperature=0.7,
            max_tokens=4096,
        )

        call_json = mock.post.call_args[1]["json"]
        assert call_json["model"] == "gpt-4"
        assert call_json["temperature"] == 0.7
        assert call_json["max_tokens"] == 4096

    async def test_chat_completion_no_tenant(self) -> None:
        resp = _mock_response({"choices": []})
        client = InferenceRouterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        await client.chat_completion([{"role": "user", "content": "Hi"}])

        call_headers = mock.post.call_args[1]["headers"]
        assert call_headers == JSON_CT

    async def test_chat_completion_raises_on_error(self) -> None:
        resp = _mock_response({}, status_code=429)
        resp.raise_for_status.side_effect = httpx.HTTPStatusError("Rate limited", request=MagicMock(), response=resp)
        client = InferenceRouterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)
        mock.post.return_value = resp

        with pytest.raises(httpx.HTTPStatusError):
            await client.chat_completion([{"role": "user", "content": "Hi"}])

    # -- health_check --------------------------------------------------------

    async def test_health_check(self) -> None:
        resp = _mock_response({"status": "healthy"})
        client = InferenceRouterClient(BASE_URL)
        mock = _inject_mock_client(client, resp)

        result = await client.health_check()

        mock.get.assert_called_once_with("/health")
        assert result == {"status": "healthy"}

    # -- close ---------------------------------------------------------------

    async def test_close_closes_client(self) -> None:
        client = InferenceRouterClient(BASE_URL)
        mock = AsyncMock()
        mock.is_closed = False
        client._client = mock

        await client.close()

        mock.aclose.assert_called_once()
        assert client._client is None


# ===========================================================================
# DiscoveryEngineClient
# ===========================================================================


class TestDiscoveryEngineClient:
    """Tests for DiscoveryEngineClient (stubbed service)."""

    def test_init_strips_trailing_slash(self) -> None:
        c = DiscoveryEngineClient("http://discovery:8000/")
        assert c.base_url == "http://discovery:8000"

    def test_init_default_timeout(self) -> None:
        c = DiscoveryEngineClient(BASE_URL)
        assert c.timeout == 60.0

    # -- search (stubbed) ----------------------------------------------------

    async def test_search_returns_empty(self) -> None:
        client = DiscoveryEngineClient(BASE_URL)
        result = await client.search("quantum entanglement")
        assert result == {"results": [], "total": 0}

    async def test_search_with_limit(self) -> None:
        client = DiscoveryEngineClient(BASE_URL)
        result = await client.search("test query", limit=5)
        assert result == {"results": [], "total": 0}

    async def test_search_with_tenant(self) -> None:
        client = DiscoveryEngineClient(BASE_URL)
        result = await client.search("test", tenant_id=TENANT)
        assert result == {"results": [], "total": 0}

    # -- get_related_claims (stubbed) ----------------------------------------

    async def test_get_related_claims_returns_empty(self) -> None:
        client = DiscoveryEngineClient(BASE_URL)
        result = await client.get_related_claims("claim-123")
        assert result == {"claims": [], "total": 0}

    async def test_get_related_claims_with_tenant(self) -> None:
        client = DiscoveryEngineClient(BASE_URL)
        result = await client.get_related_claims("claim-1", tenant_id=TENANT)
        assert result == {"claims": [], "total": 0}

    # -- health_check --------------------------------------------------------

    async def test_health_check_success(self) -> None:
        resp = _mock_response({"status": "healthy"})
        client = DiscoveryEngineClient(BASE_URL)
        _inject_mock_client(client, resp)

        result = await client.health_check()

        assert result == {"status": "healthy"}

    async def test_health_check_returns_unavailable_on_error(self) -> None:
        """DiscoveryEngine health_check catches exceptions and returns unavailable."""
        client = DiscoveryEngineClient(BASE_URL)
        mock = AsyncMock()
        mock.get.side_effect = httpx.ConnectError("Connection refused")
        mock.is_closed = False
        client._client = mock

        result = await client.health_check()

        assert result == {"status": "unavailable", "service": "discovery-engine"}

    async def test_health_check_returns_unavailable_on_http_error(self) -> None:
        """DiscoveryEngine health_check catches HTTP errors too."""
        resp = _mock_response({}, status_code=503)
        resp.raise_for_status.side_effect = httpx.HTTPStatusError("Unavailable", request=MagicMock(), response=resp)
        client = DiscoveryEngineClient(BASE_URL)
        mock = _inject_mock_client(client, resp)
        mock.get.return_value = resp

        result = await client.health_check()

        assert result == {"status": "unavailable", "service": "discovery-engine"}

    # -- close ---------------------------------------------------------------

    async def test_close_closes_client(self) -> None:
        client = DiscoveryEngineClient(BASE_URL)
        mock = AsyncMock()
        mock.is_closed = False
        client._client = mock

        await client.close()

        mock.aclose.assert_called_once()
        assert client._client is None
