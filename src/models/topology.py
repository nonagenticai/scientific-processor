"""Pydantic models for topology analysis endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# --- Subgraph ---


class SubgraphRequest(BaseModel):
    entity_id: str = Field(description="Central node ID for subgraph extraction")
    max_hops: int = Field(default=2, ge=1, le=5, description="Maximum traversal depth")
    max_nodes: int = Field(default=500, ge=1, le=5000, description="Maximum nodes to return")
    relationship_types: list[str] | None = Field(
        default=None, description="Filter by relationship types (e.g. SUPPORTS, REFUTES)"
    )
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum edge confidence")


class SubgraphResponse(BaseModel):
    entity_id: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    node_count: int
    edge_count: int
    max_hops_used: int
    computation_time_ms: float


# --- Contradictions ---


class ContradictionPair(BaseModel):
    target_id: str
    target_label: str
    supporting_nodes: list[dict[str, Any]]
    refuting_nodes: list[dict[str, Any]]
    net_support_score: float = Field(description="(supporting - refuting) / (supporting + refuting)")


class ContradictionRequest(BaseModel):
    entity_id: str = Field(description="Node ID to check for contradictions")
    max_hops: int = Field(default=2, ge=1, le=5)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ContradictionResponse(BaseModel):
    entity_id: str
    contradiction_pairs: list[ContradictionPair]
    total_contradictions: int
    computation_time_ms: float


# --- Authority ---


class AuthorityRequest(BaseModel):
    entity_id: str | None = Field(default=None, description="If set, scope centrality to subgraph around this node")
    algorithm: str = Field(default="pagerank", pattern="^(pagerank|eigenvector)$")
    max_hops: int = Field(default=2, ge=1, le=5)
    top_k: int = Field(default=20, ge=1, le=500)


class AuthorityResponse(BaseModel):
    algorithm: str
    scores: dict[str, float]
    top_entities: list[dict[str, Any]]
    entity_count: int
    computation_time_ms: float


# --- Communities ---


class CommunityRequest(BaseModel):
    algorithm: str = Field(default="louvain", pattern="^(louvain|leiden|label_propagation)$")
    min_community_size: int = Field(default=3, ge=1)
    parameters: dict[str, Any] | None = None


class CommunityResponse(BaseModel):
    algorithm: str
    entity_communities: dict[str, int]
    community_sizes: dict[int, int]
    number_of_communities: int
    modularity: float
    computation_time_ms: float
