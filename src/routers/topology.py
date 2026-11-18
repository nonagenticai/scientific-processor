"""
Topology analysis endpoints — subgraph extraction, contradiction detection,
authority scoring, and community detection.
"""

import httpx
import networkx
from fastapi import APIRouter, Depends, HTTPException, Request

from auth.dependencies import resolve_tenant_id
from clients.graph_adapter import GraphAdapterError, GraphServiceUnavailableError
from dependencies import (
    get_centrality_service,
    get_community_service,
    get_contradiction_service,
    get_subgraph_service,
)
from log_config import get_logger
from models.topology import (
    AuthorityRequest,
    AuthorityResponse,
    CommunityRequest,
    CommunityResponse,
    ContradictionRequest,
    ContradictionResponse,
    SubgraphRequest,
    SubgraphResponse,
)
from services.centrality_service import CentralityService
from services.community_service import CommunityService
from services.contradiction_service import ContradictionService
from services.subgraph_service import SubgraphService

logger = get_logger(__name__)

router = APIRouter()


@router.post("/subgraph", response_model=SubgraphResponse, operation_id="topology_extract_subgraph")
async def extract_subgraph(
    body: SubgraphRequest,
    request: Request,
    service: SubgraphService = Depends(get_subgraph_service),
) -> SubgraphResponse:
    """Extract a claim-centric subgraph from the knowledge graph."""
    try:
        data = await service.extract_subgraph(
            entity_id=body.entity_id,
            max_hops=body.max_hops,
            max_nodes=body.max_nodes,
            relationship_types=body.relationship_types,
            min_confidence=body.min_confidence,
            tenant_id=resolve_tenant_id(request),
        )
        return data.response
    except GraphServiceUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except GraphAdapterError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except httpx.HTTPError, KeyError, ValueError, networkx.NetworkXError:
        logger.exception("Subgraph extraction failed")
        raise HTTPException(status_code=500, detail="Subgraph extraction failed")


@router.post("/contradictions", response_model=ContradictionResponse, operation_id="topology_detect_contradictions")
async def detect_contradictions(
    body: ContradictionRequest,
    request: Request,
    service: ContradictionService = Depends(get_contradiction_service),
) -> ContradictionResponse:
    """Detect contradictions within a subgraph or between claims."""
    try:
        return await service.detect_contradictions(
            entity_id=body.entity_id,
            max_hops=body.max_hops,
            min_confidence=body.min_confidence,
            tenant_id=resolve_tenant_id(request),
        )
    except GraphServiceUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except GraphAdapterError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except httpx.HTTPError, KeyError, ValueError, networkx.NetworkXError:
        logger.exception("Contradiction detection failed")
        raise HTTPException(status_code=500, detail="Contradiction detection failed")


@router.post("/authority", response_model=AuthorityResponse, operation_id="topology_compute_authority")
async def compute_authority(
    body: AuthorityRequest,
    request: Request,
    service: CentralityService = Depends(get_centrality_service),
) -> AuthorityResponse:
    """Compute authority scores for nodes using PageRank/eigenvector centrality."""
    try:
        return await service.compute_authority(
            entity_id=body.entity_id,
            algorithm=body.algorithm,
            max_hops=body.max_hops,
            top_k=body.top_k,
            tenant_id=resolve_tenant_id(request),
        )
    except GraphServiceUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except GraphAdapterError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except httpx.HTTPError, KeyError, ValueError, networkx.NetworkXError:
        logger.exception("Authority computation failed")
        raise HTTPException(status_code=500, detail="Authority computation failed")


@router.post("/communities", response_model=CommunityResponse, operation_id="topology_detect_communities")
async def detect_communities(
    body: CommunityRequest,
    request: Request,
    service: CommunityService = Depends(get_community_service),
) -> CommunityResponse:
    """Detect communities within the knowledge graph."""
    try:
        return await service.detect_communities(
            algorithm=body.algorithm,
            min_community_size=body.min_community_size,
            parameters=body.parameters,
            tenant_id=resolve_tenant_id(request),
        )
    except GraphServiceUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except GraphAdapterError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except httpx.HTTPError, KeyError, ValueError, networkx.NetworkXError:
        logger.exception("Community detection failed")
        raise HTTPException(status_code=500, detail="Community detection failed")
