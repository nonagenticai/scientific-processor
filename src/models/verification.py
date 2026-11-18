"""Pydantic models for verification endpoints — CoVe, RAGAS, and integrity scoring."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# --- Integrity (existing, used by integrity_service.py) ---

#: A real, measured 0.0-1.0 score. `| None` on a field of this type means the
#: quantity could NOT be measured — it is never a low score.
UnitScore = Annotated[float, Field(ge=0.0, le=1.0)]


#: Why a component holds no number. `measured` is the only status carrying a score.
#: The other two are NOT interchangeable: `unavailable` means we could not look
#: (backend outage / failed computation) and makes the composite unknowable;
#: `not_applicable` means we looked and the metric has no denominator for this
#: entity (no incoming edges at all), which is itself a finding — the composite is
#: renormalized over the applicable components instead of being thrown away.
ComponentStatus = Literal["measured", "not_applicable", "unavailable"]


class IntegrityComponent(BaseModel):
    name: str
    score: UnitScore | None = Field(
        default=None,
        description=(
            "None when this component holds no number. A null score is NOT a low "
            "score — read `status` to learn whether it was unmeasurable (backend "
            "outage) or inapplicable (nothing to measure)."
        ),
    )
    status: ComponentStatus = Field(
        default="measured",
        description=(
            "'measured' — `score` is a real value. 'not_applicable' — the metric is "
            "undefined for this entity (e.g. no incoming edges), so its weight is "
            "redistributed over the applicable components. 'unavailable' — the "
            "measurement failed, so the composite is null and is never persisted."
        ),
    )
    weight: float = Field(description="Configured weight, as supplied by settings. Never rescaled.")
    weighted_score: float | None = Field(
        default=None,
        description="None when `score` is None — there is nothing to weight.",
    )
    effective_weight: float = Field(
        default=0.0,
        description=(
            "The weight this component actually contributed. Equals `weight` when "
            "nothing was inapplicable; 0.0 for a component that holds no number; "
            "larger than `weight` when it absorbed an inapplicable component's share. "
            "Present so a renormalized composite stays auditable from the response alone."
        ),
    )


class IntegrityRequest(BaseModel):
    entity_id: str = Field(description="Node ID to compute integrity score for")
    max_hops: int = Field(default=2, ge=1, le=5)
    include_topology: bool = True
    include_cove: bool = False
    include_ragas: bool = False
    write_back: bool = Field(default=False, description="Write integrity score back to the knowledge graph node")


class IntegrityResponse(BaseModel):
    # Nullable for the same reason RagasResponse's metrics are (#36): when a
    # component cannot be measured, the composite is unknown. Fabricating a
    # plausible mid-range number here was worse than the RAGAS case, because a
    # written-back integrity_score lands on the graph node and in the verification
    # history, where it is indistinguishable from a real measurement forever.
    entity_id: str
    integrity_score: UnitScore | None = Field(
        default=None,
        description=(
            "None when at least one component could not be MEASURED, or when no "
            "positive component was applicable at all. A null score is NOT a low "
            "score and is never written back to the graph. An entity that merely has "
            "nothing to cite still receives a real score — see not_applicable_components."
        ),
    )
    components: list[IntegrityComponent]
    details: dict[str, Any] = Field(default_factory=dict)
    computation_time_ms: float
    degraded: bool = Field(
        default=False,
        description=(
            "True exactly when `integrity_score` is null — either a component could "
            "not be measured, or no positive component was applicable at all. Read "
            "`degraded_reason` for which. An entity with nothing to cite is NOT degraded."
        ),
    )
    degraded_components: list[str] = Field(
        default_factory=list,
        description=(
            "Names of the components that could not be measured (status 'unavailable'). "
            "May be empty on a degraded response when the cause was that nothing was "
            "applicable rather than that something failed."
        ),
    )
    degraded_reason: str | None = Field(
        default=None,
        description="Human-readable explanation of why the composite could not be computed.",
    )
    not_applicable_components: list[str] = Field(
        default_factory=list,
        description=(
            "Names of the components whose metric is undefined for this entity "
            "(status 'not_applicable'). NOT a degradation: the composite is "
            "renormalized over the remaining components and IS persisted. Populated "
            "even on a degraded response, so the two reasons stay distinguishable."
        ),
    )


# --- Chain-of-Verification (CoVe) ---


class CoVeRequest(BaseModel):
    claim_text: str = Field(..., max_length=5000)
    claim_node_id: str | None = None
    num_questions: int = Field(default=5, ge=1, le=10)
    source_authority_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class VerificationQuestion(BaseModel):
    question: str
    answer: str | None = None
    source_node_ids: list[str] = Field(default_factory=list)
    source_authority_score: float = 0.0
    verdict: Literal["supports", "refutes", "inconclusive"] = "inconclusive"


class CoVeResponse(BaseModel):
    claim_text: str
    questions: list[VerificationQuestion]
    overall_verdict: Literal["verified", "disputed", "inconclusive"]
    confidence: float
    supporting_evidence: int
    refuting_evidence: int
    computation_time_ms: float
    degraded: bool = Field(
        default=False,
        description="True when an LLM call (question generation or evaluation) failed, so the verdict rests on fallback questions / forced 'inconclusive' rather than a full chain-of-verification.",
    )


# --- RAGAS Faithfulness ---


class RagasRequest(BaseModel):
    question: str = Field(..., max_length=2000)
    answer: str = Field(..., max_length=10000)
    contexts: list[str] = Field(..., min_length=1)


class RagasResponse(BaseModel):
    # Nullable: a metric is `null` when its backend (LLM judge / embedding
    # service) was unreachable, so callers can distinguish a real low score
    # from an outage instead of seeing a fabricated 0.5. See `degraded`.
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    factual_accuracy: float | None = None
    computation_time_ms: float
    degraded: bool = Field(
        default=False,
        description="True when one or more metrics could not be computed (backend outage / malformed response). Null metrics are NOT real scores.",
    )
