"""Typed domain models.

Design rule: these are the *only* shapes that cross module boundaries. The ML
layer, the RAG layer and the graph layer all speak these types, which is what
lets each be tested in isolation.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .enums import (
    ActionType,
    BarrierType,
    Channel,
    ClaimStatus,
    DeductiblePhase,
    DocumentType,
    DrugTier,
    EventType,
    GateVerdict,
    HumanDecisionType,
    MemberIntent,
    PAStatus,
    PharmacyType,
    ReasonCode,
    RejectCode,
    RiskBand,
    Sentiment,
    Urgency,
)


class Base(BaseModel):
    model_config = ConfigDict(use_enum_values=False, extra="forbid")


# --------------------------------------------------------------------------
# Reference / catalog entities
# --------------------------------------------------------------------------


class Drug(Base):
    drug_id: str
    name: str
    generic_name: str | None
    therapeutic_class: str
    is_specialty: bool
    generic_available: bool
    #: synthetic list price for a 30-day supply, USD
    list_price_30d: float


class Plan(Base):
    plan_id: str
    name: str
    deductible: float
    #: copay by tier when in initial coverage, USD
    copay_by_tier: dict[str, float]
    #: coinsurance rate applied to specialty tier
    specialty_coinsurance: float
    mail_order_discount: float
    out_of_network_penalty: float
    formulary_id: str


class FormularyEntry(Base):
    formulary_id: str
    drug_id: str
    tier: DrugTier
    pa_required: bool
    step_therapy_required: bool
    quantity_limit_30d: int | None
    covered: bool


class Pharmacy(Base):
    pharmacy_id: str
    name: str
    pharmacy_type: PharmacyType
    city: str
    state: str


class Prescriber(Base):
    prescriber_id: str
    name: str
    specialty: str
    #: synthetic latent trait: how promptly this prescriber returns PA paperwork
    responsiveness: float


# --------------------------------------------------------------------------
# Member and journey entities
# --------------------------------------------------------------------------


class Member(Base):
    member_id: str
    given_name: str
    family_name: str
    birth_date: date
    state: str
    plan_id: str
    enrolled_on: date
    preferred_channel: Channel
    #: caregiver flag -- transcript notes caring for a child/parent is context
    is_caregiver: bool = False

    @property
    def full_name(self) -> str:
        return f"{self.given_name} {self.family_name}"


class Prescription(Base):
    rx_id: str
    member_id: str
    drug_id: str
    prescriber_id: str
    pharmacy_id: str
    written_on: date
    days_supply: int
    quantity: int
    refills_authorized: int
    refills_used: int = 0
    is_new_to_therapy: bool = True


class PriorAuthorization(Base):
    pa_id: str
    rx_id: str
    member_id: str
    status: PAStatus
    requested_on: date | None
    decided_on: date | None
    #: set when status is PENDING_INFO
    missing_info: str | None = None
    denial_reason: str | None = None


class Claim(Base):
    claim_id: str
    rx_id: str
    member_id: str
    pharmacy_id: str
    submitted_on: date
    status: ClaimStatus
    reject_code: RejectCode | None = None
    member_cost: float | None = None
    plan_paid: float | None = None
    deductible_phase: DeductiblePhase | None = None


class Interaction(Base):
    interaction_id: str
    member_id: str
    occurred_on: datetime
    channel: Channel
    intent: MemberIntent
    resolved: bool
    escalated: bool = False
    #: link to the free-text artefact (transcript or message)
    document_id: str | None = None


class JourneyEvent(Base):
    """One node in the unified time-ordered journey.

    Every structured record projects into events. Downstream code reasons over
    the event stream, never over the raw tables -- that is what makes the
    ``as_of`` point-in-time guarantee enforceable in one place.
    """

    event_id: str
    member_id: str
    occurred_at: datetime
    event_type: EventType
    rx_id: str | None = None
    claim_id: str | None = None
    pa_id: str | None = None
    document_id: str | None = None
    #: small, typed payload; deliberately not a free-form blob
    payload: dict[str, Any] = Field(default_factory=dict)


class Document(Base):
    document_id: str
    document_type: DocumentType
    title: str
    body: str
    created_at: datetime
    #: null for policy corpus documents, set for member-specific artefacts
    member_id: str | None = None
    rx_id: str | None = None
    #: provenance for citation rendering
    source_label: str = "CareWeave synthetic corpus"


# --------------------------------------------------------------------------
# Assembled context
# --------------------------------------------------------------------------


class CostSnapshot(Base):
    fill_date: date
    member_cost: float
    deductible_phase: DeductiblePhase | None


class MemberContext(Base):
    """Everything the engine knows about a member *as of* a reference time."""

    member_id: str
    as_of: datetime
    member: Member
    plan: Plan
    active_prescriptions: list[Prescription]
    open_authorizations: list[PriorAuthorization]
    recent_claims: list[Claim]
    recent_interactions: list[Interaction]
    cost_history: list[CostSnapshot]
    days_since_last_contact: float | None
    contact_count_30d: int
    rejection_count_90d: int


class SignalSet(Base):
    """Output of the unstructured-data intelligence layer."""

    intent: MemberIntent = MemberIntent.OTHER
    barrier: BarrierType = BarrierType.NONE_DETECTED
    urgency: Urgency = Urgency.LOW
    sentiment: Sentiment = Sentiment.NEUTRAL
    mentioned_drugs: list[str] = Field(default_factory=list)
    mentions_cost: bool = False
    mentions_access: bool = False
    repeat_contact_language: bool = False
    clinical_question_detected: bool = False
    source_document_ids: list[str] = Field(default_factory=list)
    extractor_version: str = "unset"
    #: extractor's own confidence, used by the governance gate
    confidence: float = 0.0


class EvidenceChunk(Base):
    chunk_id: str
    document_id: str
    document_type: DocumentType
    title: str
    text: str
    score: float
    source_label: str


class RiskAssessment(Base):
    score: float
    band: RiskBand
    reasons: list[ReasonCode]
    model_version: str
    #: top contributing features with signed contributions
    contributions: dict[str, float] = Field(default_factory=dict)
    calibrated: bool = True


class ContextFrame(Base):
    """The five questions, as structured fields rather than prose.

    Making this typed (instead of letting an LLM free-write a paragraph) is what
    allows the action policy to consume it deterministically.
    """

    what_is_happening: str
    why_it_matters: str
    relevant_evidence_ids: list[str]
    what_could_happen_next: str
    candidate_next_steps: list[ActionType]
    #: facts the frame is built from, each traceable to a source
    grounded_facts: list[str] = Field(default_factory=list)


class ActionDecision(Base):
    action: ActionType
    confidence: float
    reasons: list[ReasonCode]
    #: evidence required by this action type and actually present
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    rejected_alternatives: dict[str, str] = Field(default_factory=dict)
    policy_version: str = "unset"


class GateResult(Base):
    verdict: GateVerdict
    reasons: list[ReasonCode]
    notes: list[str] = Field(default_factory=list)
    requires_human: bool = False
    gate_version: str = "unset"


class HumanDecision(Base):
    decision: HumanDecisionType
    reviewer_id: str
    decided_at: datetime
    rationale: str | None = None
    modified_action: ActionType | None = None
    modified_text: str | None = None


class GeneratedResponse(Base):
    text: str
    citations: list[str] = Field(default_factory=list)
    generator_version: str = "unset"
    grounded: bool = True
    ungrounded_spans: list[str] = Field(default_factory=list)


class TraceEntry(Base):
    node: str
    started_at: datetime
    duration_ms: float
    summary: str
    detail: dict[str, Any] = Field(default_factory=dict)


class LedgerRecord(Base):
    """Immutable audit record. One per terminal path, including no-action.

    Responsible AI as a queryable table rather than a README section.
    """

    ledger_id: str
    case_id: str
    member_id: str
    created_at: datetime
    trigger_event_type: EventType
    risk_score: float | None
    risk_band: RiskBand | None
    risk_reasons: list[ReasonCode]
    signals: SignalSet | None
    evidence_ids: list[str]
    selected_action: ActionType | None
    action_confidence: float | None
    gate_verdict: GateVerdict
    gate_reasons: list[ReasonCode]
    requires_human: bool
    human_decision: HumanDecision | None
    response_text: str | None
    citations: list[str]
    model_versions: dict[str, str]
    trace: list[TraceEntry]
    #: filled in later by the outcome loop
    outcome: str | None = None
