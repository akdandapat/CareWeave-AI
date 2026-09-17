"""Controlled vocabularies for CareWeave AI.

Every enum here is a *contract*. The LLM layer is never permitted to emit a value
outside these sets; if it tries, validation fails loudly rather than silently
introducing an unmodelled state. This is the single most important safety
property of the system.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """Python 3.11+ has enum.StrEnum but we keep our own for 3.10 compatibility."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# --------------------------------------------------------------------------
# Journey / event taxonomy
# --------------------------------------------------------------------------


class EventType(StrEnum):
    """The unified journey event stream.

    Every structured record in the system projects into one or more of these.
    The friction engine reasons over *sequences* of these, never over isolated
    rows -- that is the whole architectural point.
    """

    # prescription lifecycle
    RX_WRITTEN = "rx_written"
    RX_SUBMITTED_TO_PHARMACY = "rx_submitted_to_pharmacy"
    RX_FILLED = "rx_filled"
    RX_ABANDONED_AT_COUNTER = "rx_abandoned_at_counter"
    REFILL_DUE = "refill_due"
    REFILL_OVERDUE = "refill_overdue"

    # prior authorization lifecycle
    PA_REQUIRED_FLAGGED = "pa_required_flagged"
    PA_REQUESTED = "pa_requested"
    PA_INFO_REQUESTED = "pa_info_requested"
    PA_INFO_RECEIVED = "pa_info_received"
    PA_APPROVED = "pa_approved"
    PA_DENIED = "pa_denied"

    # claim lifecycle
    CLAIM_SUBMITTED = "claim_submitted"
    CLAIM_PAID = "claim_paid"
    CLAIM_REJECTED = "claim_rejected"
    CLAIM_REVERSED = "claim_reversed"

    # cost
    COST_CHANGED = "cost_changed"
    DEDUCTIBLE_PHASE_CHANGED = "deductible_phase_changed"
    FORMULARY_CHANGED = "formulary_changed"

    # member contact
    MEMBER_CONTACTED_SUPPORT = "member_contacted_support"
    MEMBER_SENT_MESSAGE = "member_sent_message"
    MEMBER_VIEWED_PORTAL = "member_viewed_portal"
    CASE_ESCALATED = "case_escalated"
    CASE_RESOLVED = "case_resolved"

    # documents / outreach
    LETTER_SENT = "letter_sent"
    PROACTIVE_OUTREACH_SENT = "proactive_outreach_sent"

    # pharmacy
    PHARMACY_CHANGED = "pharmacy_changed"


#: Event types that constitute *friction actually happening*. These are the
#: label ingredients for the ML model. They are deliberately observable
#: operational outcomes, not clinical outcomes.
FRICTION_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.MEMBER_CONTACTED_SUPPORT,
        EventType.CASE_ESCALATED,
        EventType.RX_ABANDONED_AT_COUNTER,
        EventType.CLAIM_REJECTED,
    }
)


# --------------------------------------------------------------------------
# Clinical / benefit vocabularies
# --------------------------------------------------------------------------


class DrugTier(StrEnum):
    GENERIC = "tier_1_generic"
    PREFERRED_BRAND = "tier_2_preferred_brand"
    NON_PREFERRED_BRAND = "tier_3_non_preferred_brand"
    SPECIALTY = "tier_4_specialty"


class PharmacyType(StrEnum):
    RETAIL_IN_NETWORK = "retail_in_network"
    RETAIL_OUT_OF_NETWORK = "retail_out_of_network"
    MAIL_ORDER = "mail_order"
    SPECIALTY = "specialty"


class PAStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    REQUIRED_NOT_STARTED = "required_not_started"
    PENDING = "pending"
    PENDING_INFO = "pending_additional_info"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


class ClaimStatus(StrEnum):
    SUBMITTED = "submitted"
    PAID = "paid"
    REJECTED = "rejected"
    REVERSED = "reversed"


class RejectCode(StrEnum):
    """Synthetic reject codes, loosely patterned on the *shape* of real
    pharmacy adjudication codes but not reproducing any real code set.

    Documented as synthetic in docs/data_dictionary.md.
    """

    PA_REQUIRED = "CW-075"
    REFILL_TOO_SOON = "CW-079"
    NOT_ON_FORMULARY = "CW-070"
    QUANTITY_LIMIT_EXCEEDED = "CW-076"
    STEP_THERAPY_REQUIRED = "CW-608"
    PHARMACY_OUT_OF_NETWORK = "CW-041"
    COVERAGE_TERMINATED = "CW-065"
    MISSING_PRESCRIBER_INFO = "CW-025"


#: Human-readable, member-safe explanation stems. The LLM personalises around
#: these; it does not invent the underlying reason.
REJECT_CODE_MEANING: dict[str, str] = {
    RejectCode.PA_REQUIRED: "this medication needs prior authorization before the plan can cover it",
    RejectCode.REFILL_TOO_SOON: "the refill was submitted earlier than the plan's refill schedule allows",
    RejectCode.NOT_ON_FORMULARY: "this medication is not on the plan's covered drug list",
    RejectCode.QUANTITY_LIMIT_EXCEEDED: "the quantity requested is above the plan's limit for this medication",
    RejectCode.STEP_THERAPY_REQUIRED: "the plan asks that another medication be tried first",
    RejectCode.PHARMACY_OUT_OF_NETWORK: "this pharmacy is not in the plan's network",
    RejectCode.COVERAGE_TERMINATED: "the plan shows coverage as inactive on the fill date",
    RejectCode.MISSING_PRESCRIBER_INFO: "required prescriber information was missing from the submission",
}


class DeductiblePhase(StrEnum):
    DEDUCTIBLE = "deductible"
    INITIAL_COVERAGE = "initial_coverage"
    CATASTROPHIC = "catastrophic"


# --------------------------------------------------------------------------
# Signals extracted from unstructured text
# --------------------------------------------------------------------------


class MemberIntent(StrEnum):
    WHERE_IS_MY_PRESCRIPTION = "where_is_my_prescription"
    WHY_DID_COST_CHANGE = "why_did_cost_change"
    IS_THIS_COVERED = "is_this_covered"
    AUTHORIZATION_STATUS = "authorization_status"
    CLAIM_REJECTED_WHY = "claim_rejected_why"
    LOWER_COST_OPTION = "lower_cost_option"
    HOME_DELIVERY = "home_delivery"
    EXPLAIN_LETTER = "explain_letter"
    WHAT_HAPPENS_NEXT = "what_happens_next"
    CLINICAL_QUESTION = "clinical_question"
    OTHER = "other"


class BarrierType(StrEnum):
    """Why a member may not be getting their medication.

    Directly derived from the transcript point that non-adherence is not one
    thing: cost, side effects, access, confusion, or life circumstances.
    Modelling the *barrier* is what separates relevant outreach from spam.
    """

    COST = "cost"
    ACCESS = "access"
    CONFUSION = "confusion"
    ADMINISTRATIVE = "administrative"
    CLINICAL_CONCERN = "clinical_concern"
    NONE_DETECTED = "none_detected"


class Urgency(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Sentiment(StrEnum):
    NEUTRAL = "neutral"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    DISTRESSED = "distressed"


# --------------------------------------------------------------------------
# Decision layer
# --------------------------------------------------------------------------


class ActionType(StrEnum):
    """THE CLOSED ACTION SPACE.

    The generative layer receives an already-selected action and narrates it.
    It cannot select, substitute, or invent one. Any output that implies an
    action outside this set is a validation failure, caught by
    ``graph.nodes.verify_response`` and by the test suite.
    """

    EXPLAIN_COVERAGE = "explain_coverage"
    EXPLAIN_CLAIM_STATUS = "explain_claim_status"
    EXPLAIN_AUTH_STATUS = "explain_auth_status"
    EXPLAIN_COST_CHANGE = "explain_cost_change"
    SURFACE_ALTERNATIVE_OPTION = "surface_alternative_option"
    REQUEST_MISSING_INFO = "request_missing_info"
    PROVIDE_NEXT_STEP = "provide_next_step"
    ROUTE_TO_ADVOCATE = "route_to_advocate"
    ROUTE_TO_PHARMACIST = "route_to_pharmacist"
    ESCALATE_CASE = "escalate_case"
    SUPPRESS_OUTREACH = "suppress_outreach"
    SCHEDULE_FOLLOW_UP = "schedule_follow_up"


#: Actions that may never be executed without a human decision, regardless of
#: model confidence. Enforced deterministically in the governance gate.
HUMAN_REQUIRED_ACTIONS: frozenset[ActionType] = frozenset(
    {
        ActionType.ROUTE_TO_PHARMACIST,
        ActionType.ESCALATE_CASE,
    }
)


#: Actions that are permitted to be delivered as *unsolicited* proactive
#: outreach. Everything else is reactive-only.
PROACTIVE_ELIGIBLE_ACTIONS: frozenset[ActionType] = frozenset(
    {
        ActionType.EXPLAIN_AUTH_STATUS,
        ActionType.EXPLAIN_COST_CHANGE,
        ActionType.SURFACE_ALTERNATIVE_OPTION,
        ActionType.REQUEST_MISSING_INFO,
        ActionType.PROVIDE_NEXT_STEP,
        ActionType.SCHEDULE_FOLLOW_UP,
    }
)


class RiskBand(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GateVerdict(StrEnum):
    ALLOW = "allow"
    HOLD = "hold"
    SUPPRESS = "suppress"
    ESCALATE = "escalate"


class HumanDecisionType(StrEnum):
    ACCEPT = "accept"
    MODIFY = "modify"
    REJECT = "reject"


class ReasonCode(StrEnum):
    """Machine-readable justifications attached to every decision.

    These are what make the ledger auditable without re-running the model.
    """

    PA_PENDING_PAST_SLA = "pa_pending_past_sla"
    PA_PENDING_INFO = "pa_pending_additional_info"
    REFILL_RUNWAY_SHORT = "refill_runway_short"
    RECENT_CLAIM_REJECTION = "recent_claim_rejection"
    REPEAT_CONTACT = "repeat_contact_in_window"
    COST_DELTA_LARGE = "cost_delta_large"
    DEDUCTIBLE_PHASE_TRANSITION = "deductible_phase_transition"
    SPECIALTY_DRUG = "specialty_drug"
    NEW_TO_THERAPY = "new_to_therapy"
    OUT_OF_NETWORK_PHARMACY = "out_of_network_pharmacy"
    PRIOR_ABANDONMENT = "prior_abandonment_history"
    LOW_MODEL_CONFIDENCE = "low_model_confidence"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    EVIDENCE_CONFLICT = "evidence_conflict"
    CLINICAL_TOPIC_DETECTED = "clinical_topic_detected"
    SUPPRESSION_WINDOW_ACTIVE = "suppression_window_active"
    FREQUENCY_CAP_REACHED = "frequency_cap_reached"
    DISTRESS_DETECTED = "distress_detected"
    NO_ACTIONABLE_FRICTION = "no_actionable_friction"


class Channel(StrEnum):
    PORTAL = "portal"
    APP_PUSH = "app_push"
    SMS = "sms"
    EMAIL = "email"
    PHONE = "phone"
    MAIL = "mail"


class DocumentType(StrEnum):
    BENEFIT_SUMMARY = "benefit_summary"
    FORMULARY_POLICY = "formulary_policy"
    PA_POLICY = "pa_policy"
    WORKFLOW_SOP = "workflow_sop"
    MEMBER_LETTER = "member_letter"
    CLAIM_EXPLANATION = "claim_explanation"
    PHARMACY_NOTE = "pharmacy_note"
    ADVOCATE_NOTE = "advocate_note"
    CALL_TRANSCRIPT = "call_transcript"
    MEMBER_MESSAGE = "member_message"
    FAQ = "faq"


#: Documents that form the retrievable *policy* corpus (the RAG knowledge base).
#: Member-specific artefacts are retrieved through the structured store instead,
#: so that policy grounding and personal context stay separable and auditable.
POLICY_DOCUMENT_TYPES: frozenset[DocumentType] = frozenset(
    {
        DocumentType.BENEFIT_SUMMARY,
        DocumentType.FORMULARY_POLICY,
        DocumentType.PA_POLICY,
        DocumentType.WORKFLOW_SOP,
        DocumentType.FAQ,
    }
)
