"""Next-best-action policy.

The LLM does not choose the action. This module does, from the twelve-member
closed set, using rules plus the calibrated risk score plus evidence
availability. The generative layer is handed the result and asked to phrase it.

Each action carries an **evidence contract**: the structured records and policy
document types that must be present for the action to be legitimate. An action
whose contract is unmet is dropped with a recorded reason, which is why the
ledger can show what was considered and rejected, not just what was sent.
"""

from __future__ import annotations

from dataclasses import dataclass

from careweave.domain.enums import (
    PROACTIVE_ELIGIBLE_ACTIONS,
    ActionType,
    BarrierType,
    ClaimStatus,
    DocumentType,
    MemberIntent,
    ReasonCode,
    RiskBand,
)
from careweave.domain.models import (
    ActionDecision,
    ContextFrame,
    EvidenceChunk,
    MemberContext,
    RiskAssessment,
    SignalSet,
)

POLICY_VERSION = "nba-policy-v1"


@dataclass(frozen=True)
class EvidenceContract:
    """What must be true for an action to be permitted."""

    requires_open_authorization: bool = False
    requires_recent_rejection: bool = False
    requires_cost_history: bool = False
    requires_policy_evidence: bool = False
    required_doc_types: frozenset[DocumentType] = frozenset()
    #: minimum base score before other adjustments
    base_score: float = 0.5


CONTRACTS: dict[ActionType, EvidenceContract] = {
    ActionType.EXPLAIN_AUTH_STATUS: EvidenceContract(
        requires_open_authorization=True, requires_policy_evidence=True,
        required_doc_types=frozenset({DocumentType.PA_POLICY}), base_score=0.72),
    ActionType.REQUEST_MISSING_INFO: EvidenceContract(
        requires_open_authorization=True, requires_policy_evidence=True,
        required_doc_types=frozenset({DocumentType.PA_POLICY}), base_score=0.78),
    ActionType.EXPLAIN_CLAIM_STATUS: EvidenceContract(
        requires_recent_rejection=True, requires_policy_evidence=True,
        required_doc_types=frozenset({DocumentType.PA_POLICY, DocumentType.FAQ}),
        base_score=0.74),
    ActionType.EXPLAIN_COST_CHANGE: EvidenceContract(
        requires_cost_history=True, requires_policy_evidence=True,
        required_doc_types=frozenset({DocumentType.FORMULARY_POLICY,
                                      DocumentType.BENEFIT_SUMMARY, DocumentType.FAQ}),
        base_score=0.70),
    ActionType.EXPLAIN_COVERAGE: EvidenceContract(
        requires_policy_evidence=True,
        required_doc_types=frozenset({DocumentType.FORMULARY_POLICY,
                                      DocumentType.BENEFIT_SUMMARY}), base_score=0.62),
    ActionType.SURFACE_ALTERNATIVE_OPTION: EvidenceContract(
        requires_policy_evidence=True,
        required_doc_types=frozenset({DocumentType.FORMULARY_POLICY, DocumentType.FAQ}),
        base_score=0.58),
    ActionType.PROVIDE_NEXT_STEP: EvidenceContract(base_score=0.55),
    ActionType.SCHEDULE_FOLLOW_UP: EvidenceContract(base_score=0.45),
    ActionType.ROUTE_TO_ADVOCATE: EvidenceContract(base_score=0.60),
    ActionType.ROUTE_TO_PHARMACIST: EvidenceContract(base_score=0.90),
    ActionType.ESCALATE_CASE: EvidenceContract(base_score=0.66),
    ActionType.SUPPRESS_OUTREACH: EvidenceContract(base_score=0.40),
}


class ActionPolicy:
    def __init__(self) -> None:
        self.version = POLICY_VERSION

    def select(
        self,
        *,
        frame: ContextFrame,
        ctx: MemberContext,
        signals: SignalSet,
        risk: RiskAssessment,
        evidence: list[EvidenceChunk],
        is_proactive: bool = False,
    ) -> ActionDecision:
        rejected: dict[str, str] = {}
        scored: list[tuple[float, ActionType, list[str]]] = []

        available_types = {e.document_type for e in evidence}
        has_rejection = any(
            c.status == ClaimStatus.REJECTED for c in ctx.recent_claims
        )

        # Unresolved repeat contact is the one pattern where continuing to
        # explain is the wrong move. The transcripts are blunt about this: being
        # made to re-explain the same problem is the most frustrating thing a
        # service system does. A fourth explanation must not outscore handing
        # the case to a person who owns it.
        repeat_deadlock = (
            ReasonCode.REPEAT_CONTACT in risk.reasons
            and ctx.contact_count_30d >= 3
            and any(not i.resolved for i in ctx.recent_interactions)
        )

        for action in frame.candidate_next_steps:
            contract = CONTRACTS.get(action)
            if contract is None:
                rejected[action.value] = "no evidence contract defined"
                continue

            # Proactive outreach has a narrower permitted set. Filtering here
            # rather than letting the gate suppress afterwards means the policy
            # picks the best action it can actually *send*, instead of choosing
            # something undeliverable and producing a silent no-op.
            if is_proactive and action not in PROACTIVE_ELIGIBLE_ACTIONS:
                rejected[action.value] = "not eligible for unsolicited outreach"
                continue

            if contract.requires_open_authorization and not ctx.open_authorizations:
                rejected[action.value] = "no open authorization on record"
                continue
            if contract.requires_recent_rejection and not has_rejection:
                rejected[action.value] = "no recent claim rejection on record"
                continue
            if contract.requires_cost_history and len(ctx.cost_history) < 2:
                rejected[action.value] = "insufficient cost history to compare"
                continue
            if contract.requires_policy_evidence:
                if not evidence:
                    rejected[action.value] = "no policy evidence retrieved"
                    continue
                if contract.required_doc_types and not (
                    contract.required_doc_types & available_types
                ):
                    rejected[action.value] = (
                        "retrieved evidence does not include a required document type"
                    )
                    continue

            score = contract.base_score
            notes: list[str] = []

            # risk sharpens or softens, it does not decide
            if risk.band == RiskBand.HIGH:
                score += 0.10
                notes.append("elevated friction risk")
            elif risk.band == RiskBand.LOW and action != ActionType.SUPPRESS_OUTREACH:
                score -= 0.12
                notes.append("low friction risk")

            # a stated intent is stronger evidence than an inferred pattern
            if self._matches_intent(action, signals.intent):
                score += 0.15
                notes.append("matches the member's stated question")
            if signals.barrier == BarrierType.COST and action in (
                ActionType.EXPLAIN_COST_CHANGE, ActionType.SURFACE_ALTERNATIVE_OPTION
            ):
                score += 0.08
                notes.append("cost barrier detected in member's words")
            if signals.confidence < 0.4 and signals.source_document_ids:
                score -= 0.08
                notes.append("weak extraction confidence")

            if repeat_deadlock:
                if action in (ActionType.ESCALATE_CASE, ActionType.ROUTE_TO_ADVOCATE):
                    score += 0.30
                    notes.append("repeat unresolved contact: hand to a person")
                elif action.value.startswith("explain_"):
                    score -= 0.25
                    notes.append("explaining again would repeat a failed approach")

            if evidence:
                score += min(0.10, 0.12 * evidence[0].score)
                notes.append(f"top evidence score {evidence[0].score:.2f}")

            scored.append((min(0.99, max(0.0, score)), action, notes))

        if not scored:
            return ActionDecision(
                action=ActionType.SUPPRESS_OUTREACH,
                confidence=0.5,
                reasons=[ReasonCode.NO_ACTIONABLE_FRICTION, ReasonCode.EVIDENCE_INSUFFICIENT],
                supporting_evidence_ids=[],
                rejected_alternatives=rejected,
                policy_version=self.version,
            )

        scored.sort(key=lambda t: -t[0])
        best_score, best_action, best_notes = scored[0]
        for _, action, _ in scored[1:]:
            rejected.setdefault(action.value, "lower policy score")

        reasons = list(dict.fromkeys(risk.reasons))
        if signals.clinical_question_detected:
            reasons.insert(0, ReasonCode.CLINICAL_TOPIC_DETECTED)
        if best_score < 0.55:
            reasons.append(ReasonCode.LOW_MODEL_CONFIDENCE)

        contract = CONTRACTS[best_action]
        supporting = [
            e.chunk_id for e in evidence
            if not contract.required_doc_types or e.document_type in contract.required_doc_types
        ][:4]

        return ActionDecision(
            action=best_action,
            confidence=round(best_score, 3),
            reasons=reasons,
            supporting_evidence_ids=supporting,
            rejected_alternatives=rejected | {"__notes__": "; ".join(best_notes)},
            policy_version=self.version,
        )

    @staticmethod
    def _matches_intent(action: ActionType, intent: MemberIntent) -> bool:
        mapping = {
            MemberIntent.AUTHORIZATION_STATUS: {ActionType.EXPLAIN_AUTH_STATUS,
                                                ActionType.REQUEST_MISSING_INFO},
            MemberIntent.CLAIM_REJECTED_WHY: {ActionType.EXPLAIN_CLAIM_STATUS},
            MemberIntent.WHY_DID_COST_CHANGE: {ActionType.EXPLAIN_COST_CHANGE},
            MemberIntent.IS_THIS_COVERED: {ActionType.EXPLAIN_COVERAGE},
            MemberIntent.LOWER_COST_OPTION: {ActionType.SURFACE_ALTERNATIVE_OPTION},
            MemberIntent.HOME_DELIVERY: {ActionType.SURFACE_ALTERNATIVE_OPTION},
            MemberIntent.WHERE_IS_MY_PRESCRIPTION: {ActionType.EXPLAIN_AUTH_STATUS,
                                                    ActionType.EXPLAIN_CLAIM_STATUS,
                                                    ActionType.PROVIDE_NEXT_STEP},
            MemberIntent.WHAT_HAPPENS_NEXT: {ActionType.PROVIDE_NEXT_STEP},
            MemberIntent.CLINICAL_QUESTION: {ActionType.ROUTE_TO_PHARMACIST},
        }
        return action in mapping.get(intent, set())
