"""Governance gate.

The last checkpoint before anything reaches a member. Four properties define it:

* **Fully deterministic.** No model call, no probability. Given the same inputs
  it returns the same verdict, forever. A safety control that is itself
  stochastic is not a control.
* **It can only restrict.** The gate can downgrade ALLOW to HOLD, SUPPRESS or
  ESCALATE. It has no path to upgrade anything. A bug here fails closed.
* **It runs on every path**, including the ones where the engine decided to do
  nothing. Silence is a governed decision too.
* **Every verdict carries reason codes**, so an auditor can ask "why did this
  member get this message on this day" and get an answer without re-running
  anything.

Rules are ordered by severity. The first matching restrictive rule wins, so a
clinical-topic escalation is never overridden by a frequency cap further down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from careweave.config import SETTINGS
from careweave.domain.enums import (
    HUMAN_REQUIRED_ACTIONS,
    PROACTIVE_ELIGIBLE_ACTIONS,
    ActionType,
    GateVerdict,
    ReasonCode,
    RiskBand,
    Sentiment,
)
from careweave.domain.models import (
    ActionDecision,
    GateResult,
    MemberContext,
    RiskAssessment,
    SignalSet,
)


@dataclass
class OutreachLedger:
    """Suppression state: what has already been sent to whom, and when.

    In production this is a table. Here it is an in-process structure with the
    same interface, so the suppression logic is exercised identically.
    """

    sent: dict[str, list[tuple[datetime, ActionType]]] = field(default_factory=dict)

    def record(self, member_id: str, when: datetime, action: ActionType) -> None:
        self.sent.setdefault(member_id, []).append((when, action))

    def count_in_window(self, member_id: str, as_of: datetime, days: int) -> int:
        floor = as_of - timedelta(days=days)
        return sum(1 for t, _ in self.sent.get(member_id, []) if t >= floor)

    def topic_sent_recently(
        self, member_id: str, as_of: datetime, action: ActionType, days: int
    ) -> bool:
        floor = as_of - timedelta(days=days)
        return any(
            t >= floor and a == action for t, a in self.sent.get(member_id, [])
        )


class GovernanceGate:
    def __init__(self, ledger: OutreachLedger | None = None) -> None:
        self.ledger = ledger or OutreachLedger()
        self.cfg = SETTINGS.governance
        self.version = self.cfg.gate_version

    def evaluate(
        self,
        *,
        decision: ActionDecision,
        ctx: MemberContext,
        signals: SignalSet,
        risk: RiskAssessment,
        evidence_sufficient: bool,
        evidence_conflict: bool,
        as_of: datetime,
        is_proactive: bool,
    ) -> GateResult:
        reasons: list[ReasonCode] = []
        notes: list[str] = []

        def result(verdict: GateVerdict, human: bool) -> GateResult:
            return GateResult(
                verdict=verdict, reasons=list(dict.fromkeys(reasons)),
                notes=notes, requires_human=human, gate_version=self.version,
            )

        # -- 1. clinical topics never receive an automated answer ----------
        if signals.clinical_question_detected:
            reasons.append(ReasonCode.CLINICAL_TOPIC_DETECTED)
            notes.append(
                "A clinical question was detected. Automated response is blocked "
                "regardless of confidence; routed to a pharmacist."
            )
            return result(GateVerdict.ESCALATE, human=True)

        # -- 2. member distress goes to a person ---------------------------
        if signals.sentiment == Sentiment.DISTRESSED:
            reasons.append(ReasonCode.DISTRESS_DETECTED)
            notes.append("Distress language detected; handled by a person, not a message.")
            return result(GateVerdict.ESCALATE, human=True)

        # -- 3. actions that structurally require a human ------------------
        if decision.action in HUMAN_REQUIRED_ACTIONS:
            notes.append(
                f"'{decision.action.value}' always requires human authorisation."
            )
            return result(GateVerdict.ESCALATE, human=True)

        # -- 4. evidence integrity ------------------------------------------
        if evidence_conflict:
            reasons.append(ReasonCode.EVIDENCE_CONFLICT)
            notes.append("Retrieved evidence is internally inconsistent; held for review.")
            return result(GateVerdict.HOLD, human=True)
        if not evidence_sufficient and decision.action != ActionType.SUPPRESS_OUTREACH:
            reasons.append(ReasonCode.EVIDENCE_INSUFFICIENT)
            notes.append("Insufficient grounded evidence to make an assertion.")
            return result(GateVerdict.HOLD, human=True)

        # -- 5. confidence floor ---------------------------------------------
        if decision.confidence < self.cfg.min_action_confidence:
            reasons.append(ReasonCode.LOW_MODEL_CONFIDENCE)
            notes.append(
                f"Action confidence {decision.confidence:.2f} is below the "
                f"{self.cfg.min_action_confidence:.2f} floor."
            )
            return result(GateVerdict.HOLD, human=True)

        # -- 6. explicit no-action -------------------------------------------
        if decision.action == ActionType.SUPPRESS_OUTREACH:
            reasons.append(ReasonCode.NO_ACTIONABLE_FRICTION)
            notes.append("No actionable friction; deliberately taking no action.")
            return result(GateVerdict.SUPPRESS, human=False)

        # -- 7. proactive-only constraints ------------------------------------
        if is_proactive:
            if decision.action not in PROACTIVE_ELIGIBLE_ACTIONS:
                notes.append(
                    f"'{decision.action.value}' is reactive-only and may not be sent "
                    "unsolicited."
                )
                return result(GateVerdict.SUPPRESS, human=False)
            if risk.band == RiskBand.LOW:
                reasons.append(ReasonCode.NO_ACTIONABLE_FRICTION)
                notes.append("Risk band is low; not worth interrupting the member.")
                return result(GateVerdict.SUPPRESS, human=False)
            if any(not i.resolved for i in ctx.recent_interactions[:1]):
                notes.append(
                    "An unresolved case is already open with an advocate. Contact from "
                    "two directions increases confusion rather than reducing it."
                )
                return result(GateVerdict.SUPPRESS, human=False)
            if self.ledger.topic_sent_recently(
                ctx.member_id, as_of, decision.action, self.cfg.topic_suppression_days
            ):
                reasons.append(ReasonCode.SUPPRESSION_WINDOW_ACTIVE)
                notes.append(
                    f"Same topic already sent within {self.cfg.topic_suppression_days} days."
                )
                return result(GateVerdict.SUPPRESS, human=False)
            if (
                self.ledger.count_in_window(ctx.member_id, as_of, 30)
                >= self.cfg.frequency_cap_30d
            ):
                reasons.append(ReasonCode.FREQUENCY_CAP_REACHED)
                notes.append(
                    f"Frequency cap of {self.cfg.frequency_cap_30d} per 30 days reached."
                )
                return result(GateVerdict.SUPPRESS, human=False)

        notes.append("All governance checks passed.")
        return result(GateVerdict.ALLOW, human=False)


#: Fields that may be referenced in member-facing generated text. Anything not
#: on this list is withheld -- the "understood, not watched" line, enforced as an
#: allowlist rather than trusted to a prompt instruction.
MEMBER_FACING_FIELD_ALLOWLIST: frozenset[str] = frozenset(
    {
        "given_name", "drug_name", "pharmacy_name", "member_cost", "previous_cost",
        "fill_date", "days_supply", "authorization_status", "authorization_age_days",
        "missing_info", "reject_reason", "plan_name", "tier", "deductible_phase",
        "days_of_supply_remaining",
    }
)

#: Never referenced in member-facing text, even though the engine uses them.
MEMBER_FACING_FIELD_DENYLIST: frozenset[str] = frozenset(
    {
        "risk_score", "risk_band", "contact_propensity", "health_literacy",
        "prior_escalation_count", "model_version", "abandonment_history",
        "unresolved_contact_count", "birth_date", "is_caregiver",
    }
)
