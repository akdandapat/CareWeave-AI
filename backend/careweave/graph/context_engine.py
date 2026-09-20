"""Context engine.

Sits between raw data and any generative output, and answers five questions in
*typed fields* rather than prose:

    what is happening / why it matters / what evidence is relevant /
    what could happen next / what should happen now

Deliberately deterministic. Every statement it produces is derived from a
structured record, and each carries the record it came from. That is what makes
the downstream generation checkable: if a sentence in the member's answer is not
traceable to a grounded fact here or to a retrieved policy chunk, the verifier
flags it.

Letting an LLM write this frame would be easier and would destroy the property.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from careweave.data.store import JourneyStore, _as_dt
from careweave.domain.enums import (
    REJECT_CODE_MEANING,
    ActionType,
    BarrierType,
    ClaimStatus,
    DeductiblePhase,
    EventType,
    MemberIntent,
    PAStatus,
    ReasonCode,
    RejectCode,
    RiskBand,
)
from careweave.domain.models import (
    ContextFrame,
    MemberContext,
    RiskAssessment,
    SignalSet,
)

#: Standard review window quoted in the PA policy document. Kept here as a named
#: constant so the number in generated text and the number in the policy corpus
#: cannot drift apart silently.
PA_STANDARD_REVIEW_BUSINESS_DAYS = 5


def _days(n: int) -> str:
    """'1 day' not '1 days'. Small, but a member reads this."""
    n = max(0, int(n))
    return "1 day" if n == 1 else f"{n} days"
INTERIM_SUPPLY_TRIGGER_DAYS = 7


@dataclass(frozen=True)
class GroundedFact:
    """A statement plus the record it came from. Never free-standing prose."""

    text: str
    source_kind: str  # "authorization" | "claim" | "cost" | "interaction" | "prescription"
    source_id: str

    def render(self) -> str:
        return f"{self.text} [{self.source_kind}:{self.source_id}]"


class ContextEngine:
    def __init__(self, store: JourneyStore) -> None:
        self.store = store

    # -- fact extraction -------------------------------------------------

    def collect_facts(
        self, ctx: MemberContext, as_of: datetime
    ) -> tuple[list[GroundedFact], list[ReasonCode]]:
        facts: list[GroundedFact] = []
        reasons: list[ReasonCode] = []

        # -- open authorizations -----------------------------------------
        for pa in ctx.open_authorizations:
            if pa.requested_on is None:
                continue
            age = (as_of - _as_dt(pa.requested_on)).days
            drug = self.store.drug_for_rx(pa.rx_id)
            name = drug.name if drug else "a medication"
            if pa.status == PAStatus.PENDING_INFO:
                facts.append(GroundedFact(
                    f"A coverage review for {name} has been open for {_days(age)} and is "
                    f"waiting on {pa.missing_info or 'additional documentation'} from the "
                    "prescriber",
                    "authorization", pa.pa_id))
                reasons.append(ReasonCode.PA_PENDING_INFO)
            else:
                facts.append(GroundedFact(
                    f"A coverage review for {name} has been open for {_days(age)}",
                    "authorization", pa.pa_id))
            if age > PA_STANDARD_REVIEW_BUSINESS_DAYS * 7 / 5:
                reasons.append(ReasonCode.PA_PENDING_PAST_SLA)

        # -- recent rejections -------------------------------------------
        rejected = [c for c in ctx.recent_claims if c.status == ClaimStatus.REJECTED]
        if rejected:
            latest = max(rejected, key=lambda c: c.submitted_on)
            days = (as_of - _as_dt(latest.submitted_on)).days
            drug = self.store.drug_for_rx(latest.rx_id)
            name = drug.name if drug else "a medication"
            meaning = (
                REJECT_CODE_MEANING.get(latest.reject_code, "the claim did not process")
                if latest.reject_code else "the claim did not process"
            )
            facts.append(GroundedFact(
                f"A pharmacy claim for {name} did not process {_days(days)} ago because "
                f"{meaning}", "claim", latest.claim_id))
            reasons.append(ReasonCode.RECENT_CLAIM_REJECTION)
            if latest.reject_code == RejectCode.PHARMACY_OUT_OF_NETWORK:
                reasons.append(ReasonCode.OUT_OF_NETWORK_PHARMACY)

        # -- cost trajectory ---------------------------------------------
        if len(ctx.cost_history) >= 2:
            prev, curr = ctx.cost_history[-2], ctx.cost_history[-1]
            delta = curr.member_cost - prev.member_cost
            if abs(delta) >= max(8.0, 0.25 * max(prev.member_cost, 1.0)):
                direction = "increased" if delta > 0 else "decreased"
                facts.append(GroundedFact(
                    f"The amount paid at the pharmacy {direction} from "
                    f"${prev.member_cost:.2f} on {prev.fill_date:%B %d} to "
                    f"${curr.member_cost:.2f} on {curr.fill_date:%B %d}",
                    "cost", f"{curr.fill_date:%Y-%m-%d}"))
                if delta > 0:
                    reasons.append(ReasonCode.COST_DELTA_LARGE)
            if (curr.deductible_phase == DeductiblePhase.DEDUCTIBLE
                    and prev.deductible_phase != DeductiblePhase.DEDUCTIBLE):
                facts.append(GroundedFact(
                    "The plan year reset and the annual deductible is being met again, "
                    "which raises the amount paid at the pharmacy until it is satisfied",
                    "cost", f"{curr.fill_date:%Y-%m-%d}"))
                reasons.append(ReasonCode.DEDUCTIBLE_PHASE_TRANSITION)

        # -- contact history ----------------------------------------------
        if ctx.contact_count_30d >= 2:
            facts.append(GroundedFact(
                f"There have been {ctx.contact_count_30d} contacts with support in the "
                "last 30 days", "interaction",
                ctx.recent_interactions[0].interaction_id if ctx.recent_interactions else "n/a"))
            reasons.append(ReasonCode.REPEAT_CONTACT)
        unresolved = [i for i in ctx.recent_interactions if not i.resolved]
        if unresolved:
            facts.append(GroundedFact(
                f"The most recent contact on {unresolved[0].occurred_on:%B %d} was not "
                "recorded as resolved", "interaction", unresolved[0].interaction_id))

        # -- supply runway -------------------------------------------------
        runway = self.refill_runway(ctx.member_id, as_of)
        if runway is not None and runway < 14:
            facts.append(GroundedFact(
                f"Based on the last fill, about {_days(max(0, runway))} of supply remain",
                "prescription", "runway"))
            if runway < INTERIM_SUPPLY_TRIGGER_DAYS:
                reasons.append(ReasonCode.REFILL_RUNWAY_SHORT)

        # -- medication profile ---------------------------------------------
        specialty = [
            rx for rx in ctx.active_prescriptions
            if self.store.drugs[rx.drug_id].is_specialty
        ]
        if specialty:
            reasons.append(ReasonCode.SPECIALTY_DRUG)
        if any(rx.is_new_to_therapy for rx in ctx.active_prescriptions):
            reasons.append(ReasonCode.NEW_TO_THERAPY)

        return facts, list(dict.fromkeys(reasons))

    def refill_runway(self, member_id: str, as_of: datetime) -> int | None:
        events = self.store.events(member_id, as_of, lookback_days=180)
        for e in reversed(events):
            if e.event_type == EventType.RX_FILLED and e.rx_id in self.store.prescriptions:
                ds = self.store.prescriptions[e.rx_id].days_supply
                return ds - (as_of - e.occurred_at).days
        return None

    # -- frame construction ----------------------------------------------

    def build_frame(
        self,
        ctx: MemberContext,
        signals: SignalSet,
        risk: RiskAssessment,
        trigger: EventType,
        as_of: datetime,
    ) -> tuple[ContextFrame, list[GroundedFact]]:
        facts, reasons = self.collect_facts(ctx, as_of)
        reasons = list(dict.fromkeys(reasons + risk.reasons))

        what = self._what_is_happening(facts, signals, trigger)
        why = self._why_it_matters(reasons, risk, signals)
        next_ = self._what_could_happen_next(reasons, risk, ctx, as_of)
        candidates = self._candidate_actions(reasons, signals, risk, ctx)

        frame = ContextFrame(
            what_is_happening=what,
            why_it_matters=why,
            relevant_evidence_ids=[f.source_id for f in facts],
            what_could_happen_next=next_,
            candidate_next_steps=candidates,
            grounded_facts=[f.render() for f in facts],
        )
        return frame, facts

    @staticmethod
    def _what_is_happening(facts, signals: SignalSet, trigger: EventType) -> str:
        if not facts:
            return (
                "No open operational issue is visible on the record at this point "
                f"in the journey (triggered by: {trigger.value})."
            )
        head = facts[0].text
        if len(facts) > 1:
            return f"{head}. Also on the record: {facts[1].text.lower()}."
        return f"{head}."

    @staticmethod
    def _why_it_matters(reasons, risk: RiskAssessment, signals: SignalSet) -> str:
        if risk.band == RiskBand.HIGH:
            lead = (
                "This combination has historically preceded a support contact or an "
                "abandoned fill"
            )
        elif risk.band == RiskBand.MEDIUM:
            lead = "This is the kind of situation that often produces a follow-up question"
        else:
            lead = "Nothing here suggests an imminent problem"

        drivers = {
            ReasonCode.PA_PENDING_INFO: "the review is blocked on paperwork rather than progressing",
            ReasonCode.REFILL_RUNWAY_SHORT: "supply runs out before the review is likely to finish",
            ReasonCode.REPEAT_CONTACT: "the same issue has already come up more than once",
            ReasonCode.COST_DELTA_LARGE: "the cost moved by enough to be noticed at the counter",
            ReasonCode.RECENT_CLAIM_REJECTION: "a recent fill did not go through",
            ReasonCode.DEDUCTIBLE_PHASE_TRANSITION: "the change has a benefit-design explanation "
                                                    "that is not obvious from the receipt",
        }
        matched = [drivers[r] for r in reasons if r in drivers][:2]
        if matched:
            return f"{lead}, because {' and '.join(matched)}."
        return f"{lead}."

    def _what_could_happen_next(self, reasons, risk, ctx, as_of) -> str:
        if ReasonCode.REFILL_RUNWAY_SHORT in reasons and ctx.open_authorizations:
            return (
                "Without intervention the supply is likely to run out while the review "
                "is still open, which usually results in a call from the pharmacy "
                "counter."
            )
        if ReasonCode.PA_PENDING_INFO in reasons:
            return (
                "The review will stay open until the prescriber returns the requested "
                "documentation. Nothing advances on its own."
            )
        if ReasonCode.RECENT_CLAIM_REJECTION in reasons:
            return (
                "If the underlying reason is not addressed, the next fill attempt will "
                "reject for the same reason."
            )
        if ReasonCode.COST_DELTA_LARGE in reasons:
            return (
                "The next fill will carry a similar amount unless the benefit phase or "
                "the fill channel changes."
            )
        if risk.band == RiskBand.LOW:
            return "No specific follow-on issue is indicated by the current record."
        return "The situation may resolve without contact, but the pattern is elevated."

    @staticmethod
    def _candidate_actions(reasons, signals: SignalSet, risk, ctx) -> list[ActionType]:
        """Candidate generation. Deliberately generous; ``policy`` narrows it.

        Separating generation from selection means the audit trail can record
        what was considered *and* rejected, not only what was chosen.
        """
        c: list[ActionType] = []

        if signals.clinical_question_detected:
            return [ActionType.ROUTE_TO_PHARMACIST]

        if ReasonCode.PA_PENDING_INFO in reasons:
            c += [ActionType.REQUEST_MISSING_INFO, ActionType.EXPLAIN_AUTH_STATUS]
        elif ctx.open_authorizations:
            c += [ActionType.EXPLAIN_AUTH_STATUS]

        if ReasonCode.REFILL_RUNWAY_SHORT in reasons:
            c += [ActionType.PROVIDE_NEXT_STEP, ActionType.SURFACE_ALTERNATIVE_OPTION]

        if ReasonCode.RECENT_CLAIM_REJECTION in reasons:
            c += [ActionType.EXPLAIN_CLAIM_STATUS]
        if ReasonCode.OUT_OF_NETWORK_PHARMACY in reasons:
            c += [ActionType.SURFACE_ALTERNATIVE_OPTION]
        if ReasonCode.COST_DELTA_LARGE in reasons or ReasonCode.DEDUCTIBLE_PHASE_TRANSITION in reasons:
            c += [ActionType.EXPLAIN_COST_CHANGE, ActionType.SURFACE_ALTERNATIVE_OPTION]

        if signals.intent == MemberIntent.IS_THIS_COVERED:
            c.insert(0, ActionType.EXPLAIN_COVERAGE)
        elif signals.intent == MemberIntent.LOWER_COST_OPTION:
            c.insert(0, ActionType.SURFACE_ALTERNATIVE_OPTION)
        elif signals.intent == MemberIntent.WHY_DID_COST_CHANGE:
            c.insert(0, ActionType.EXPLAIN_COST_CHANGE)
        elif signals.intent == MemberIntent.AUTHORIZATION_STATUS:
            c.insert(0, ActionType.EXPLAIN_AUTH_STATUS)
        elif signals.intent == MemberIntent.CLAIM_REJECTED_WHY:
            c.insert(0, ActionType.EXPLAIN_CLAIM_STATUS)

        if ReasonCode.REPEAT_CONTACT in reasons or signals.repeat_contact_language:
            c += [ActionType.ROUTE_TO_ADVOCATE]
        if ctx.contact_count_30d >= 3 and any(not i.resolved for i in ctx.recent_interactions):
            c.insert(0, ActionType.ESCALATE_CASE)

        if signals.barrier == BarrierType.COST:
            c += [ActionType.SURFACE_ALTERNATIVE_OPTION]

        if not c:
            c = [ActionType.SUPPRESS_OUTREACH]
        # stable de-duplication preserves priority order
        return list(dict.fromkeys(c))
