"""Graph state and nodes.

Nodes are deliberately thin. Every node does three things and nothing else:
read state, call one plain-Python component, write state. All the real logic
lives in ``context_engine``, ``policy``, ``gate``, ``generate`` and ``score``,
which are importable and testable without a graph runtime.

That separation is the reason the test suite can cover the decision logic
directly, and the reason swapping the orchestrator later would be a mechanical
change rather than a rewrite.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta
from typing import Annotated, Any, Literal, TypedDict

import operator

from careweave.config import SETTINGS
from careweave.data.store import JourneyStore
from careweave.domain.enums import (
    ActionType,
    EventType,
    GateVerdict,
    HumanDecisionType,
    ReasonCode,
    RiskBand,
)
from careweave.domain.models import (
    ActionDecision,
    ContextFrame,
    EvidenceChunk,
    GateResult,
    GeneratedResponse,
    HumanDecision,
    LedgerRecord,
    MemberContext,
    RiskAssessment,
    SignalSet,
    TraceEntry,
)
from careweave.governance.gate import GovernanceGate
from careweave.governance.ledger import FrictionLedger
from careweave.graph.context_engine import ContextEngine
from careweave.graph.generate import GenerationInputs, Responder, advocate_summary
from careweave.graph.policy import ActionPolicy
from careweave.ml.features import ObservationPoint
from careweave.ml.score import FrictionScorer
from careweave.nlp.extract import SignalExtractor
from careweave.rag.retriever import PolicyRetriever


#: How far back a document can be and still speak for the member's *current*
#: intent. Beyond this it is history, available to the context engine but not
#: to signal extraction.
SIGNAL_RECENCY_DAYS = 21


class CaseState(TypedDict, total=False):
    # -- inputs
    case_id: str
    member_id: str
    as_of: datetime
    trigger: EventType
    is_proactive: bool
    member_utterance: str | None

    # -- assembled context
    ctx: MemberContext
    signals: SignalSet

    # -- inference
    risk: RiskAssessment

    # -- evidence
    evidence: list[EvidenceChunk]
    evidence_sufficient: bool
    evidence_conflict: bool
    evidence_note: str

    # -- decision
    frame: ContextFrame
    decision: ActionDecision
    gate: GateResult

    # -- human
    human_decision: HumanDecision | None

    # -- output
    response: GeneratedResponse | None
    advocate_brief: str | None

    # -- audit (accumulating reducer, never overwritten)
    trace: Annotated[list[TraceEntry], operator.add]
    ledger_id: str | None


class Components:
    """Dependency bundle. Constructed once, injected into every node."""

    def __init__(
        self,
        store: JourneyStore,
        retriever: PolicyRetriever,
        scorer: FrictionScorer,
        ledger: FrictionLedger,
        gate: GovernanceGate | None = None,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.scorer = scorer
        self.ledger = ledger
        self.context_engine = ContextEngine(store)
        self.policy = ActionPolicy()
        self.gate = gate or GovernanceGate()
        self.responder = Responder()
        self.extractor = SignalExtractor(
            known_drugs={d.name for d in store.drugs.values()}
        )


def _trace(node: str, started: float, summary: str, **detail: Any) -> list[TraceEntry]:
    return [
        TraceEntry(
            node=node,
            started_at=datetime.now(),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            summary=summary,
            detail=detail,
        )
    ]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def make_nodes(c: Components) -> dict[str, Any]:
    def ingest_event(state: CaseState) -> dict[str, Any]:
        t = time.perf_counter()
        case_id = state.get("case_id") or f"CASE-{uuid.uuid4().hex[:10]}"
        return {
            "case_id": case_id,
            "trace": _trace(
                "ingest_event", t,
                f"Opened case for {state['member_id']} triggered by "
                f"{state['trigger'].value}",
                as_of=state["as_of"].isoformat(),
                proactive=state.get("is_proactive", False),
            ),
        }

    def assemble_context(state: CaseState) -> dict[str, Any]:
        t = time.perf_counter()
        ctx = c.store.member_context(state["member_id"], state["as_of"])
        return {
            "ctx": ctx,
            "trace": _trace(
                "assemble_context", t,
                f"Joined {len(ctx.active_prescriptions)} prescriptions, "
                f"{len(ctx.open_authorizations)} open authorizations, "
                f"{len(ctx.recent_claims)} recent claims, "
                f"{len(ctx.recent_interactions)} interactions",
                contact_count_30d=ctx.contact_count_30d,
                rejection_count_90d=ctx.rejection_count_90d,
            ),
        }

    def extract_signals(state: CaseState) -> dict[str, Any]:
        t = time.perf_counter()
        # Historical documents only inform the *current* signal if they are
        # recent. A clinical aside written four months ago is part of the
        # member's history, not evidence that they are asking a clinical
        # question today -- without this bound it silently forces every future
        # case for that member into pharmacist routing.
        all_docs = c.store.documents(state["member_id"], state["as_of"], limit=8)
        cutoff = state["as_of"] - timedelta(days=SIGNAL_RECENCY_DAYS)
        docs = [d for d in all_docs if d.created_at >= cutoff][:3]
        utterance = state.get("member_utterance")
        if utterance:
            from careweave.domain.models import Document
            from careweave.domain.enums import DocumentType

            # When the member has just said something, their own words decide the
            # intent. Blending in historical documents let an old cost complaint
            # override a live authorization question -- the current utterance is
            # the signal, history is context for everything downstream.
            docs = [
                Document(
                    document_id=f"LIVE-{state['case_id']}",
                    document_type=DocumentType.MEMBER_MESSAGE,
                    title="Live member message",
                    body=utterance,
                    created_at=state["as_of"],
                    member_id=state["member_id"],
                )
            ]
        signals = c.extractor.extract(docs)
        return {
            "signals": signals,
            "trace": _trace(
                "extract_signals", t,
                f"intent={signals.intent.value} barrier={signals.barrier.value} "
                f"urgency={signals.urgency.value} clinical={signals.clinical_question_detected}",
                n_documents=len(docs), confidence=signals.confidence,
            ),
        }

    def score_friction(state: CaseState) -> dict[str, Any]:
        t = time.perf_counter()
        risk, _features = c.scorer.score_point(
            ObservationPoint(state["member_id"], state["as_of"], state["trigger"], None)
        )
        return {
            "risk": risk,
            "trace": _trace(
                "score_friction", t,
                f"risk={risk.score:.3f} band={risk.band.value} "
                f"({len(risk.reasons)} reason codes)",
                model_version=risk.model_version,
                reasons=[r.value for r in risk.reasons],
                top_contributions=risk.contributions,
            ),
        }

    def retrieve_evidence(state: CaseState) -> dict[str, Any]:
        t = time.perf_counter()
        signals: SignalSet = state["signals"]
        ctx: MemberContext = state["ctx"]

        query_parts = [signals.intent.value.replace("_", " ")]
        if ctx.open_authorizations:
            query_parts.append("prior authorization pending review timeframe")
        if any(cl.status.value == "rejected" for cl in ctx.recent_claims):
            query_parts.append("pharmacy claim rejection reason")
        if signals.mentions_cost:
            query_parts.append("cost share tier deductible")
        if signals.mentions_access:
            query_parts.append("emergency interim supply")
        query = " ".join(query_parts)

        evidence = c.retriever.retrieve(query, plan_id=ctx.plan.plan_id)
        sufficient, note = c.retriever.sufficiency(evidence)
        conflict = _detect_conflict(evidence)
        return {
            "evidence": evidence,
            "evidence_sufficient": sufficient,
            "evidence_conflict": conflict,
            "evidence_note": note,
            "trace": _trace(
                "retrieve_evidence", t,
                f"{len(evidence)} chunks, sufficient={sufficient}, conflict={conflict}",
                query=query,
                chunks=[{"id": e.chunk_id, "score": e.score} for e in evidence],
            ),
        }

    def build_context_frame(state: CaseState) -> dict[str, Any]:
        t = time.perf_counter()
        frame, facts = c.context_engine.build_frame(
            state["ctx"], state["signals"], state["risk"],
            state["trigger"], state["as_of"],
        )
        return {
            "frame": frame,
            "trace": _trace(
                "build_context_frame", t,
                f"{len(facts)} grounded facts, "
                f"{len(frame.candidate_next_steps)} candidate actions",
                what=frame.what_is_happening,
                candidates=[a.value for a in frame.candidate_next_steps],
            ),
        }

    def select_action(state: CaseState) -> dict[str, Any]:
        t = time.perf_counter()
        decision = c.policy.select(
            frame=state["frame"], ctx=state["ctx"], signals=state["signals"],
            risk=state["risk"], evidence=state.get("evidence", []),
            is_proactive=state.get("is_proactive", False),
        )
        return {
            "decision": decision,
            "trace": _trace(
                "select_action", t,
                f"selected {decision.action.value} (confidence {decision.confidence:.2f})",
                policy_version=decision.policy_version,
                rejected=decision.rejected_alternatives,
            ),
        }

    def governance_gate(state: CaseState) -> dict[str, Any]:
        t = time.perf_counter()
        result = c.gate.evaluate(
            decision=state["decision"], ctx=state["ctx"], signals=state["signals"],
            risk=state["risk"],
            evidence_sufficient=state.get("evidence_sufficient", False),
            evidence_conflict=state.get("evidence_conflict", False),
            as_of=state["as_of"], is_proactive=state.get("is_proactive", False),
        )
        return {
            "gate": result,
            "trace": _trace(
                "governance_gate", t,
                f"verdict={result.verdict.value} requires_human={result.requires_human}",
                gate_version=result.gate_version,
                reasons=[r.value for r in result.reasons], notes=result.notes,
            ),
        }

    def human_review(state: CaseState) -> dict[str, Any]:
        """Suspend the graph and wait for a person.

        This is the node that justifies the orchestration framework. Execution
        stops here, state is checkpointed, and the process can exit. When a
        reviewer responds -- minutes or days later -- the graph resumes from this
        exact point rather than replaying from the start.
        """
        t = time.perf_counter()
        from langgraph.types import interrupt

        decision: ActionDecision = state["decision"]
        gate: GateResult = state["gate"]
        brief = advocate_summary(
            GenerationInputs(
                action=decision.action, frame=state["frame"], ctx=state["ctx"],
                signals=state["signals"], evidence=state.get("evidence", []),
                decision=decision, audience="advocate",
            ),
            risk_note=(
                f"{state['risk'].score:.2f} ({state['risk'].band.value}) — "
                + ", ".join(r.value for r in state["risk"].reasons[:4])
            ),
        )

        review = interrupt(
            {
                "case_id": state["case_id"],
                "member_id": state["member_id"],
                "recommended_action": decision.action.value,
                "confidence": decision.confidence,
                "gate_verdict": gate.verdict.value,
                "gate_reasons": [r.value for r in gate.reasons],
                "gate_notes": gate.notes,
                "brief": brief,
                "options": [d.value for d in HumanDecisionType],
                "available_actions": [a.value for a in ActionType],
            }
        )

        human = _parse_human_response(review)
        return {
            "human_decision": human,
            "advocate_brief": brief,
            "trace": _trace(
                "human_review", t,
                f"human {human.decision.value}"
                + (f" -> {human.modified_action.value}" if human.modified_action else ""),
                reviewer=human.reviewer_id, rationale=human.rationale,
            ),
        }

    def generate_response(state: CaseState) -> dict[str, Any]:
        t = time.perf_counter()
        decision: ActionDecision = state["decision"]
        human = state.get("human_decision")

        action = decision.action
        if human and human.decision == HumanDecisionType.MODIFY and human.modified_action:
            action = human.modified_action

        inputs = GenerationInputs(
            action=action, frame=state["frame"], ctx=state["ctx"],
            signals=state["signals"], evidence=state.get("evidence", []),
            decision=decision, audience="member",
        )
        response = c.responder.generate(inputs)

        if human and human.decision == HumanDecisionType.MODIFY and human.modified_text:
            response = response.model_copy(
                update={"text": human.modified_text, "grounded": True,
                        "ungrounded_spans": [], "generator_version":
                        f"{response.generator_version}+human_edit"}
            )

        brief = state.get("advocate_brief") or advocate_summary(
            inputs,
            risk_note=f"{state['risk'].score:.2f} ({state['risk'].band.value})",
        )
        return {
            "response": response,
            "advocate_brief": brief,
            "trace": _trace(
                "generate_response", t,
                f"{len(response.text.split())} words, grounded={response.grounded}, "
                f"{len(response.citations)} citations",
                generator=response.generator_version,
            ),
        }

    def verify_response(state: CaseState) -> dict[str, Any]:
        t = time.perf_counter()
        response = state.get("response")
        if response is None:
            return {"trace": _trace("verify_response", t, "no response to verify")}
        ok = response.grounded
        return {
            "trace": _trace(
                "verify_response", t,
                "grounding verified" if ok
                else f"withheld: {len(response.ungrounded_spans)} problems",
                problems=response.ungrounded_spans,
            ),
        }

    def commit_ledger(state: CaseState) -> dict[str, Any]:
        """Terminal node for EVERY path, including deliberate no-action."""
        t = time.perf_counter()
        gate: GateResult | None = state.get("gate")
        decision: ActionDecision | None = state.get("decision")
        risk: RiskAssessment | None = state.get("risk")
        response: GeneratedResponse | None = state.get("response")

        record = LedgerRecord(
            ledger_id=f"LED-{uuid.uuid4().hex[:12]}",
            case_id=state["case_id"],
            member_id=state["member_id"],
            created_at=state["as_of"],
            trigger_event_type=state["trigger"],
            risk_score=risk.score if risk else None,
            risk_band=risk.band if risk else None,
            risk_reasons=risk.reasons if risk else [],
            signals=state.get("signals"),
            evidence_ids=[e.chunk_id for e in state.get("evidence", [])],
            selected_action=decision.action if decision else None,
            action_confidence=decision.confidence if decision else None,
            gate_verdict=gate.verdict if gate else GateVerdict.SUPPRESS,
            gate_reasons=gate.reasons if gate else [ReasonCode.NO_ACTIONABLE_FRICTION],
            requires_human=gate.requires_human if gate else False,
            human_decision=state.get("human_decision"),
            response_text=response.text if response else None,
            citations=response.citations if response else [],
            model_versions={
                "risk_model": risk.model_version if risk else "n/a",
                "extractor": state["signals"].extractor_version
                if state.get("signals") else "n/a",
                "retriever": c.retriever.index_version,
                "policy": decision.policy_version if decision else "n/a",
                "gate": gate.gate_version if gate else "n/a",
                "generator": response.generator_version if response else "n/a",
            },
            trace=state.get("trace", []),
        )
        c.ledger.append(record)

        # a delivered proactive message updates the suppression ledger
        if (
            gate and gate.verdict == GateVerdict.ALLOW
            and state.get("is_proactive") and decision
        ):
            c.gate.ledger.record(state["member_id"], state["as_of"], decision.action)

        return {
            "ledger_id": record.ledger_id,
            "trace": _trace(
                "commit_ledger", t,
                f"audit record {record.ledger_id} written "
                f"(verdict={record.gate_verdict.value})",
            ),
        }

    return {
        "ingest_event": ingest_event,
        "assemble_context": assemble_context,
        "extract_signals": extract_signals,
        "score_friction": score_friction,
        "retrieve_evidence": retrieve_evidence,
        "build_context_frame": build_context_frame,
        "select_action": select_action,
        "governance_gate": governance_gate,
        "human_review": human_review,
        "generate_response": generate_response,
        "verify_response": verify_response,
        "commit_ledger": commit_ledger,
    }


# ---------------------------------------------------------------------------
# Routing predicates
# ---------------------------------------------------------------------------


def route_after_risk(state: CaseState) -> Literal["retrieve_evidence", "commit_ledger"]:
    """Low risk with nothing the member asked about ends the run early.

    Cheap and deliberate: most journey events are unremarkable, and running
    retrieval plus generation on all of them would be waste dressed up as
    thoroughness. The ledger still records that the case was seen and dismissed.
    """
    risk: RiskAssessment = state["risk"]
    signals: SignalSet = state["signals"]
    member_asked = bool(state.get("member_utterance")) or signals.confidence >= 0.5
    if risk.band == RiskBand.LOW and not member_asked:
        return "commit_ledger"
    return "retrieve_evidence"


def route_after_gate(
    state: CaseState,
) -> Literal["human_review", "generate_response", "commit_ledger"]:
    gate: GateResult = state["gate"]
    if gate.requires_human:
        return "human_review"
    if gate.verdict == GateVerdict.ALLOW:
        return "generate_response"
    return "commit_ledger"


def route_after_human(state: CaseState) -> Literal["generate_response", "commit_ledger"]:
    human = state.get("human_decision")
    if human is None or human.decision == HumanDecisionType.REJECT:
        return "commit_ledger"
    return "generate_response"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_conflict(evidence: list[EvidenceChunk]) -> bool:
    """Flag mutually inconsistent retrieved evidence.

    Detects the specific failure this corpus can produce: two chunks quoting
    different review timeframes or different day counts for the same concept.
    Narrow by design -- a general contradiction detector would be a research
    project, and a fake one would be worse than none.
    """
    import re

    day_claims: dict[str, set[str]] = {}
    for e in evidence:
        text = e.text.lower()
        for concept, pattern in (
            ("review_days", r"reviewed within (\d+) (?:business )?days"),
            ("appeal_days", r"within (\d+) calendar days"),
            ("supply_days", r"(?:up to a )?(\d+) day fill"),
        ):
            for m in re.finditer(pattern, text):
                day_claims.setdefault(concept, set()).add(m.group(1))
    return any(len(v) > 1 for v in day_claims.values())


def _parse_human_response(review: Any) -> HumanDecision:
    """Normalise whatever the resume payload contains into a typed decision."""
    if isinstance(review, HumanDecision):
        return review
    if not isinstance(review, dict):
        return HumanDecision(
            decision=HumanDecisionType.REJECT, reviewer_id="unknown",
            decided_at=datetime.now(), rationale="unparseable review payload",
        )
    try:
        kind = HumanDecisionType(review.get("decision", "reject"))
    except ValueError:
        kind = HumanDecisionType.REJECT
    modified = review.get("modified_action")
    try:
        modified_action = ActionType(modified) if modified else None
    except ValueError:
        modified_action = None
    return HumanDecision(
        decision=kind,
        reviewer_id=str(review.get("reviewer_id", "advocate")),
        decided_at=datetime.now(),
        rationale=review.get("rationale"),
        modified_action=modified_action,
        modified_text=review.get("modified_text"),
    )
