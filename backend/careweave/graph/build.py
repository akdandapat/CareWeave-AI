"""Graph assembly and the engine facade.

The checkpointer is what makes the prior-authorization case work. A PA opens on
a Tuesday and resolves the following Monday; the workflow that opened it has to
survive that gap, and survive the process restarting in between. Keying the
checkpoint on ``case_id`` means resuming is a lookup, not a replay.

``MemorySaver`` is the default so the repository runs with no infrastructure.
``SqliteSaver`` or ``PostgresSaver`` is a one-line swap and is what a real
deployment would use; see docs/architecture.md.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Iterator

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from careweave.data.store import JourneyStore, get_store
from careweave.domain.enums import ActionType, EventType, GateVerdict, HumanDecisionType
from careweave.governance.gate import GovernanceGate
from careweave.governance.ledger import FrictionLedger, get_ledger
from careweave.graph.nodes import (
    CaseState,
    Components,
    make_nodes,
    route_after_gate,
    route_after_human,
    route_after_risk,
)
from careweave.ml.score import FrictionScorer
from careweave.rag.retriever import PolicyRetriever, load_retriever


def _quiet_checkpoint_notices() -> None:
    """Silence LangGraph's per-type msgpack notices.

    The checkpointer round-trips our pydantic models and enums, which it warns
    about once per type per process. The behaviour is intended -- the models are
    ours -- so the notice is noise. Registering the modules explicitly is the
    forward-compatible fix and is tracked in docs/roadmap.md.
    """
    for name in ("langgraph.checkpoint.serde.jsonplus", "langgraph.checkpoint"):
        logging.getLogger(name).setLevel(logging.ERROR)


def build_graph(components: Components, checkpointer=None):
    _quiet_checkpoint_notices()
    nodes = make_nodes(components)
    g = StateGraph(CaseState)

    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.add_edge(START, "ingest_event")
    g.add_edge("ingest_event", "assemble_context")
    g.add_edge("assemble_context", "extract_signals")
    g.add_edge("extract_signals", "score_friction")

    # low risk and nothing asked -> straight to audit, no retrieval, no generation
    g.add_conditional_edges(
        "score_friction", route_after_risk,
        {"retrieve_evidence": "retrieve_evidence", "commit_ledger": "commit_ledger"},
    )

    g.add_edge("retrieve_evidence", "build_context_frame")
    g.add_edge("build_context_frame", "select_action")
    g.add_edge("select_action", "governance_gate")

    g.add_conditional_edges(
        "governance_gate", route_after_gate,
        {
            "human_review": "human_review",
            "generate_response": "generate_response",
            "commit_ledger": "commit_ledger",
        },
    )
    g.add_conditional_edges(
        "human_review", route_after_human,
        {"generate_response": "generate_response", "commit_ledger": "commit_ledger"},
    )

    g.add_edge("generate_response", "verify_response")
    g.add_edge("verify_response", "commit_ledger")
    g.add_edge("commit_ledger", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


def _new_case_id(member_id: str) -> str:
    return f"CASE-{member_id}-{uuid.uuid4().hex[:8]}"


class CareWeaveEngine:
    """Facade over the graph. The API and the CLI both talk to this."""

    def __init__(
        self,
        store: JourneyStore | None = None,
        retriever: PolicyRetriever | None = None,
        scorer: FrictionScorer | None = None,
        ledger: FrictionLedger | None = None,
        checkpointer=None,
    ) -> None:
        self.store = store or get_store()
        self.retriever = retriever or load_retriever(self.store)
        self.scorer = scorer or FrictionScorer(self.store)
        self.ledger = ledger or get_ledger()
        self.components = Components(
            store=self.store, retriever=self.retriever, scorer=self.scorer,
            ledger=self.ledger, gate=GovernanceGate(),
        )
        self.graph = build_graph(self.components, checkpointer)
        # Suspended cases never reach commit_ledger -- that is the whole point of
        # an interrupt -- so they are invisible to a ledger-backed queue. This
        # registry is the work queue: what the engine is waiting on a human for.
        self.pending: dict[str, dict[str, Any]] = {}

    # -- execution --------------------------------------------------------

    @staticmethod
    def _config(case_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": case_id}}

    def run(
        self,
        *,
        member_id: str,
        as_of: datetime,
        trigger: EventType = EventType.MEMBER_CONTACTED_SUPPORT,
        member_utterance: str | None = None,
        is_proactive: bool = False,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        # The case id is the checkpointer's thread key. Deriving it only from
        # member and timestamp made two different questions asked at the same
        # `as_of` collide onto one thread, so the second silently resumed the
        # first. Each invocation gets its own thread unless the caller pins one.
        case_id = case_id or _new_case_id(member_id)
        state: CaseState = {
            "case_id": case_id,
            "member_id": member_id,
            "as_of": as_of,
            "trigger": trigger,
            "is_proactive": is_proactive,
            "member_utterance": member_utterance,
            "trace": [],
        }
        result = self.graph.invoke(state, config=self._config(case_id))
        shaped = self._shape(result, case_id)
        self._track_pending(shaped)
        return shaped

    def stream(
        self, *, member_id: str, as_of: datetime, trigger: EventType,
        member_utterance: str | None = None, is_proactive: bool = False,
        case_id: str | None = None,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Node-by-node execution, for the live trace view in the UI."""
        case_id = case_id or _new_case_id(member_id)
        state: CaseState = {
            "case_id": case_id, "member_id": member_id, "as_of": as_of,
            "trigger": trigger, "is_proactive": is_proactive,
            "member_utterance": member_utterance, "trace": [],
        }
        for chunk in self.graph.stream(state, config=self._config(case_id)):
            for node_name, payload in chunk.items():
                yield node_name, payload

    def resume(self, case_id: str, review: dict[str, Any]) -> dict[str, Any]:
        """Resume a suspended case with a reviewer's decision."""
        result = self.graph.invoke(
            Command(resume=review), config=self._config(case_id)
        )
        shaped = self._shape(result, case_id)
        self.pending.pop(case_id, None)
        if shaped.get("ledger_id") and shaped.get("human_decision"):
            from careweave.domain.models import HumanDecision

            hd = shaped["human_decision"]
            if isinstance(hd, HumanDecision):
                self.ledger.attach_human_decision(shaped["ledger_id"], hd)
        return shaped

    def _track_pending(self, shaped: dict[str, Any]) -> None:
        if not shaped.get("suspended"):
            self.pending.pop(shaped["case_id"], None)
            return
        payload = self.pending_review(shaped["case_id"]) or {}
        self.pending[shaped["case_id"]] = {
            "case_id": shaped["case_id"],
            "member_id": shaped.get("member_id"),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "risk_score": shaped.get("risk_score"),
            "risk_band": shaped.get("risk_band"),
            "risk_reasons": shaped.get("risk_reasons", []),
            "recommended_action": shaped.get("selected_action"),
            "confidence": shaped.get("action_confidence"),
            "gate_verdict": shaped.get("gate_verdict"),
            "gate_reasons": shaped.get("gate_reasons", []),
            "gate_notes": shaped.get("gate_notes", []),
            "brief": payload.get("brief") or shaped.get("advocate_brief"),
            "available_actions": payload.get("available_actions", []),
        }

    def pending_queue(self) -> list[dict[str, Any]]:
        """Cases suspended awaiting a human, newest first."""
        return sorted(
            self.pending.values(), key=lambda r: r["created_at"], reverse=True
        )

    def pending_review(self, case_id: str) -> dict[str, Any] | None:
        """The payload a suspended case is waiting on, if any."""
        snapshot = self.graph.get_state(self._config(case_id))
        interrupts = getattr(snapshot, "interrupts", None) or []
        if not interrupts:
            tasks = getattr(snapshot, "tasks", ()) or ()
            interrupts = [i for task in tasks for i in (getattr(task, "interrupts", ()) or ())]
        if not interrupts:
            return None
        return getattr(interrupts[0], "value", None)

    # -- shaping ----------------------------------------------------------

    @staticmethod
    def _shape(result: dict[str, Any], case_id: str) -> dict[str, Any]:
        gate = result.get("gate")
        decision = result.get("decision")
        risk = result.get("risk")
        response = result.get("response")
        frame = result.get("frame")

        return {
            "case_id": case_id,
            "member_id": result.get("member_id"),
            "suspended": "__interrupt__" in result,
            "risk_score": risk.score if risk else None,
            "risk_band": risk.band.value if risk else None,
            "risk_reasons": [r.value for r in risk.reasons] if risk else [],
            "risk_contributions": risk.contributions if risk else {},
            "signals": (
                result["signals"].model_dump(mode="json") if result.get("signals") else None
            ),
            "context_frame": frame.model_dump(mode="json") if frame else None,
            "selected_action": decision.action.value if decision else None,
            "action_confidence": decision.confidence if decision else None,
            "rejected_alternatives": decision.rejected_alternatives if decision else {},
            "gate_verdict": gate.verdict.value if gate else None,
            "gate_reasons": [r.value for r in gate.reasons] if gate else [],
            "gate_notes": gate.notes if gate else [],
            "requires_human": gate.requires_human if gate else False,
            "human_decision": result.get("human_decision"),
            "response_text": response.text if response else None,
            "grounded": response.grounded if response else None,
            "citations": response.citations if response else [],
            "evidence": [
                {"chunk_id": e.chunk_id, "title": e.title, "score": e.score,
                 "source": e.source_label, "text": e.text[:400]}
                for e in result.get("evidence", [])
            ],
            "advocate_brief": result.get("advocate_brief"),
            "ledger_id": result.get("ledger_id"),
            "trace": [t.model_dump(mode="json") for t in result.get("trace", [])],
        }


_ENGINE: CareWeaveEngine | None = None


def get_engine() -> CareWeaveEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = CareWeaveEngine()
    return _ENGINE
