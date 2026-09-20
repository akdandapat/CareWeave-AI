"""HTTP API.

Deliberately thin: it validates input, calls the engine, and shapes output. All
decision logic lives behind ``CareWeaveEngine``, so the API can be swapped for a
CLI or a queue consumer without touching anything that matters.

Three surfaces, one intelligence layer -- the split from the transcripts:

* ``/member/*``    -- what the member sees. No risk scores, no reason codes.
* ``/advocate/*``  -- the case view, including risk, evidence and overrides.
* ``/ops/*``       -- aggregate friction, recurring causes, governance summary.

The disclosure boundary between member and advocate is enforced by *separate
endpoints returning separate shapes*, not by a flag on one response. A field
that is never serialised cannot leak through a UI bug.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from careweave.config import SETTINGS
from careweave.data.store import get_store
from careweave.domain.enums import ActionType, EventType, GateVerdict, HumanDecisionType
from careweave.governance.ledger import get_ledger
from careweave.graph.build import get_engine

app = FastAPI(
    title="CareWeave AI",
    version="1.0.0",
    description=(
        "Journey-state intelligence and action layer for pharmacy benefits. "
        "Synthetic data only. Research prototype, not a clinical system."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    member_id: str
    message: str = Field(min_length=1, max_length=2000)
    as_of: datetime | None = None


class TriggerRequest(BaseModel):
    member_id: str
    trigger: EventType = EventType.REFILL_DUE
    as_of: datetime | None = None
    is_proactive: bool = True


class ReviewRequest(BaseModel):
    decision: HumanDecisionType
    reviewer_id: str = "advocate"
    rationale: str | None = None
    modified_action: ActionType | None = None
    modified_text: str | None = None


_FRONTEND = SETTINGS.paths.root / "frontend" / "index.html"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the UI from the same origin so the demo needs no second process."""
    if not _FRONTEND.exists():
        raise HTTPException(404, "frontend/index.html not found")
    return FileResponse(_FRONTEND)


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    store = get_store()
    engine = get_engine()
    return {
        "status": "ok",
        "members": len(store.members),
        "policy_documents": len(store.policy_documents),
        "risk_model": engine.scorer.model_version,
        "risk_model_available": engine.scorer.available,
        "llm_provider": SETTINGS.llm.version_tag,
        "notice": "All data is synthetic. Not a clinical decision system.",
    }


@app.get("/members")
def list_members(
    limit: int = Query(50, le=500),
    search: str | None = None,
) -> list[dict[str, Any]]:
    store = get_store()
    out = []
    for mid in store.member_ids:
        m = store.members[mid]
        if search and search.lower() not in f"{mid} {m.full_name}".lower():
            continue
        out.append(
            {
                "member_id": mid,
                "name": m.full_name,
                "plan": store.plans[m.plan_id].name,
                "state": m.state,
            }
        )
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Member surface
# ---------------------------------------------------------------------------


@app.post("/member/ask")
def member_ask(req: AskRequest) -> dict[str, Any]:
    """Answer a member's question. Returns only member-appropriate fields."""
    engine = get_engine()
    if req.member_id not in engine.store.members:
        raise HTTPException(404, f"unknown member {req.member_id}")

    as_of = req.as_of or _default_as_of()
    result = engine.run(
        member_id=req.member_id, as_of=as_of,
        trigger=EventType.MEMBER_CONTACTED_SUPPORT,
        member_utterance=req.message,
    )

    if result["suspended"]:
        pending = engine.pending_review(result["case_id"])
        return {
            "case_id": result["case_id"],
            "status": "handed_to_a_person",
            # deliberately vague: the member does not need to know which
            # internal rule fired, only that a person is involved
            "message": _handoff_message(pending),
            "awaiting_review": True,
        }

    return {
        "case_id": result["case_id"],
        "status": "answered" if result["response_text"] else "no_action",
        "message": result["response_text"],
        "grounded": result["grounded"],
        # sources are shown so the member can see the answer is not invented,
        # but the internal chunk ids are mapped to human titles
        "sources": [
            {"title": e["title"], "source": e["source"]}
            for e in result["evidence"][:3]
        ],
        "next_steps_available": ["speak_to_an_advocate"],
    }


@app.get("/member/{member_id}/timeline")
def member_timeline(
    member_id: str, as_of: datetime | None = None, limit: int = Query(40, le=200)
) -> list[dict[str, Any]]:
    store = get_store()
    if member_id not in store.members:
        raise HTTPException(404, f"unknown member {member_id}")
    as_of = as_of or _default_as_of()
    events = store.events(member_id, as_of)[-limit:]
    return [
        {
            "event_id": e.event_id,
            "occurred_at": e.occurred_at.isoformat(),
            "type": e.event_type.value,
            "rx_id": e.rx_id,
            "payload": e.payload,
        }
        for e in events
    ]


# ---------------------------------------------------------------------------
# Advocate surface
# ---------------------------------------------------------------------------


@app.get("/advocate/{member_id}/context")
def advocate_context(member_id: str, as_of: datetime | None = None) -> dict[str, Any]:
    """Member 360 -- the 'do not make them repeat themselves' view."""
    store = get_store()
    if member_id not in store.members:
        raise HTTPException(404, f"unknown member {member_id}")
    as_of = as_of or _default_as_of()
    ctx = store.member_context(member_id, as_of)

    return {
        "member": {
            "member_id": ctx.member_id,
            "name": ctx.member.full_name,
            "plan": ctx.plan.name,
            "state": ctx.member.state,
            "preferred_channel": ctx.member.preferred_channel.value,
        },
        "as_of": as_of.isoformat(),
        "prescriptions": [
            {
                "rx_id": rx.rx_id,
                "drug": store.drugs[rx.drug_id].name,
                "is_specialty": store.drugs[rx.drug_id].is_specialty,
                "days_supply": rx.days_supply,
                "written_on": rx.written_on.isoformat(),
            }
            for rx in ctx.active_prescriptions
        ],
        "open_authorizations": [
            {
                "pa_id": pa.pa_id,
                "drug": (d.name if (d := store.drug_for_rx(pa.rx_id)) else None),
                "status": pa.status.value,
                "requested_on": pa.requested_on.isoformat() if pa.requested_on else None,
                "days_open": (
                    (as_of.date() - pa.requested_on).days if pa.requested_on else None
                ),
                "missing_info": pa.missing_info,
            }
            for pa in ctx.open_authorizations
        ],
        "recent_claims": [
            {
                "claim_id": c.claim_id,
                "drug": (d.name if (d := store.drug_for_rx(c.rx_id)) else None),
                "submitted_on": c.submitted_on.isoformat(),
                "status": c.status.value,
                "reject_code": c.reject_code.value if c.reject_code else None,
                "member_cost": c.member_cost,
            }
            for c in ctx.recent_claims[:10]
        ],
        "cost_history": [
            {
                "fill_date": s.fill_date.isoformat(),
                "member_cost": s.member_cost,
                "phase": s.deductible_phase.value if s.deductible_phase else None,
            }
            for s in ctx.cost_history
        ],
        "interactions": [
            {
                "interaction_id": i.interaction_id,
                "occurred_on": i.occurred_on.isoformat(),
                "channel": i.channel.value,
                "intent": i.intent.value,
                "resolved": i.resolved,
                "escalated": i.escalated,
            }
            for i in ctx.recent_interactions
        ],
        "documents": [
            {
                "document_id": d.document_id,
                "type": d.document_type.value,
                "title": d.title,
                "created_at": d.created_at.isoformat(),
                "excerpt": d.body[:300],
            }
            for d in store.documents(member_id, as_of, limit=6)
        ],
        "summary_counters": {
            "contacts_30d": ctx.contact_count_30d,
            "rejections_90d": ctx.rejection_count_90d,
            "days_since_last_contact": ctx.days_since_last_contact,
        },
    }


@app.post("/advocate/assess")
def advocate_assess(req: TriggerRequest) -> dict[str, Any]:
    """Run the engine and return the full internal view, including risk."""
    engine = get_engine()
    if req.member_id not in engine.store.members:
        raise HTTPException(404, f"unknown member {req.member_id}")
    as_of = req.as_of or _default_as_of()
    result = engine.run(
        member_id=req.member_id, as_of=as_of, trigger=req.trigger,
        is_proactive=req.is_proactive,
    )
    if result["suspended"]:
        result["pending_review"] = engine.pending_review(result["case_id"])
    return result


@app.post("/advocate/case/{case_id}/review")
def advocate_review(case_id: str, req: ReviewRequest) -> dict[str, Any]:
    """Submit a human decision and resume the suspended graph."""
    engine = get_engine()
    payload = req.model_dump(mode="json", exclude_none=True)
    try:
        return engine.resume(case_id, payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"could not resume case {case_id}: {exc}") from exc


@app.get("/advocate/queue")
def advocate_queue(limit: int = Query(20, le=100)) -> list[dict[str, Any]]:
    """Cases the engine has flagged for a person, most recent first."""
    engine = get_engine()
    store = get_store()
    out = []
    for item in engine.pending_queue()[:limit]:
        mid = item.get("member_id")
        out.append(
            {
                **item,
                "member_name": (
                    store.members[mid].full_name if mid in store.members else None
                ),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Operations surface
# ---------------------------------------------------------------------------


@app.get("/ops/summary")
def ops_summary() -> dict[str, Any]:
    return get_ledger().summary()


@app.get("/ops/trends")
def ops_trends(bucket_days: int = Query(7, ge=1, le=30)) -> list[dict[str, Any]]:
    return get_ledger().friction_trends(bucket_days)


@app.get("/ops/root-causes")
def ops_root_causes(top_n: int = Query(8, le=20)) -> list[dict[str, Any]]:
    """What keeps causing friction, aggregated across the population."""
    return get_ledger().recurring_root_causes(top_n)


@app.get("/ops/ledger")
def ops_ledger(
    member_id: str | None = None,
    verdict: GateVerdict | None = None,
    limit: int = Query(50, le=500),
) -> list[dict[str, Any]]:
    """Raw audit records. The 'why did this member get this message' query."""
    records = get_ledger().query(member_id=member_id, verdict=verdict)
    records.sort(key=lambda r: r.created_at, reverse=True)
    return [
        {
            "ledger_id": r.ledger_id,
            "case_id": r.case_id,
            "member_id": r.member_id,
            "created_at": r.created_at.isoformat(),
            "trigger": r.trigger_event_type.value,
            "risk_score": r.risk_score,
            "risk_band": r.risk_band.value if r.risk_band else None,
            "risk_reasons": [x.value for x in r.risk_reasons],
            "action": r.selected_action.value if r.selected_action else None,
            "confidence": r.action_confidence,
            "verdict": r.gate_verdict.value,
            "gate_reasons": [x.value for x in r.gate_reasons],
            "requires_human": r.requires_human,
            "human_decision": (
                r.human_decision.decision.value if r.human_decision else None
            ),
            "evidence_ids": r.evidence_ids,
            "model_versions": r.model_versions,
            "response_text": r.response_text,
        }
        for r in records[:limit]
    ]


@app.get("/ops/reports/{name}")
def ops_report(name: Literal["ml", "rag", "extraction", "scenario"]) -> dict[str, Any]:
    """Serve the generated evaluation reports so the UI can show real numbers."""
    import json

    path = SETTINGS.paths.artifacts / f"{name}_report.json"
    if not path.exists():
        raise HTTPException(
            404,
            f"{name} report not generated yet -- run the corresponding "
            "evaluation module first",
        )
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_as_of() -> datetime:
    """Latest timestamp the synthetic world knows about."""
    return get_store().window[1]


def _handoff_message(pending: dict[str, Any] | None) -> str:
    reasons = set((pending or {}).get("gate_reasons", []))
    if "clinical_topic_detected" in reasons:
        return (
            "That's a question for a pharmacist rather than for me. I've passed it "
            "along and someone will follow up with you directly."
        )
    if "distress_detected" in reasons:
        return (
            "I want to make sure you get real help with this, so I've asked an "
            "advocate to pick it up. They'll have your history in front of them."
        )
    return (
        "I don't have enough on your record to answer this confidently, so I've "
        "handed it to an advocate rather than guess. They'll have the full history."
    )
