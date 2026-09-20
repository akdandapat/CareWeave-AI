"""Demonstration scenarios.

Each scenario is a real case selected from the synthetic population by *searching
for the structural conditions*, not by hand-picking an ID that happens to look
good. If the generator produced no member matching a scenario's conditions, the
scenario reports that honestly rather than substituting a weaker case.

Every scenario exercises the full chain:

    raw events -> context -> risk -> evidence -> reasoning -> action ->
    gate -> human or automated -> response -> audit record

Run with::

    python -m careweave.scenarios
    python -m careweave.scenarios --only cost_surprise --verbose
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

from careweave.config import SETTINGS
from careweave.data.store import JourneyStore, _as_dt, get_store
from careweave.domain.enums import ClaimStatus, EventType, PAStatus, RejectCode
from careweave.governance.ledger import FrictionLedger
from careweave.graph.build import CareWeaveEngine


@dataclass
class ScenarioCase:
    member_id: str
    as_of: datetime
    trigger: EventType
    utterance: str | None
    is_proactive: bool
    note: str


@dataclass
class Scenario:
    key: str
    title: str
    transcript_basis: str
    find: Callable[[JourneyStore], ScenarioCase | None]
    #: optional reviewer response used to demonstrate resume-after-interrupt
    human_review: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Case finders
# ---------------------------------------------------------------------------


def _members(store: JourneyStore, limit: int = 900):
    return store.member_ids[:limit]


def find_cost_surprise(store: JourneyStore) -> ScenarioCase | None:
    """A member whose cost jumped materially between consecutive fills."""
    for mid in _members(store):
        for e in store._events_by_member.get(mid, []):
            if e.event_type != EventType.COST_CHANGED:
                continue
            delta = e.payload.get("delta", 0)
            if isinstance(delta, (int, float)) and delta >= 25:
                return ScenarioCase(
                    mid, e.occurred_at + timedelta(days=1),
                    EventType.COST_CHANGED,
                    "I picked up my prescription and it cost a lot more than last "
                    "month. Nothing changed that I know of. Why am I being charged "
                    "more?",
                    False,
                    f"cost moved by ${delta:.2f} between fills",
                )
    return None


def find_pa_friction(store: JourneyStore) -> ScenarioCase | None:
    """Authorization pending while the supply runway is closing.

    The scenario the transcripts describe most directly: nothing is broken, but
    a silent countdown is running and the member has no visibility into it.
    """
    best: tuple[int, ScenarioCase] | None = None
    for mid in _members(store):
        for pa in store._pa_by_member.get(mid, []):
            if pa.requested_on is None or pa.decided_on is None:
                continue
            duration = (_as_dt(pa.decided_on) - _as_dt(pa.requested_on)).days
            if duration < 10:
                continue
            as_of = _as_dt(pa.requested_on) + timedelta(days=duration - 3)
            case = ScenarioCase(
                mid, as_of, EventType.PA_REQUESTED, None, True,
                f"authorization open {duration - 3} days, still undecided",
            )
            if best is None or duration > best[0]:
                best = (duration, case)
    return best[1] if best else None


def find_repeat_contact(store: JourneyStore) -> ScenarioCase | None:
    """Three or more contacts inside 30 days with the issue still unresolved."""
    for mid in _members(store):
        interactions = sorted(
            store._interactions_by_member.get(mid, []), key=lambda i: i.occurred_on
        )
        for i in range(2, len(interactions)):
            window = interactions[i - 2: i + 1]
            span = (_as_dt(window[-1].occurred_on) - _as_dt(window[0].occurred_on)).days
            if span <= 30 and not window[-1].resolved:
                return ScenarioCase(
                    mid, _as_dt(window[-1].occurred_on) + timedelta(hours=2),
                    EventType.MEMBER_CONTACTED_SUPPORT,
                    "This is the third time I've called about this. I've already "
                    "explained everything twice and nobody has been able to give me "
                    "a straight answer.",
                    False,
                    f"{len(window)} contacts in {span} days, latest unresolved",
                )
    return None


def find_claim_rejection(store: JourneyStore) -> ScenarioCase | None:
    """A rejection whose explanation needs several sources reconciled."""
    interesting = {
        RejectCode.NOT_ON_FORMULARY,
        RejectCode.STEP_THERAPY_REQUIRED,
        RejectCode.QUANTITY_LIMIT_EXCEEDED,
        RejectCode.PHARMACY_OUT_OF_NETWORK,
    }
    for mid in _members(store):
        for cl in store._claims_by_member.get(mid, []):
            if cl.status == ClaimStatus.REJECTED and cl.reject_code in interesting:
                return ScenarioCase(
                    mid, _as_dt(cl.submitted_on) + timedelta(days=1),
                    EventType.CLAIM_REJECTED,
                    "The pharmacy said my prescription was rejected and handed me a "
                    "slip with a code on it. They told me to call you. What does it "
                    "mean and do I owe money?",
                    False,
                    f"rejection {cl.reject_code.value}",
                )
    return None


def find_clinical_routing(store: JourneyStore) -> ScenarioCase | None:
    """A safety question that must never receive an automated answer.

    Included as a demo scenario because the *refusal* path is a feature. A
    system that only demonstrates its successes has not been demonstrated.
    """
    for mid in _members(store):
        ctx = store.member_context(mid, datetime(2026, 4, 1))
        if ctx.active_prescriptions:
            return ScenarioCase(
                mid, datetime(2026, 4, 1), EventType.MEMBER_CONTACTED_SUPPORT,
                "I've been feeling dizzy since I started this medication. Is that "
                "normal? Should I stop taking it?",
                False,
                "clinical question -- must route to a pharmacist",
            )
    return None


def find_proactive_intervention(store: JourneyStore) -> ScenarioCase | None:
    """Friction detected before the member has contacted anyone."""
    for mid in _members(store):
        events = store._events_by_member.get(mid, [])
        for e in events:
            if e.event_type != EventType.PA_INFO_REQUESTED:
                continue
            as_of = e.occurred_at + timedelta(days=5)
            contacted = any(
                x.event_type == EventType.MEMBER_CONTACTED_SUPPORT
                and e.occurred_at <= x.occurred_at <= as_of
                for x in events
            )
            if not contacted:
                return ScenarioCase(
                    mid, as_of, EventType.PA_INFO_REQUESTED, None, True,
                    "review blocked on prescriber paperwork; member has not called",
                )
    return None


SCENARIOS: list[Scenario] = [
    Scenario(
        "cost_surprise", "Cost surprise at the counter",
        "T1: affordability is not the dollar amount, it is whether the member "
        "understands what happened and what options exist",
        find_cost_surprise,
    ),
    Scenario(
        "pa_friction", "Authorization pending, supply running out",
        "T1: 'could we have helped them understand this sooner?' / T2: prior "
        "authorization is a high-friction workflow",
        find_pa_friction,
    ),
    Scenario(
        "repeat_contact", "Repeat contact without resolution",
        "T1: having to explain the same problem over and over is the most "
        "frustrating consumer experience",
        find_repeat_contact,
        human_review={
            "decision": "modify", "reviewer_id": "ADV-2031",
            "modified_action": "escalate_case",
            "rationale": "Third contact on one unresolved issue. Owning it end to end "
                         "rather than sending another explanation.",
        },
    ),
    Scenario(
        "claim_rejection", "Claim rejection needing multi-source reconciliation",
        "T2: processing a claim requires information across multiple areas -- "
        "encounter, benefit, eligibility",
        find_claim_rejection,
    ),
    Scenario(
        "clinical_routing", "Clinical question refused and routed",
        "T1: clear guard rails for when human or clinical expertise must be "
        "involved; AI should not oversimplify clinical reality",
        find_clinical_routing,
        human_review={
            "decision": "accept", "reviewer_id": "PHM-0412",
            "rationale": "Correctly withheld from automation. Pharmacist will call.",
        },
    ),
    Scenario(
        "proactive", "Proactive intervention before contact",
        "T1: proactive service asks 'what could we have done sooner?' and is not "
        "a bunch of random messages",
        find_proactive_intervention,
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_scenario(
    engine: CareWeaveEngine, scenario: Scenario, store: JourneyStore
) -> dict[str, Any]:
    case = scenario.find(store)
    if case is None:
        return {
            "key": scenario.key, "title": scenario.title, "found": False,
            "note": "No member in the synthetic population matched this scenario's "
                    "structural conditions.",
        }

    case_id = f"SCN-{scenario.key}"
    result = engine.run(
        member_id=case.member_id, as_of=case.as_of, trigger=case.trigger,
        member_utterance=case.utterance, is_proactive=case.is_proactive,
        case_id=case_id,
    )

    resumed = None
    if result["suspended"]:
        pending = engine.pending_review(case_id)
        review = scenario.human_review or {
            "decision": "accept", "reviewer_id": "ADV-0001",
            "rationale": "Recommendation matches the record.",
        }
        resumed = engine.resume(case_id, review)
        result = {
            **result,
            "pending_review_payload": pending,
            "human_review_submitted": review,
            "after_resume": {
                "selected_action": resumed["selected_action"],
                "response_text": resumed["response_text"],
                "ledger_id": resumed["ledger_id"],
                "grounded": resumed["grounded"],
            },
        }

    return {
        "key": scenario.key,
        "title": scenario.title,
        "transcript_basis": scenario.transcript_basis,
        "found": True,
        "case": {
            "member_id": case.member_id,
            "as_of": case.as_of.isoformat(),
            "trigger": case.trigger.value,
            "proactive": case.is_proactive,
            "selection_note": case.note,
            "utterance": case.utterance,
        },
        "result": result,
    }


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run a single scenario by key")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    store = get_store()
    ledger_path = SETTINGS.paths.artifacts / "scenario_ledger.jsonl"
    FrictionLedger(ledger_path).clear()
    ledger = FrictionLedger(ledger_path)
    engine = CareWeaveEngine(store=store, ledger=ledger)

    chosen = [s for s in SCENARIOS if not args.only or s.key == args.only]
    out: list[dict[str, Any]] = []

    for scenario in chosen:
        report = run_scenario(engine, scenario, store)
        out.append(report)

        print("=" * 78)
        print(f"{report['title']}")
        print("=" * 78)
        if not report["found"]:
            print(f"  SKIPPED: {report['note']}\n")
            continue

        case, res = report["case"], report["result"]
        print(f"  member {case['member_id']}  as of {case['as_of'][:10]}  "
              f"trigger {case['trigger']}"
              + ("  [proactive]" if case["proactive"] else ""))
        print(f"  selected because: {case['selection_note']}")
        if case["utterance"]:
            print(f'\n  MEMBER SAYS: "{case["utterance"]}"')
        if res.get("signals"):
            s = res["signals"]
            print(f"  SIGNALS:     intent={s['intent']} barrier={s['barrier']} "
                  f"urgency={s['urgency']} clinical={s['clinical_question_detected']}")
        print(f"  RISK:        {res['risk_score']} ({res['risk_band']})  "
              f"{', '.join(res['risk_reasons'][:4])}")
        print(f"  EVIDENCE:    {len(res['evidence'])} chunks "
              f"{[e['chunk_id'] for e in res['evidence'][:3]]}")
        print(f"  ACTION:      {res['selected_action']} "
              f"(confidence {res['action_confidence']})")
        print(f"  GATE:        {res['gate_verdict']} "
              f"(human required: {res['requires_human']})")
        for note in res.get("gate_notes", [])[:2]:
            print(f"               {note}")

        if res.get("human_review_submitted"):
            hr = res["human_review_submitted"]
            print(f"  HUMAN:       {hr['decision']}"
                  + (f" -> {hr.get('modified_action')}" if hr.get("modified_action") else ""))
            print(f"               \"{hr['rationale']}\"")
            after = res["after_resume"]
            print(f"  AFTER RESUME action={after['selected_action']} "
                  f"grounded={after['grounded']} ledger={after['ledger_id']}")
            if after["response_text"]:
                print(f"\n  RESPONSE:\n    {after['response_text']}")
        elif res.get("response_text"):
            print(f"\n  RESPONSE:\n    {res['response_text']}")
        else:
            print("\n  RESPONSE:    (none -- deliberately took no action)")

        audit_id = (res.get("after_resume") or {}).get("ledger_id") or res["ledger_id"]
        print(f"\n  AUDIT:       {audit_id}")
        if args.verbose:
            print("  TRACE:")
            for t in res["trace"]:
                print(f"    {t['node']:<22} {t['duration_ms']:>7.1f}ms  {t['summary']}")
        print()

    path = SETTINGS.paths.artifacts / "scenario_report.json"
    path.write_text(json.dumps(out, indent=2, default=str))

    summary = ledger.summary()
    print("=" * 78)
    print("LEDGER SUMMARY")
    print("=" * 78)
    print(f"  decisions:          {summary['n_decisions']}")
    print(f"  by verdict:         {summary['by_verdict']}")
    print(f"  by action:          {summary['by_action']}")
    print(f"  human review rate:  {summary['human_review_rate']:.0%}")
    print(f"\nWritten to {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
