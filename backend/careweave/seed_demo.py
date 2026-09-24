"""Seed the decision ledger so the Operations view has content on first load.

Without this the hosted demo opens on an empty Operations tab, because the
ledger is written at decision time and a freshly built image has never made a
decision. ``careweave.scenarios`` does not help: it writes to its own
``scenario_ledger.jsonl`` so that the demo run stays isolated and repeatable,
while the API reads the default ``friction_ledger.jsonl``.

This runs the real engine over a sample of the population at varied timestamps
and triggers, mixing proactive and reactive cases so the verdict distribution,
the recurring-root-cause view and the weekly trend all have something honest to
show. Nothing here is fabricated: every row is a decision the engine actually
made, audited the same way a live one would be.

Deterministic by seed, so two builds of the same image produce the same ledger.

    python -m careweave.seed_demo --cases 200
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta

from careweave.config import SETTINGS
from careweave.domain.enums import EventType
from careweave.governance.ledger import get_ledger
from careweave.graph.build import CareWeaveEngine

#: Triggers weighted roughly as they occur in the event stream, so the seeded
#: ledger is not dominated by the rare-but-interesting cases.
_TRIGGERS: list[tuple[EventType, float]] = [
    (EventType.REFILL_DUE, 0.42),
    (EventType.CLAIM_REJECTED, 0.18),
    (EventType.PA_REQUESTED, 0.14),
    (EventType.COST_CHANGED, 0.12),
    (EventType.MEMBER_CONTACTED_SUPPORT, 0.09),
    (EventType.PA_INFO_REQUESTED, 0.05),
]

#: A few reactive cases carry a member's own words, so the extraction layer and
#: the intent-matching part of the action policy are exercised too.
_UTTERANCES = [
    "The pharmacy said my prescription needs approval. Where does it stand?",
    "My medication cost more this month than last month. Why?",
    "My claim was rejected at the counter. Do I owe money?",
    "I'm nearly out and nothing has moved. What happens next?",
    "Is there a cheaper way to get this filled?",
]


def seed(n_cases: int, seed_value: int) -> dict[str, int]:
    rng = random.Random(seed_value)
    ledger = get_ledger()
    engine = CareWeaveEngine(ledger=ledger)

    members = engine.store.member_ids
    _, window_end = engine.store.window
    # Stay inside the simulated window, and far enough from its edge that the
    # members have journey history to reason over.
    latest = window_end - timedelta(days=1)

    triggers = [t for t, _ in _TRIGGERS]
    weights = [w for _, w in _TRIGGERS]

    counts = {"total": 0, "suspended": 0}
    for i in range(n_cases):
        member_id = members[rng.randrange(len(members))]
        as_of = latest - timedelta(days=rng.randint(0, 120))
        trigger = rng.choices(triggers, weights=weights, k=1)[0]

        reactive = trigger == EventType.MEMBER_CONTACTED_SUPPORT or rng.random() < 0.25
        result = engine.run(
            member_id=member_id,
            as_of=as_of,
            trigger=trigger,
            member_utterance=rng.choice(_UTTERANCES) if reactive else None,
            is_proactive=not reactive,
            case_id=f"SEED-{i:05d}",
        )
        counts["total"] += 1
        if result["suspended"]:
            # A case held for a human never reaches the ledger by design, so it
            # is left suspended and shows up in the advocate queue instead.
            counts["suspended"] += 1

    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--seed", type=int, default=SETTINGS.generation.seed)
    args = ap.parse_args()

    counts = seed(args.cases, args.seed)
    summary = get_ledger().summary()

    print(f"Seeded {counts['total']} cases "
          f"({counts['suspended']} held for a human, not ledgered)")
    print(f"  ledger records:   {summary['n_decisions']}")
    print(f"  by verdict:       {summary.get('by_verdict', {})}")
    print(f"  automation rate:  {summary.get('automation_rate', 0):.0%}")
    print(f"  written to:       {SETTINGS.paths.artifacts / 'friction_ledger.jsonl'}")


if __name__ == "__main__":
    main()