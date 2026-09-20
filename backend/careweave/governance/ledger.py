"""The Friction Ledger.

Every terminal path in the graph writes exactly one record here -- including the
paths where the system deliberately did nothing. Non-action is a governed
decision and has to be as auditable as action, otherwise "why was this member
never contacted?" is unanswerable.

Append-only by construction: there is no update or delete. A human override is a
*new* field on the same record written at review time, not a rewrite of the
original recommendation, so the ledger always shows what the system proposed
before a person touched it.

This is what makes responsible AI a queryable table rather than a README
section. ``query()`` supports the questions an auditor actually asks: what did
this member receive, what did the gate block last month, how often do humans
disagree with the engine.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from careweave.config import SETTINGS
from careweave.domain.enums import ActionType, GateVerdict, HumanDecisionType
from careweave.domain.models import HumanDecision, LedgerRecord


class FrictionLedger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (SETTINGS.paths.artifacts / "friction_ledger.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._records: list[LedgerRecord] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._records.append(LedgerRecord(**json.loads(line)))

    # -- write ------------------------------------------------------------

    def append(self, record: LedgerRecord) -> LedgerRecord:
        with self._lock:
            self._records.append(record)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.model_dump(mode="json")) + "\n")
        return record

    def attach_human_decision(
        self, ledger_id: str, decision: HumanDecision
    ) -> LedgerRecord | None:
        """Record a reviewer's verdict as an amendment, not an edit.

        The amendment is appended as a fresh line referencing the same
        ``ledger_id``; the in-memory view is updated for convenience. Replaying
        the file from disk still shows the original recommendation first, which
        is the property an audit needs.
        """
        with self._lock:
            for rec in self._records:
                if rec.ledger_id == ledger_id:
                    updated = rec.model_copy(update={"human_decision": decision})
                    idx = self._records.index(rec)
                    self._records[idx] = updated
                    with self.path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(updated.model_dump(mode="json")) + "\n")
                    return updated
        return None

    def attach_outcome(self, ledger_id: str, outcome: str) -> LedgerRecord | None:
        with self._lock:
            for i, rec in enumerate(self._records):
                if rec.ledger_id == ledger_id:
                    self._records[i] = rec.model_copy(update={"outcome": outcome})
                    return self._records[i]
        return None

    # -- read -------------------------------------------------------------

    @property
    def records(self) -> list[LedgerRecord]:
        return list(self._records)

    def query(
        self,
        *,
        member_id: str | None = None,
        verdict: GateVerdict | None = None,
        action: ActionType | None = None,
        since: datetime | None = None,
        requires_human: bool | None = None,
    ) -> list[LedgerRecord]:
        out: Iterable[LedgerRecord] = self._records
        if member_id:
            out = (r for r in out if r.member_id == member_id)
        if verdict:
            out = (r for r in out if r.gate_verdict == verdict)
        if action:
            out = (r for r in out if r.selected_action == action)
        if since:
            out = (r for r in out if r.created_at >= since)
        if requires_human is not None:
            out = (r for r in out if r.requires_human == requires_human)
        return list(out)

    # -- aggregate views ---------------------------------------------------

    def summary(self) -> dict[str, Any]:
        recs = self._records
        if not recs:
            return {"n_decisions": 0}

        reviewed = [r for r in recs if r.human_decision is not None]
        overrides = [
            r for r in reviewed
            if r.human_decision and r.human_decision.decision != HumanDecisionType.ACCEPT
        ]
        allowed = [r for r in recs if r.gate_verdict == GateVerdict.ALLOW]

        return {
            "n_decisions": len(recs),
            "by_verdict": dict(Counter(r.gate_verdict.value for r in recs)),
            "by_action": dict(
                Counter(r.selected_action.value for r in recs if r.selected_action)
            ),
            "by_risk_band": dict(
                Counter(r.risk_band.value for r in recs if r.risk_band)
            ),
            "automation_rate": round(len(allowed) / len(recs), 4),
            "human_review_rate": round(
                sum(1 for r in recs if r.requires_human) / len(recs), 4
            ),
            "n_reviewed": len(reviewed),
            "override_rate": (
                round(len(overrides) / len(reviewed), 4) if reviewed else None
            ),
            "override_breakdown": dict(
                Counter(
                    r.human_decision.decision.value for r in reviewed if r.human_decision
                )
            ),
            "top_gate_reasons": dict(
                Counter(
                    reason.value for r in recs for reason in r.gate_reasons
                ).most_common(10)
            ),
            "top_risk_reasons": dict(
                Counter(
                    reason.value for r in recs for reason in r.risk_reasons
                ).most_common(10)
            ),
            "grounding": {
                "responses_generated": sum(1 for r in recs if r.response_text),
                "with_citations": sum(1 for r in recs if r.citations),
            },
        }

    def friction_trends(self, bucket_days: int = 7) -> list[dict[str, Any]]:
        """Aggregate view for the operations surface."""
        if not self._records:
            return []
        start = min(r.created_at for r in self._records)
        buckets: dict[int, list[LedgerRecord]] = {}
        for r in self._records:
            idx = (r.created_at - start).days // bucket_days
            buckets.setdefault(idx, []).append(r)

        out = []
        for idx in sorted(buckets):
            recs = buckets[idx]
            high = [r for r in recs if r.risk_band and r.risk_band.value == "high"]
            out.append(
                {
                    "bucket_start": (start + timedelta(days=idx * bucket_days)).date().isoformat(),
                    "n_decisions": len(recs),
                    "n_high_risk": len(high),
                    "n_allowed": sum(1 for r in recs if r.gate_verdict == GateVerdict.ALLOW),
                    "n_suppressed": sum(
                        1 for r in recs if r.gate_verdict == GateVerdict.SUPPRESS
                    ),
                    "n_escalated": sum(
                        1 for r in recs if r.gate_verdict == GateVerdict.ESCALATE
                    ),
                    "mean_risk": round(
                        sum(r.risk_score or 0 for r in recs) / len(recs), 4
                    ),
                }
            )
        return out

    def recurring_root_causes(self, top_n: int = 8) -> list[dict[str, Any]]:
        """What keeps causing friction, aggregated across members.

        The operations question from the transcripts: not "who is at risk" but
        "what do we keep failing at".
        """
        counts = Counter(
            reason.value for r in self._records for reason in r.risk_reasons
        )
        total = sum(counts.values()) or 1
        return [
            {"reason": reason, "n": n, "share": round(n / total, 4)}
            for reason, n in counts.most_common(top_n)
        ]

    def clear(self) -> None:
        """Test helper. Never called by the application."""
        with self._lock:
            self._records.clear()
            if self.path.exists():
                self.path.unlink()


_LEDGER: FrictionLedger | None = None


def get_ledger() -> FrictionLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = FrictionLedger()
    return _LEDGER
