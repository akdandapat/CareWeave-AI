"""Journey store.

The single most important invariant in the system lives here:

    Nothing dated after ``as_of`` is ever returned.

Every consumer -- feature builder, context engine, advocate copilot, evaluation
harness -- goes through this class. Centralising the time filter means leakage
is prevented in one auditable place rather than re-implemented (and eventually
mis-implemented) in five.

Storage is Parquet + JSONL loaded into memory. At this dataset size that is
faster than a database round trip and keeps the clone-and-run experience to a
single command. ``docs/architecture.md`` documents the Postgres swap.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from functools import cached_property
from pathlib import Path

import pandas as pd

from careweave.config import SETTINGS
from careweave.domain.enums import ClaimStatus, DocumentType, PAStatus
from careweave.domain.models import (
    Claim,
    CostSnapshot,
    Document,
    Drug,
    FormularyEntry,
    Interaction,
    JourneyEvent,
    Member,
    MemberContext,
    Pharmacy,
    Plan,
    Prescriber,
    Prescription,
    PriorAuthorization,
)


def _as_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    return datetime.fromisoformat(str(value))


class JourneyStore:
    def __init__(self, synthetic_dir: Path | None = None, documents_dir: Path | None = None):
        self.synthetic_dir = synthetic_dir or SETTINGS.paths.synthetic
        self.documents_dir = documents_dir or SETTINGS.paths.documents
        self._load()

    # -- loading ---------------------------------------------------------

    def _pq(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.synthetic_dir / f"{name}.parquet")

    @staticmethod
    def _rows(df: pd.DataFrame) -> list[dict]:
        """Parquet round-trips missing values as NaN; pydantic wants None."""
        return [
            {k: (None if (v is None or (isinstance(v, float) and pd.isna(v))) else v)
             for k, v in row.items()}
            for row in df.to_dict("records")
        ]

    def _load(self) -> None:
        self.members: dict[str, Member] = {
            r["member_id"]: Member(**r) for r in self._rows(self._pq("members"))
        }
        self.plans: dict[str, Plan] = {}
        for r in self._rows(self._pq("plans")):
            r["copay_by_tier"] = json.loads(r["copay_by_tier"])
            self.plans[r["plan_id"]] = Plan(**r)
        self.drugs: dict[str, Drug] = {
            r["drug_id"]: Drug(**r) for r in self._rows(self._pq("drugs"))
        }
        self.formulary: dict[str, FormularyEntry] = {
            r["drug_id"]: FormularyEntry(**r) for r in self._rows(self._pq("formulary"))
        }
        self.pharmacies: dict[str, Pharmacy] = {
            r["pharmacy_id"]: Pharmacy(**r) for r in self._rows(self._pq("pharmacies"))
        }
        self.prescribers: dict[str, Prescriber] = {
            r["prescriber_id"]: Prescriber(**r) for r in self._rows(self._pq("prescribers"))
        }

        self.prescriptions: dict[str, Prescription] = {}
        self._rx_by_member: dict[str, list[Prescription]] = defaultdict(list)
        for r in self._rows(self._pq("prescriptions")):
            rx = Prescription(**r)
            self.prescriptions[rx.rx_id] = rx
            self._rx_by_member[rx.member_id].append(rx)

        self.authorizations: dict[str, PriorAuthorization] = {}
        self._pa_by_member: dict[str, list[PriorAuthorization]] = defaultdict(list)
        for r in self._rows(self._pq("authorizations")):
            pa = PriorAuthorization(**r)
            self.authorizations[pa.pa_id] = pa
            self._pa_by_member[pa.member_id].append(pa)

        self.claims: dict[str, Claim] = {}
        self._claims_by_member: dict[str, list[Claim]] = defaultdict(list)
        for r in self._rows(self._pq("claims")):
            cl = Claim(**r)
            self.claims[cl.claim_id] = cl
            self._claims_by_member[cl.member_id].append(cl)

        self._interactions_by_member: dict[str, list[Interaction]] = defaultdict(list)
        for r in self._rows(self._pq("interactions")):
            it = Interaction(**r)
            self._interactions_by_member[it.member_id].append(it)

        self._events_by_member: dict[str, list[JourneyEvent]] = defaultdict(list)
        with (self.synthetic_dir / "events.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                ev = JourneyEvent(**json.loads(line))
                self._events_by_member[ev.member_id].append(ev)
        for evs in self._events_by_member.values():
            evs.sort(key=lambda e: e.occurred_at)

        self.member_documents: dict[str, Document] = {}
        self._docs_by_member: dict[str, list[Document]] = defaultdict(list)
        mdoc = self.documents_dir / "member_documents.jsonl"
        if mdoc.exists():
            with mdoc.open(encoding="utf-8") as fh:
                for line in fh:
                    doc = Document(**json.loads(line))
                    self.member_documents[doc.document_id] = doc
                    if doc.member_id:
                        self._docs_by_member[doc.member_id].append(doc)

        self.policy_documents: list[Document] = []
        pdoc = self.documents_dir / "policy_documents.jsonl"
        if pdoc.exists():
            with pdoc.open(encoding="utf-8") as fh:
                self.policy_documents = [Document(**json.loads(l)) for l in fh]

    # -- point-in-time accessors ----------------------------------------

    def events(
        self, member_id: str, as_of: datetime, *, lookback_days: int | None = None
    ) -> list[JourneyEvent]:
        """Events strictly at or before ``as_of``. The core invariant."""
        evs = self._events_by_member.get(member_id, [])
        floor = as_of - timedelta(days=lookback_days) if lookback_days else None
        return [
            e for e in evs
            if e.occurred_at <= as_of and (floor is None or e.occurred_at >= floor)
        ]

    def future_events(
        self, member_id: str, after: datetime, horizon_days: int
    ) -> list[JourneyEvent]:
        """Events in ``(after, after + horizon]``.

        Used *only* by the label builder and the evaluation harness. Any import
        of this method from feature code is a bug and is asserted against in
        tests/test_no_leakage.py.
        """
        until = after + timedelta(days=horizon_days)
        return [
            e for e in self._events_by_member.get(member_id, [])
            if after < e.occurred_at <= until
        ]

    def interactions(self, member_id: str, as_of: datetime) -> list[Interaction]:
        return [
            i for i in self._interactions_by_member.get(member_id, [])
            if _as_dt(i.occurred_on) <= as_of
        ]

    def claims_asof(self, member_id: str, as_of: datetime) -> list[Claim]:
        return [
            c for c in self._claims_by_member.get(member_id, [])
            if _as_dt(c.submitted_on) <= as_of
        ]

    def authorizations_asof(self, member_id: str, as_of: datetime) -> list[PriorAuthorization]:
        """Authorizations as they *appeared* at ``as_of``.

        Subtle and important: a PA decided after ``as_of`` must be reported as
        still pending, not with its eventual outcome. Rolling the status back is
        the difference between an honest feature and a time-travelling one.
        """
        out: list[PriorAuthorization] = []
        for pa in self._pa_by_member.get(member_id, []):
            if pa.requested_on is None or _as_dt(pa.requested_on) > as_of:
                continue
            if pa.decided_on is not None and _as_dt(pa.decided_on) > as_of:
                rolled = pa.model_copy(
                    update={"status": PAStatus.PENDING, "decided_on": None,
                            "denial_reason": None}
                )
                out.append(rolled)
            else:
                out.append(pa)
        return out

    def documents(self, member_id: str, as_of: datetime, *, limit: int = 20) -> list[Document]:
        docs = [
            d for d in self._docs_by_member.get(member_id, []) if d.created_at <= as_of
        ]
        docs.sort(key=lambda d: d.created_at, reverse=True)
        return docs[:limit]

    def prescriptions_asof(self, member_id: str, as_of: datetime) -> list[Prescription]:
        return [
            rx for rx in self._rx_by_member.get(member_id, [])
            if _as_dt(rx.written_on) <= as_of
        ]

    # -- assembled context ----------------------------------------------

    def member_context(self, member_id: str, as_of: datetime) -> MemberContext:
        """The 'member 360' the transcripts describe, assembled in one call.

        This is the concrete answer to the fragmentation problem: one function,
        one timestamp, every source joined.
        """
        member = self.members[member_id]
        plan = self.plans[member.plan_id]
        claims = self.claims_asof(member_id, as_of)
        interactions = self.interactions(member_id, as_of)
        auths = [
            pa for pa in self.authorizations_asof(member_id, as_of)
            if pa.status in (PAStatus.PENDING, PAStatus.PENDING_INFO,
                             PAStatus.REQUIRED_NOT_STARTED)
        ]

        recent_claims = sorted(claims, key=lambda c: c.submitted_on, reverse=True)[:15]
        cost_history = [
            CostSnapshot(
                fill_date=c.submitted_on,
                member_cost=c.member_cost or 0.0,
                deductible_phase=c.deductible_phase,
            )
            for c in sorted(claims, key=lambda c: c.submitted_on)
            if c.status == ClaimStatus.PAID and c.member_cost is not None
        ][-12:]

        last_contact = max((_as_dt(i.occurred_on) for i in interactions), default=None)
        return MemberContext(
            member_id=member_id,
            as_of=as_of,
            member=member,
            plan=plan,
            active_prescriptions=self.prescriptions_asof(member_id, as_of),
            open_authorizations=auths,
            recent_claims=recent_claims,
            recent_interactions=sorted(
                interactions, key=lambda i: i.occurred_on, reverse=True
            )[:10],
            cost_history=cost_history,
            days_since_last_contact=(
                (as_of - last_contact).total_seconds() / 86400.0 if last_contact else None
            ),
            contact_count_30d=sum(
                1 for i in interactions if (as_of - _as_dt(i.occurred_on)).days <= 30
            ),
            rejection_count_90d=sum(
                1 for c in claims
                if c.status == ClaimStatus.REJECTED
                and (as_of - _as_dt(c.submitted_on)).days <= 90
            ),
        )

    # -- convenience -----------------------------------------------------

    def drug_for_rx(self, rx_id: str) -> Drug | None:
        rx = self.prescriptions.get(rx_id)
        return self.drugs.get(rx.drug_id) if rx else None

    @cached_property
    def member_ids(self) -> list[str]:
        return sorted(self.members)

    @cached_property
    def window(self) -> tuple[datetime, datetime]:
        lo = min(e.occurred_at for evs in self._events_by_member.values() for e in evs)
        hi = max(e.occurred_at for evs in self._events_by_member.values() for e in evs)
        return lo, hi

    def policy_docs_of_type(self, *types: DocumentType) -> list[Document]:
        wanted = set(types)
        return [d for d in self.policy_documents if d.document_type in wanted]


_STORE: JourneyStore | None = None


def get_store() -> JourneyStore:
    global _STORE
    if _STORE is None:
        _STORE = JourneyStore()
    return _STORE
