"""Feature and label construction for the friction risk model.

Two rules, both enforced rather than documented:

1. **Features read only from ``store.events(..., as_of)`` and its siblings.**
   No feature function may call ``store.future_events``. A test asserts this by
   source inspection.

2. **Latent traits are unreachable.** They live in a separately named file that
   the store never loads. A model that wants them has to work for them, through
   behavioural proxies -- which is the honest version of the problem.

Observation points are the moments at which the engine would actually run in
production: a refill comes due, an authorization is requested, a claim rejects, a
cost changes, a member makes contact. Plus routine sampled points, so the model
sees quiet periods and is not trained only on crises.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from careweave.config import SETTINGS
from careweave.data.store import JourneyStore, _as_dt
from careweave.domain.enums import ClaimStatus, EventType, PAStatus, RejectCode

#: Events that *are* friction from the member's point of view -- the member had
#: to do something, or gave up. Deliberately not "claim rejected", which is an
#: operational precursor and is used as a feature instead.
LABEL_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.MEMBER_CONTACTED_SUPPORT,
        EventType.CASE_ESCALATED,
        EventType.RX_ABANDONED_AT_COUNTER,
    }
)

#: Event types that trigger an engine invocation in production.
TRIGGER_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.REFILL_DUE,
        EventType.PA_REQUESTED,
        EventType.PA_INFO_REQUESTED,
        EventType.CLAIM_REJECTED,
        EventType.COST_CHANGED,
        EventType.MEMBER_CONTACTED_SUPPORT,
        EventType.RX_WRITTEN,
    }
)

_REJECT_FEATURE_CODES = [
    RejectCode.PA_REQUIRED,
    RejectCode.REFILL_TOO_SOON,
    RejectCode.NOT_ON_FORMULARY,
    RejectCode.QUANTITY_LIMIT_EXCEEDED,
    RejectCode.PHARMACY_OUT_OF_NETWORK,
]


@dataclass(frozen=True)
class ObservationPoint:
    member_id: str
    as_of: datetime
    trigger: EventType
    rx_id: str | None


def build_observation_points(
    store: JourneyStore, *, seed: int, routine_per_member: int = 3
) -> list[ObservationPoint]:
    rng = random.Random(seed)
    lo, hi = store.window
    # leave room at the end so every point has a full label horizon
    label_cutoff = hi - timedelta(days=SETTINGS.ml.horizon_days)
    points: list[ObservationPoint] = []

    for mid in store.member_ids:
        evs = store._events_by_member.get(mid, [])
        triggered = [
            e for e in evs
            if e.event_type in TRIGGER_EVENT_TYPES and e.occurred_at <= label_cutoff
        ]
        # cap per member so heavy utilisers do not dominate the training set
        if len(triggered) > 12:
            triggered = rng.sample(triggered, 12)
        for e in triggered:
            points.append(
                ObservationPoint(mid, e.occurred_at, e.event_type, e.rx_id)
            )

        span_days = max(1, (label_cutoff - lo).days)
        for _ in range(routine_per_member):
            when = lo + timedelta(days=rng.randint(30, span_days))
            points.append(ObservationPoint(mid, when, EventType.MEMBER_VIEWED_PORTAL, None))

    points.sort(key=lambda p: p.as_of)
    return points


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def build_features(store: JourneyStore, point: ObservationPoint) -> dict[str, Any]:
    """All features for one observation point. Reads nothing after ``as_of``."""
    mid, t = point.member_id, point.as_of
    member = store.members[mid]
    plan = store.plans[member.plan_id]

    events = store.events(mid, t)
    ev90 = [e for e in events if (t - e.occurred_at).days <= 90]
    ev30 = [e for e in events if (t - e.occurred_at).days <= 30]
    ev180 = [e for e in events if (t - e.occurred_at).days <= 180]

    def count(evs, *types: EventType) -> int:
        s = set(types)
        return sum(1 for e in evs if e.event_type in s)

    claims = store.claims_asof(mid, t)
    claims90 = [c for c in claims if (t - _as_dt(c.submitted_on)).days <= 90]
    rejected90 = [c for c in claims90 if c.status == ClaimStatus.REJECTED]
    paid = [c for c in claims if c.status == ClaimStatus.PAID and c.member_cost is not None]
    paid_sorted = sorted(paid, key=lambda c: c.submitted_on)

    interactions = store.interactions(mid, t)
    int90 = [i for i in interactions if (t - _as_dt(i.occurred_on)).days <= 90]
    int30 = [i for i in interactions if (t - _as_dt(i.occurred_on)).days <= 30]

    auths = store.authorizations_asof(mid, t)
    open_auths = [
        a for a in auths if a.status in (PAStatus.PENDING, PAStatus.PENDING_INFO)
    ]
    pa_pending_days = [
        (t - _as_dt(a.requested_on)).days for a in open_auths if a.requested_on
    ]

    rxs = store.prescriptions_asof(mid, t)
    drugs = [store.drugs[rx.drug_id] for rx in rxs]
    fes = [store.formulary[rx.drug_id] for rx in rxs]

    # -- refill runway: days of supply remaining on the most recent fill ----
    last_fill = None
    for e in reversed(events):
        if e.event_type == EventType.RX_FILLED:
            last_fill = e
            break
    runway = 999.0
    if last_fill is not None and last_fill.rx_id in store.prescriptions:
        ds = store.prescriptions[last_fill.rx_id].days_supply
        runway = ds - (t - last_fill.occurred_at).days

    # -- cost trajectory ---------------------------------------------------
    last_cost = paid_sorted[-1].member_cost if paid_sorted else 0.0
    prev_cost = paid_sorted[-2].member_cost if len(paid_sorted) > 1 else last_cost
    cost_delta = (last_cost or 0.0) - (prev_cost or 0.0)

    last_reject = max(rejected90, key=lambda c: c.submitted_on, default=None)
    last_contact = max((_as_dt(i.occurred_on) for i in interactions), default=None)

    f: dict[str, Any] = {
        # identity (dropped before training, kept for joins and audit)
        "member_id": mid,
        "as_of": t,
        "trigger_event": point.trigger.value,
        # demographics and plan
        "age": (t.date() - member.birth_date).days / 365.25,
        "tenure_days": (t.date() - member.enrolled_on).days,
        "is_caregiver": int(member.is_caregiver),
        "plan_deductible": plan.deductible,
        "plan_specialty_coins": plan.specialty_coinsurance,
        "pref_channel_digital": int(
            member.preferred_channel.value in {"app_push", "portal", "email"}
        ),
        # medication profile
        "n_active_rx": len(rxs),
        "n_specialty_rx": sum(1 for d in drugs if d.is_specialty),
        "any_pa_drug": int(any(fe.pa_required for fe in fes)),
        "any_step_therapy": int(any(fe.step_therapy_required for fe in fes)),
        "max_list_price": max((d.list_price_30d for d in drugs), default=0.0),
        "frac_generic": _safe_div(sum(1 for d in drugs if d.generic_available), len(drugs)),
        # prior authorization
        "n_open_pa": len(open_auths),
        "max_pa_pending_days": max(pa_pending_days, default=0),
        "any_pa_pending_info": int(
            any(a.status == PAStatus.PENDING_INFO for a in open_auths)
        ),
        "n_pa_denied_180d": count(ev180, EventType.PA_DENIED),
        "n_pa_approved_180d": count(ev180, EventType.PA_APPROVED),
        "n_pa_info_req_90d": count(ev90, EventType.PA_INFO_REQUESTED),
        # claims
        "n_claims_90d": len(claims90),
        "n_rejected_90d": len(rejected90),
        "reject_rate_90d": _safe_div(len(rejected90), len(claims90)),
        "days_since_last_rejection": (
            (t - _as_dt(last_reject.submitted_on)).days if last_reject else 999
        ),
        # cost
        "last_member_cost": last_cost or 0.0,
        "cost_delta": cost_delta,
        "cost_delta_pct": _safe_div(cost_delta, max(prev_cost or 1.0, 1.0)),
        "max_cost_180d": max(
            (c.member_cost or 0.0 for c in paid
             if (t - _as_dt(c.submitted_on)).days <= 180),
            default=0.0,
        ),
        "n_cost_changes_90d": count(ev90, EventType.COST_CHANGED),
        "in_deductible_phase": int(
            bool(paid_sorted) and paid_sorted[-1].deductible_phase is not None
            and paid_sorted[-1].deductible_phase.value == "deductible"
        ),
        # contact history -- behavioural proxies for the latent traits
        "n_contacts_30d": len(int30),
        "n_contacts_90d": len(int90),
        "days_since_last_contact": (
            (t - last_contact).days if last_contact else 999
        ),
        "n_unresolved_90d": sum(1 for i in int90 if not i.resolved),
        "n_escalations_180d": count(ev180, EventType.CASE_ESCALATED),
        "n_portal_views_30d": count(ev30, EventType.MEMBER_VIEWED_PORTAL),
        "n_abandonments_180d": count(ev180, EventType.RX_ABANDONED_AT_COUNTER),
        "n_fills_90d": count(ev90, EventType.RX_FILLED),
        # timing
        "refill_runway_days": runway,
        "runway_under_7": int(runway < 7),
        "pa_pending_and_short_runway": int(bool(open_auths) and runway < 10),
        "journey_event_count": len(events),
    }

    for code in _REJECT_FEATURE_CODES:
        key = f"reject_{code.name.lower()}_90d"
        f[key] = sum(1 for c in rejected90 if c.reject_code == code)

    return f


def build_label(store: JourneyStore, point: ObservationPoint, horizon_days: int) -> int:
    """1 if a member-visible friction event occurs in the horizon after ``as_of``.

    This is the only place ``future_events`` is called outside evaluation.
    """
    fut = store.future_events(point.member_id, point.as_of, horizon_days)
    return int(any(e.event_type in LABEL_EVENT_TYPES for e in fut))


def build_dataset(
    store: JourneyStore, *, seed: int | None = None, horizon_days: int | None = None
) -> pd.DataFrame:
    seed = seed if seed is not None else SETTINGS.generation.seed
    horizon = horizon_days or SETTINGS.ml.horizon_days
    points = build_observation_points(store, seed=seed)

    rows: list[dict[str, Any]] = []
    for p in points:
        row = build_features(store, p)
        row["label"] = build_label(store, p, horizon)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values("as_of").reset_index(drop=True)


#: Columns excluded from the model matrix. Identity and time only -- everything
#: else is a genuine feature.
NON_FEATURE_COLUMNS = ["member_id", "as_of", "label"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
