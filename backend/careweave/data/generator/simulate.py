"""Journey simulator.

The realism target is not "looks plausible in a table viewer". It is: *does a
model trained on this have to work for its accuracy?* Three design choices make
that true.

1. **Latent traits are unobservable.** Health literacy, contact propensity and
   persistence drive behaviour but never appear as features. The model must
   infer from behavioural proxies, exactly as it would in reality.

2. **Confounders are deliberate.** Digitally engaged members self-serve, so they
   contact support *less* at the same underlying risk. A model that keys only on
   operational severity will be miscalibrated for them.

3. **Friction is a hazard, not a rule.** Every friction event is drawn from a
   probability, so no feature is a giveaway and irreducible noise sets a real
   ceiling on achievable AUC.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

from careweave.domain.enums import (
    Channel,
    ClaimStatus,
    DeductiblePhase,
    DrugTier,
    EventType,
    MemberIntent,
    PAStatus,
    PharmacyType,
    RejectCode,
)
from careweave.domain.models import (
    Claim,
    Drug,
    FormularyEntry,
    Interaction,
    JourneyEvent,
    Member,
    Pharmacy,
    Plan,
    Prescriber,
    Prescription,
    PriorAuthorization,
)

from .catalogs import CHRONIC_CLASSES, CITIES, STATES

_GIVEN = [
    "Marcus", "Priya", "Delia", "Tomas", "Aisha", "Grant", "Yolanda", "Ravi",
    "Nadia", "Owen", "Camille", "Felix", "Imani", "Hugo", "Sofia", "Dean",
    "Lucia", "Amara", "Noel", "Rosa", "Bennett", "Talia", "Quinn", "Marisol",
]
_FAMILY = [
    "Vance", "Okonkwo", "Bergstrom", "Salazar", "Whitcomb", "Nayak", "Duarte",
    "Ellison", "Farouk", "Holloway", "Ibrahim", "Jessup", "Kaminski", "Lombard",
    "Moreau", "Nunes", "Ortega", "Petrov", "Quiroga", "Rennick", "Sandoval",
]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Traits:
    """Latent member traits. Never exposed as model features."""

    health_literacy: float
    cost_sensitivity: float
    digital_engagement: float
    contact_propensity: float
    persistence: float

    @classmethod
    def draw(cls, rng: random.Random) -> "Traits":
        return cls(
            health_literacy=rng.betavariate(3, 2),
            cost_sensitivity=rng.betavariate(2, 2),
            digital_engagement=rng.betavariate(2.5, 2.5),
            contact_propensity=rng.betavariate(2, 3),
            persistence=rng.betavariate(3, 2),
        )


@dataclass
class SimOutput:
    members: list[Member] = field(default_factory=list)
    prescriptions: list[Prescription] = field(default_factory=list)
    authorizations: list[PriorAuthorization] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    interactions: list[Interaction] = field(default_factory=list)
    events: list[JourneyEvent] = field(default_factory=list)
    #: member_id -> traits, retained only for generator diagnostics, never used
    #: as ML features. Written to a separate file and excluded from the feature
    #: builder by an explicit assertion in tests.
    traits: dict[str, Traits] = field(default_factory=dict)


class JourneySimulator:
    def __init__(
        self,
        *,
        rng: random.Random,
        drugs: list[Drug],
        plans: list[Plan],
        formulary: list[FormularyEntry],
        pharmacies: list[Pharmacy],
        prescribers: list[Prescriber],
        start: date,
        end: date,
    ) -> None:
        self.rng = rng
        self.drugs = {d.drug_id: d for d in drugs}
        self.plans = {p.plan_id: p for p in plans}
        self.formulary = {f.drug_id: f for f in formulary}
        self.pharmacies = {p.pharmacy_id: p for p in pharmacies}
        self.prescribers = prescribers
        self.start = start
        self.end = end

        self.retail_by_state: dict[str, list[Pharmacy]] = {}
        for ph in pharmacies:
            if ph.pharmacy_type in (
                PharmacyType.RETAIL_IN_NETWORK,
                PharmacyType.RETAIL_OUT_OF_NETWORK,
            ):
                self.retail_by_state.setdefault(ph.state, []).append(ph)

        self._counters: dict[str, int] = {}

    # -- id helpers -------------------------------------------------------

    def _nid(self, prefix: str, width: int = 6) -> str:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}-{self._counters[prefix]:0{width}d}"

    def _dt(self, d: date, hour: int | None = None) -> datetime:
        h = hour if hour is not None else self.rng.randint(8, 19)
        return datetime.combine(d, time(hour=h, minute=self.rng.randint(0, 59)))

    # -- population -------------------------------------------------------

    def run(self, n_members: int) -> SimOutput:
        out = SimOutput()
        plan_ids = list(self.plans)
        plan_weights = [0.4, 0.4, 0.2]

        for i in range(n_members):
            member, traits = self._make_member(i, plan_ids, plan_weights)
            out.members.append(member)
            out.traits[member.member_id] = traits
            self._simulate_member(member, traits, out)

        out.events.sort(key=lambda e: (e.member_id, e.occurred_at))
        return out

    def _make_member(
        self, i: int, plan_ids: list[str], plan_weights: list[float]
    ) -> tuple[Member, Traits]:
        rng = self.rng
        state = rng.choice(STATES)
        birth_year = rng.randint(1945, 2004)
        enrolled = self.start - timedelta(days=rng.randint(30, 900))
        traits = Traits.draw(rng)
        # channel preference correlates with digital engagement, which is a
        # realistic confounder for how outreach lands
        if traits.digital_engagement > 0.65:
            channel = rng.choice([Channel.APP_PUSH, Channel.PORTAL, Channel.EMAIL])
        elif traits.digital_engagement > 0.35:
            channel = rng.choice([Channel.EMAIL, Channel.SMS, Channel.PORTAL])
        else:
            channel = rng.choice([Channel.PHONE, Channel.MAIL, Channel.SMS])

        member = Member(
            member_id=f"MBR-{i + 1:06d}",
            given_name=rng.choice(_GIVEN),
            family_name=rng.choice(_FAMILY),
            birth_date=date(birth_year, rng.randint(1, 12), rng.randint(1, 28)),
            state=state,
            plan_id=rng.choices(plan_ids, weights=plan_weights, k=1)[0],
            enrolled_on=enrolled,
            preferred_channel=channel,
            is_caregiver=rng.random() < 0.18,
        )
        return member, traits

    # -- per-member simulation -------------------------------------------

    def _simulate_member(self, member: Member, traits: Traits, out: SimOutput) -> None:
        rng = self.rng
        plan = self.plans[member.plan_id]

        n_drugs = rng.choices([1, 2, 3, 4], weights=[0.32, 0.33, 0.22, 0.13], k=1)[0]
        chosen = rng.sample(list(self.drugs.values()), k=n_drugs)

        # per-calendar-year accumulated plan-covered spend, drives deductible phase
        accum: dict[int, float] = {}
        # rolling record of this member's friction, used by the hazard model
        recent_contacts: list[datetime] = []
        recent_rejections: list[date] = []
        abandonment_count = 0

        home_pharmacies = self.retail_by_state.get(member.state, [])
        default_pharmacy = (
            rng.choice(home_pharmacies) if home_pharmacies else self.pharmacies["PHM-MAIL"]
        )

        for drug in chosen:
            fe = self.formulary[drug.drug_id]
            prescriber = rng.choice(self.prescribers)
            is_chronic = drug.therapeutic_class in CHRONIC_CLASSES

            pharmacy = (
                self.pharmacies["PHM-SPEC"]
                if drug.is_specialty
                else (
                    self.pharmacies["PHM-MAIL"]
                    if rng.random() < 0.18 * (0.5 + traits.digital_engagement)
                    else default_pharmacy
                )
            )

            days_supply = 30 if drug.is_specialty else rng.choice([30, 30, 30, 90])
            written = self.start + timedelta(
                days=rng.randint(0, max(1, (self.end - self.start).days // 3))
            )
            n_cycles = (
                max(1, min(14, (self.end - written).days // days_supply))
                if is_chronic
                else rng.choice([1, 1, 2])
            )

            rx = Prescription(
                rx_id=self._nid("RX"),
                member_id=member.member_id,
                drug_id=drug.drug_id,
                prescriber_id=prescriber.prescriber_id,
                pharmacy_id=pharmacy.pharmacy_id,
                written_on=written,
                days_supply=days_supply,
                quantity=days_supply,
                refills_authorized=max(0, n_cycles - 1),
                is_new_to_therapy=True,
            )
            out.prescriptions.append(rx)
            out.events.append(
                self._event(member, EventType.RX_WRITTEN, written, rx_id=rx.rx_id,
                            payload={"drug": drug.name, "days_supply": days_supply})
            )

            pa_approved_until: date | None = None
            last_member_cost: float | None = None
            cycle_date = written

            for cycle in range(n_cycles):
                if cycle_date > self.end:
                    break
                self._simulate_fill_cycle(
                    member=member,
                    traits=traits,
                    plan=plan,
                    drug=drug,
                    fe=fe,
                    rx=rx,
                    prescriber=prescriber,
                    pharmacy=pharmacy,
                    cycle=cycle,
                    cycle_date=cycle_date,
                    accum=accum,
                    state=_CycleState(
                        pa_approved_until=pa_approved_until,
                        last_member_cost=last_member_cost,
                        recent_contacts=recent_contacts,
                        recent_rejections=recent_rejections,
                        abandonment_count=abandonment_count,
                    ),
                    out=out,
                )
                # pull mutated state back out
                st = self._last_state
                pa_approved_until = st.pa_approved_until
                last_member_cost = st.last_member_cost
                abandonment_count = st.abandonment_count

                jitter = rng.randint(-3, 6)
                cycle_date = cycle_date + timedelta(days=days_supply + jitter)

    # ------------------------------------------------------------------

    def _simulate_fill_cycle(
        self,
        *,
        member: Member,
        traits: Traits,
        plan: Plan,
        drug: Drug,
        fe: FormularyEntry,
        rx: Prescription,
        prescriber: Prescriber,
        pharmacy: Pharmacy,
        cycle: int,
        cycle_date: date,
        accum: dict[int, float],
        state: "_CycleState",
        out: SimOutput,
    ) -> None:
        rng = self.rng
        self._last_state = state

        if cycle > 0:
            out.events.append(
                self._event(member, EventType.REFILL_DUE, cycle_date, rx_id=rx.rx_id,
                            payload={"drug": drug.name})
            )

        needs_pa = fe.pa_required and (
            state.pa_approved_until is None or state.pa_approved_until < cycle_date
        )

        # ---------------- prior authorization path ----------------
        pa: PriorAuthorization | None = None
        pa_resolved_on: date | None = None
        if needs_pa:
            out.events.append(
                self._event(member, EventType.PA_REQUIRED_FLAGGED, cycle_date, rx_id=rx.rx_id,
                            payload={"drug": drug.name})
            )
            requested_on = cycle_date + timedelta(days=rng.randint(0, 2))
            # turnaround is driven by prescriber responsiveness and specialty status
            base_days = 3 + (5 if drug.is_specialty else 0)
            spread = 14 * (1.0 - prescriber.responsiveness)
            turnaround = max(1, int(rng.gauss(base_days + spread, 3)))
            needs_more_info = rng.random() < (0.35 - 0.20 * prescriber.responsiveness)

            pa = PriorAuthorization(
                pa_id=self._nid("PA"),
                rx_id=rx.rx_id,
                member_id=member.member_id,
                status=PAStatus.PENDING,
                requested_on=requested_on,
                decided_on=None,
            )
            out.events.append(
                self._event(member, EventType.PA_REQUESTED, requested_on, rx_id=rx.rx_id,
                            pa_id=pa.pa_id, payload={"drug": drug.name})
            )

            if needs_more_info:
                info_on = requested_on + timedelta(days=rng.randint(2, 6))
                pa.status = PAStatus.PENDING_INFO
                pa.missing_info = rng.choice(
                    [
                        "recent lab results",
                        "documentation of prior therapy",
                        "chart notes supporting medical necessity",
                        "corrected prescriber NPI",
                    ]
                )
                out.events.append(
                    self._event(member, EventType.PA_INFO_REQUESTED, info_on, rx_id=rx.rx_id,
                                pa_id=pa.pa_id, payload={"missing_info": pa.missing_info})
                )
                turnaround += int(9 * (1.0 - prescriber.responsiveness)) + rng.randint(2, 8)
                if rng.random() < prescriber.responsiveness:
                    out.events.append(
                        self._event(
                            member, EventType.PA_INFO_RECEIVED,
                            info_on + timedelta(days=rng.randint(2, 10)),
                            rx_id=rx.rx_id, pa_id=pa.pa_id,
                        )
                    )

            decided_on = requested_on + timedelta(days=turnaround)
            approved = rng.random() < (0.86 if not fe.step_therapy_required else 0.62)
            pa.decided_on = decided_on
            if approved:
                pa.status = PAStatus.APPROVED
                state.pa_approved_until = decided_on + timedelta(days=365)
                out.events.append(
                    self._event(member, EventType.PA_APPROVED, decided_on, rx_id=rx.rx_id,
                                pa_id=pa.pa_id)
                )
            else:
                pa.status = PAStatus.DENIED
                pa.denial_reason = (
                    "step therapy requirement not met"
                    if fe.step_therapy_required
                    else "submitted documentation did not establish medical necessity"
                )
                out.events.append(
                    self._event(member, EventType.PA_DENIED, decided_on, rx_id=rx.rx_id,
                                pa_id=pa.pa_id, payload={"reason": pa.denial_reason})
                )
            out.authorizations.append(pa)
            pa_resolved_on = decided_on

            # the pharmacy attempt that triggered the PA gets rejected
            self._emit_claim(
                member, rx, pharmacy, cycle_date, ClaimStatus.REJECTED,
                RejectCode.PA_REQUIRED, None, None, None, out
            )
            state.recent_rejections.append(cycle_date)
            self._maybe_friction(
                member, traits, cycle_date, rx, drug, state, out,
                driver_pa_pending_days=0,
                driver_rejection=True,
                driver_cost_delta=0.0,
                driver_specialty=drug.is_specialty,
                driver_new_to_therapy=(cycle == 0),
                intent=MemberIntent.AUTHORIZATION_STATUS,
                pa_pending_window=(cycle_date, pa_resolved_on),
                turnaround=turnaround,
            )
            if pa.status == PAStatus.DENIED:
                return
            fill_date = decided_on + timedelta(days=rng.randint(0, 4))
        else:
            fill_date = cycle_date

        if fill_date > self.end:
            return

        # ---------------- non-PA rejection paths ----------------
        reject: RejectCode | None = None
        if not fe.covered:
            reject = RejectCode.NOT_ON_FORMULARY
        elif pharmacy.pharmacy_type == PharmacyType.RETAIL_OUT_OF_NETWORK and rng.random() < 0.45:
            reject = RejectCode.PHARMACY_OUT_OF_NETWORK
        elif cycle > 0 and rng.random() < 0.07:
            reject = RejectCode.REFILL_TOO_SOON
        elif fe.quantity_limit_30d and rx.quantity > fe.quantity_limit_30d and rng.random() < 0.5:
            reject = RejectCode.QUANTITY_LIMIT_EXCEEDED
        elif fe.step_therapy_required and not needs_pa and rng.random() < 0.05:
            reject = RejectCode.STEP_THERAPY_REQUIRED
        elif rng.random() < 0.015:
            reject = RejectCode.MISSING_PRESCRIBER_INFO

        if reject is not None:
            self._emit_claim(
                member, rx, pharmacy, fill_date, ClaimStatus.REJECTED,
                reject, None, None, None, out
            )
            state.recent_rejections.append(fill_date)
            self._maybe_friction(
                member, traits, fill_date, rx, drug, state, out,
                driver_pa_pending_days=0,
                driver_rejection=True,
                driver_cost_delta=0.0,
                driver_specialty=drug.is_specialty,
                driver_new_to_therapy=(cycle == 0),
                intent=MemberIntent.CLAIM_REJECTED_WHY,
            )
            return

        # ---------------- paid claim + member cost ----------------
        year = fill_date.year
        spent = accum.get(year, 0.0)
        phase = (
            DeductiblePhase.DEDUCTIBLE if spent < plan.deductible else DeductiblePhase.INITIAL_COVERAGE
        )
        gross = drug.list_price_30d * (rx.days_supply / 30.0)

        if phase == DeductiblePhase.DEDUCTIBLE:
            member_cost = min(gross, plan.deductible - spent)
            plan_paid = gross - member_cost
        elif fe.tier == DrugTier.SPECIALTY:
            member_cost = gross * plan.specialty_coinsurance
            plan_paid = gross - member_cost
        else:
            member_cost = plan.copay_by_tier[fe.tier.value] * (rx.days_supply / 30.0)
            member_cost = min(member_cost, gross)
            plan_paid = gross - member_cost

        if pharmacy.pharmacy_type == PharmacyType.MAIL_ORDER:
            member_cost *= 1.0 - plan.mail_order_discount
        elif pharmacy.pharmacy_type == PharmacyType.RETAIL_OUT_OF_NETWORK:
            member_cost *= 1.0 + plan.out_of_network_penalty

        member_cost = round(member_cost, 2)
        accum[year] = spent + plan_paid

        prev_cost = state.last_member_cost
        cost_delta = 0.0 if prev_cost is None else member_cost - prev_cost

        self._emit_claim(
            member, rx, pharmacy, fill_date, ClaimStatus.PAID, None,
            member_cost, round(plan_paid, 2), phase, out
        )

        if prev_cost is not None and abs(cost_delta) >= max(8.0, 0.25 * max(prev_cost, 1.0)):
            out.events.append(
                self._event(member, EventType.COST_CHANGED, fill_date, rx_id=rx.rx_id,
                            payload={"previous": prev_cost, "current": member_cost,
                                     "delta": round(cost_delta, 2), "phase": phase.value})
            )
        if prev_cost is not None and phase != state.last_phase and state.last_phase is not None:
            out.events.append(
                self._event(member, EventType.DEDUCTIBLE_PHASE_CHANGED, fill_date,
                            rx_id=rx.rx_id, payload={"from": state.last_phase.value,
                                                     "to": phase.value})
            )
        state.last_phase = phase

        # ---------------- abandonment at the counter ----------------
        # cost shock plus low persistence is the abandonment driver
        # log scaling: cost shock has diminishing marginal effect
        shock = math.log1p(member_cost) / 3.0
        p_abandon = _sigmoid(
            -5.0 + 1.00 * shock + 1.30 * (1.0 - traits.persistence) + 0.75 * traits.cost_sensitivity
        )
        if rng.random() < p_abandon:
            out.events.append(
                self._event(member, EventType.RX_ABANDONED_AT_COUNTER, fill_date, rx_id=rx.rx_id,
                            payload={"member_cost": member_cost})
            )
            state.abandonment_count += 1
        else:
            out.events.append(
                self._event(member, EventType.RX_FILLED, fill_date, rx_id=rx.rx_id,
                            payload={"member_cost": member_cost, "drug": drug.name})
            )

        self._maybe_friction(
            member, traits, fill_date, rx, drug, state, out,
            driver_pa_pending_days=0,
            driver_rejection=False,
            driver_cost_delta=cost_delta,
            driver_specialty=drug.is_specialty,
            driver_new_to_therapy=(cycle == 0),
            intent=(
                MemberIntent.WHY_DID_COST_CHANGE
                if cost_delta > 10
                else MemberIntent.WHAT_HAPPENS_NEXT
            ),
        )

        state.last_member_cost = member_cost
        rx.refills_used = cycle

    # ------------------------------------------------------------------

    def _emit_claim(
        self, member, rx, pharmacy, on: date, status: ClaimStatus,
        reject: RejectCode | None, member_cost: float | None,
        plan_paid: float | None, phase: DeductiblePhase | None, out: SimOutput,
    ) -> None:
        claim = Claim(
            claim_id=self._nid("CLM"),
            rx_id=rx.rx_id,
            member_id=member.member_id,
            pharmacy_id=pharmacy.pharmacy_id,
            submitted_on=on,
            status=status,
            reject_code=reject,
            member_cost=member_cost,
            plan_paid=plan_paid,
            deductible_phase=phase,
        )
        out.claims.append(claim)
        out.events.append(
            self._event(member, EventType.CLAIM_SUBMITTED, on, rx_id=rx.rx_id,
                        claim_id=claim.claim_id)
        )
        if status == ClaimStatus.REJECTED:
            out.events.append(
                self._event(member, EventType.CLAIM_REJECTED, on, rx_id=rx.rx_id,
                            claim_id=claim.claim_id,
                            payload={"reject_code": reject.value if reject else None})
            )
        else:
            out.events.append(
                self._event(member, EventType.CLAIM_PAID, on, rx_id=rx.rx_id,
                            claim_id=claim.claim_id,
                            payload={"member_cost": member_cost})
            )

    def _maybe_friction(
        self,
        member: Member,
        traits: Traits,
        when: date,
        rx: Prescription,
        drug: Drug,
        state: "_CycleState",
        out: SimOutput,
        *,
        driver_pa_pending_days: int,
        driver_rejection: bool,
        driver_cost_delta: float,
        driver_specialty: bool,
        driver_new_to_therapy: bool,
        intent: MemberIntent,
        pa_pending_window: tuple[date, date | None] | None = None,
        turnaround: int = 0,
    ) -> None:
        """Draw friction events from a hazard, not a rule.

        The linear predictor mixes *observable* operational severity with
        *latent* member traits. Because the traits are never exposed to the
        model, a meaningful and irreducible error floor exists.
        """
        rng = self.rng

        recent_contact_30d = sum(
            1 for c in state.recent_contacts if 0 <= (when - c.date()).days <= 30
        )
        recent_reject_90d = sum(
            1 for r in state.recent_rejections if 0 <= (when - r).days <= 90
        )

        z = -4.15
        z += 1.45 if driver_rejection else 0.0
        z += 0.85 * min(2.0, max(0.0, driver_cost_delta) / 40.0)
        z += 0.55 if driver_specialty else 0.0
        z += 0.40 if driver_new_to_therapy else 0.0
        z += 0.30 * min(4, recent_reject_90d)
        z += 0.42 * min(3, recent_contact_30d)              # prior contact begets contact
        z += 0.28 * min(3, state.abandonment_count)
        # latent traits
        z += 1.30 * traits.contact_propensity
        z += 0.95 * (1.0 - traits.health_literacy)
        z -= 0.85 * traits.digital_engagement       # self-service confounder
        z += 0.55 * traits.cost_sensitivity * min(1.0, max(0.0, driver_cost_delta) / 30.0)
        z += rng.gauss(0.0, 0.45)                   # irreducible noise

        if rng.random() >= _sigmoid(z):
            return

        # contact lands somewhere in the days after the trigger; if a PA is
        # pending, it clusters as the refill runway shortens
        if pa_pending_window and turnaround > 4:
            offset = rng.randint(3, max(4, min(turnaround, 21)))
        else:
            offset = rng.randint(0, 9)
        occurred = self._dt(when + timedelta(days=offset))
        if occurred.date() > self.end:
            return

        # resolution depends on how tangled the situation is
        p_resolve = _sigmoid(
            1.35 - 0.7 * float(driver_rejection) - 0.5 * float(driver_specialty)
            + 0.8 * traits.health_literacy - 0.35 * recent_contact_30d
        )
        resolved = rng.random() < p_resolve
        escalated = (not resolved) and rng.random() < 0.42

        interaction = Interaction(
            interaction_id=self._nid("INT"),
            member_id=member.member_id,
            occurred_on=occurred,
            channel=(
                Channel.PHONE if traits.digital_engagement < 0.4 else member.preferred_channel
            ),
            intent=intent,
            resolved=resolved,
            escalated=escalated,
        )
        out.interactions.append(interaction)
        out.events.append(
            self._event(member, EventType.MEMBER_CONTACTED_SUPPORT, occurred.date(),
                        rx_id=rx.rx_id, at=occurred,
                        payload={"intent": intent.value, "resolved": resolved,
                                 "channel": interaction.channel.value})
        )
        state.recent_contacts.append(occurred)

        if escalated:
            esc_at = occurred + timedelta(hours=rng.randint(2, 72))
            if esc_at.date() <= self.end:
                out.events.append(
                    self._event(member, EventType.CASE_ESCALATED, esc_at.date(),
                                rx_id=rx.rx_id, at=esc_at)
                )
        if resolved:
            res_at = occurred + timedelta(hours=rng.randint(1, 48))
            if res_at.date() <= self.end:
                out.events.append(
                    self._event(member, EventType.CASE_RESOLVED, res_at.date(),
                                rx_id=rx.rx_id, at=res_at)
                )
        elif rng.random() < 0.45:
            # unresolved issues generate repeat contact -- the transcript's
            # "explaining the same problem over and over" pattern
            again = occurred + timedelta(days=rng.randint(2, 12))
            if again.date() <= self.end:
                out.interactions.append(
                    Interaction(
                        interaction_id=self._nid("INT"),
                        member_id=member.member_id,
                        occurred_on=again,
                        channel=interaction.channel,
                        intent=intent,
                        resolved=rng.random() < 0.6,
                        escalated=rng.random() < 0.3,
                    )
                )
                out.events.append(
                    self._event(member, EventType.MEMBER_CONTACTED_SUPPORT, again.date(),
                                rx_id=rx.rx_id, at=again,
                                payload={"intent": intent.value, "repeat": True})
                )
                state.recent_contacts.append(again)

        # digitally engaged members often look at the portal instead
        if rng.random() < traits.digital_engagement * 0.5:
            out.events.append(
                self._event(member, EventType.MEMBER_VIEWED_PORTAL,
                            when + timedelta(days=rng.randint(0, 5)), rx_id=rx.rx_id)
            )

    def _event(
        self,
        member: Member,
        etype: EventType,
        on: date,
        *,
        rx_id: str | None = None,
        claim_id: str | None = None,
        pa_id: str | None = None,
        document_id: str | None = None,
        payload: dict | None = None,
        at: datetime | None = None,
    ) -> JourneyEvent:
        return JourneyEvent(
            event_id=self._nid("EVT", 8),
            member_id=member.member_id,
            occurred_at=at or self._dt(on),
            event_type=etype,
            rx_id=rx_id,
            claim_id=claim_id,
            pa_id=pa_id,
            document_id=document_id,
            payload=payload or {},
        )


@dataclass
class _CycleState:
    pa_approved_until: date | None
    last_member_cost: float | None
    recent_contacts: list[datetime]
    recent_rejections: list[date]
    abandonment_count: int
    last_phase: DeductiblePhase | None = None


__all__ = ["JourneySimulator", "SimOutput", "Traits"]
