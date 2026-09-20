"""Unstructured content generation.

Two distinct corpora, deliberately kept separate:

* **Policy corpus** -- plan-level, member-agnostic. This is what RAG retrieves
  from. Facts are stated once, in one place, so groundedness is checkable.
* **Member artefacts** -- letters, messages, call transcripts, advocate and
  pharmacy notes tied to real entities in the structured tables. These feed the
  signal-extraction layer, not the policy index.

Keeping them separate is what lets the system distinguish "the plan says X"
(citable policy) from "this member said Y" (personal context). Blending them
into one index is the most common way a healthcare RAG demo becomes unsafe.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from careweave.domain.enums import (
    REJECT_CODE_MEANING,
    DocumentType,
    DrugTier,
    EventType,
    MemberIntent,
    PAStatus,
    RejectCode,
)
from careweave.domain.models import (
    Document,
    Drug,
    FormularyEntry,
    Member,
    Plan,
)

from .simulate import SimOutput

# ==========================================================================
# POLICY CORPUS
# ==========================================================================

_PA_POLICY = """\
# Prior Authorization Policy

## When prior authorization applies
Prior authorization is required before the plan will cover certain medications.
A medication requires prior authorization when it is listed on the covered drug
list with a prior authorization flag, when it falls in the specialty tier, or
when the prescribed quantity exceeds the plan's quantity limit for that drug.

## How a request is started
The prescriber submits the request. A member cannot submit a prior
authorization request themselves. If a pharmacy claim is rejected because prior
authorization is required, the pharmacy notifies the prescriber's office, and
the prescriber submits the request on the member's behalf.

## Standard review timeframe
Standard prior authorization requests are reviewed within 5 business days of
receiving a complete submission. Requests that are missing required clinical
documentation are placed in a pending-additional-information state and are not
counted as complete until the missing documentation arrives.

## Expedited review
An expedited review may be requested when waiting the standard timeframe could
seriously jeopardize the member's health. Expedited requests are reviewed within
72 hours of a complete submission.

## Pending additional information
When a reviewer needs more information, the plan notifies the prescriber with a
specific list of what is missing. Common requests are recent laboratory results,
documentation of medications previously tried, chart notes supporting medical
necessity, and corrected prescriber identification details. The prior
authorization clock does not restart, but the request stays open until the
documentation is received.

## Approval duration
An approved prior authorization is generally valid for 12 months from the
approval date unless the approval notice states a shorter period. A new request
is required when the approval expires, when the prescribed dose changes
materially, or when the member changes plans.

## If a request is denied
A denial notice states the reason for the denial and the member's appeal rights.
A member has 60 calendar days from the date of the denial notice to file a
first-level appeal. A prescriber may also request a peer-to-peer review before
an appeal is filed.

## Interim supply
When a prior authorization for an ongoing medication is pending and the member
has fewer than 7 days of medication remaining, the pharmacy may request a
one-time emergency supply of up to a 7 day fill so that therapy is not
interrupted. This applies to continuing therapy only, not to a first fill.
"""

_FORMULARY_POLICY = """\
# Covered Drug List and Tier Policy

## Tier structure
The covered drug list places each covered medication into one of four tiers.
Tier 1 is generic medications. Tier 2 is preferred brand medications. Tier 3 is
non-preferred brand medications. Tier 4 is specialty medications.

## How cost share is calculated
Tier 1, tier 2 and tier 3 medications use a flat copay per fill once the member
has met any applicable deductible. Tier 4 specialty medications use coinsurance,
which is a percentage of the medication's cost rather than a flat amount. This
is the reason a specialty medication's cost can change from fill to fill while a
generic medication's cost usually does not.

## Deductible
A member on a plan with a deductible pays the full discounted price of covered
medications until the deductible is met for the calendar year. Once the
deductible is met, the member moves to initial coverage and pays the tier copay
or coinsurance instead. The deductible resets on January 1 each year, which is
why a member may see a higher cost on their first fills of a new year even
though nothing about the medication or the plan has changed.

## Step therapy
Step therapy means the plan asks that a lower-cost medication be tried first
before covering a specific alternative. A prescriber may request an exception
where the step therapy medication is not clinically appropriate for that member.

## Quantity limits
Some medications carry a quantity limit expressed as a maximum amount per 30 day
period. A prescription written above the limit will reject at the pharmacy. The
prescriber may request a quantity limit exception.

## Non-covered medications
A medication that is not on the covered drug list will reject at the pharmacy.
Options are a formulary exception request submitted by the prescriber, or a
switch to a covered alternative selected by the prescriber.

## Changes to the covered drug list
The covered drug list can change during the plan year. When a change affects a
medication a member is currently taking, the plan sends written notice before
the change takes effect.
"""

_PHARMACY_NETWORK_POLICY = """\
# Pharmacy Network and Home Delivery

## In-network retail
Covered medications filled at an in-network retail pharmacy are charged at the
member's plan cost share. A standard retail fill is up to a 30 day supply. Some
in-network retail pharmacies also support a 90 day fill.

## Out-of-network retail
A pharmacy that is not in the plan's network is charged at a higher member cost
share, and some claims at out-of-network pharmacies will reject entirely. A
member who sees an unexpectedly high cost at the counter should check whether
the pharmacy is in network before paying.

## Home delivery
Home delivery is the plan's mail service pharmacy. It supports up to a 90 day
supply of maintenance medications and applies a discount to the member cost
share relative to a retail fill of the same medication. Home delivery typically
takes 7 to 10 days for a first order once the prescription and any required
prior authorization are in place, and shorter for refills.

## Specialty pharmacy
Specialty medications are dispensed through the plan's specialty pharmacy. The
specialty pharmacy contacts the member to schedule each shipment and confirms
the cost share before shipping.

## Transferring a prescription
A member may transfer a prescription between pharmacies. Any active prior
authorization stays with the member and the medication, and does not need to be
resubmitted because of a pharmacy change.
"""

_CLAIMS_POLICY = """\
# Pharmacy Claim Rejections

## What a rejection is
A pharmacy claim rejection means the pharmacy's request for coverage was not
approved at the point of sale. A rejection is not a bill and does not mean the
member owes money. It means the fill did not go through as submitted.

## Common rejection reasons
A claim can reject because prior authorization is required, because the refill
was requested earlier than the plan's refill schedule permits, because the
medication is not on the covered drug list, because the quantity exceeds the
plan limit, because a step therapy requirement has not been met, because the
pharmacy is out of network, because plan coverage shows as inactive on the fill
date, or because required prescriber information was missing.

## Refill too soon
The plan permits a refill once approximately 75 percent of the previous supply
has been used. For a 30 day supply this is roughly 23 days after the last fill.
A member travelling may request a vacation override for an early fill.

## Who resolves what
Prior authorization, step therapy and quantity limit rejections are resolved by
the prescriber. Network and coverage-status rejections are resolved by the plan.
Refill-too-soon rejections resolve on their own with time, or through a vacation
override.

## Reprocessing
When a claim rejected in error and the member paid out of pocket, the member may
submit a direct member reimbursement request. Requests must be submitted within
12 months of the fill date.
"""

_SOP_PROACTIVE = """\
# Standard Operating Procedure: Proactive Member Outreach

## Purpose
Proactive outreach exists to prevent a member from encountering a problem they
would otherwise discover at the pharmacy counter or through a support call. It
does not exist to increase message volume.

## Eligibility
An outreach is eligible only when all of the following hold: a specific
operational condition has been detected on the member's record, the outreach
names a concrete next step, the supporting evidence is current, and the member
has not received an outreach on the same topic within the suppression window.

## Suppression rules
No member receives more than three proactive outreaches in any rolling 30 day
period. No two outreaches on the same topic are sent within 14 days. Outreach is
suppressed entirely while a case is open with an advocate, because duplicate
contact from two directions increases confusion rather than reducing it.

## Content standards
An outreach states what is happening, why it matters to the member now, and what
the member can do next. It does not speculate about clinical matters, does not
recommend starting or stopping a medication, and does not reference member
information beyond what is necessary to make the message actionable.

## Escalation
Outreach is never used to deliver a denial, an appeal outcome, or any clinically
sensitive information. Those are handled by written notice and, where
appropriate, a live conversation.
"""

_SOP_ADVOCATE = """\
# Standard Operating Procedure: Advocate Case Handling

## Preparation
An advocate opening a case reviews the member's recent journey before speaking:
open prior authorizations, claims in the last 90 days, cost changes, and prior
contacts on the same topic. A member should not have to repeat information that
is already on the record.

## Repeat contact
When a member is contacting support about a topic they have already contacted
about, the case is treated as a continuation, not a new case. The advocate
states what has happened since the previous contact rather than restarting the
intake.

## Clinical questions
Questions about whether a medication is appropriate, about side effects, about
interactions, or about changing a dose are routed to a pharmacist. An advocate
does not answer clinical questions, and neither does any automated assistant.

## Escalation criteria
A case is escalated when a member has contacted three or more times on the same
unresolved issue, when a member reports being without a medication for an
ongoing condition, when a denial is being appealed, or when the member requests
escalation.

## Documentation
Every case records what the member asked, what was found, what action was taken,
and what the member was told to expect next.
"""

_FAQ = """\
# Member Frequently Asked Questions

## Why did my medication cost more this month than last month?
The three most common reasons are that the calendar year reset and the
deductible is being met again, that the medication moved to a different tier on
the covered drug list, or that the fill was made at a different pharmacy than
the previous fill. For specialty medications the cost is a percentage rather
than a flat amount, so it moves with the medication's price.

## What does it mean when my pharmacy says my prescription needs approval?
It means the medication requires prior authorization. The prescriber submits the
request. A standard review takes up to 5 business days after a complete
submission.

## How do I know if my prior authorization has been decided?
The status is visible on the member portal and can be confirmed by support. When
a decision is made, written notice is sent.

## Can I get a short supply while I wait for a decision?
If this is an ongoing medication and fewer than 7 days of supply remain, the
pharmacy may request a one-time emergency supply of up to 7 days.

## My claim was rejected. Do I owe money?
No. A rejected claim means the fill did not process. It is not a bill.

## How long does home delivery take?
About 7 to 10 days for a first order once everything needed is in place, and
less for refills.

## Can I use a different pharmacy?
Yes. Any active prior authorization follows the member and the medication, so it
does not need to be submitted again because of a pharmacy change.

## Who can answer a question about side effects?
A pharmacist. Support advocates and automated assistants do not answer clinical
questions.
"""


def build_policy_corpus(plans: list[Plan]) -> list[Document]:
    """The retrievable knowledge base. Member-agnostic and citable."""
    now = datetime(2026, 1, 1, 9, 0)
    docs: list[Document] = [
        Document(document_id="POL-PA-001", document_type=DocumentType.PA_POLICY,
                 title="Prior Authorization Policy", body=_PA_POLICY, created_at=now,
                 source_label="CareWeave Plan Policy Manual (synthetic)"),
        Document(document_id="POL-FRM-001", document_type=DocumentType.FORMULARY_POLICY,
                 title="Covered Drug List and Tier Policy", body=_FORMULARY_POLICY,
                 created_at=now, source_label="CareWeave Plan Policy Manual (synthetic)"),
        Document(document_id="POL-NET-001", document_type=DocumentType.FORMULARY_POLICY,
                 title="Pharmacy Network and Home Delivery", body=_PHARMACY_NETWORK_POLICY,
                 created_at=now, source_label="CareWeave Plan Policy Manual (synthetic)"),
        Document(document_id="POL-CLM-001", document_type=DocumentType.PA_POLICY,
                 title="Pharmacy Claim Rejections", body=_CLAIMS_POLICY, created_at=now,
                 source_label="CareWeave Plan Policy Manual (synthetic)"),
        Document(document_id="SOP-OUT-001", document_type=DocumentType.WORKFLOW_SOP,
                 title="SOP: Proactive Member Outreach", body=_SOP_PROACTIVE, created_at=now,
                 source_label="CareWeave Operations SOP (synthetic)"),
        Document(document_id="SOP-ADV-001", document_type=DocumentType.WORKFLOW_SOP,
                 title="SOP: Advocate Case Handling", body=_SOP_ADVOCATE, created_at=now,
                 source_label="CareWeave Operations SOP (synthetic)"),
        Document(document_id="FAQ-001", document_type=DocumentType.FAQ,
                 title="Member Frequently Asked Questions", body=_FAQ, created_at=now,
                 source_label="CareWeave Member Support (synthetic)"),
    ]

    for plan in plans:
        body = _plan_benefit_summary(plan)
        docs.append(
            Document(
                document_id=f"BEN-{plan.plan_id}",
                document_type=DocumentType.BENEFIT_SUMMARY,
                title=f"Benefit Summary: {plan.name}",
                body=body,
                created_at=now,
                source_label=f"CareWeave Benefit Summary {plan.plan_id} (synthetic)",
            )
        )
    return docs


def _plan_benefit_summary(plan: Plan) -> str:
    c = plan.copay_by_tier
    ded = (
        f"This plan has an annual pharmacy deductible of ${plan.deductible:,.0f} per member. "
        "Until the deductible is met, the member pays the full discounted price of covered "
        "medications."
        if plan.deductible > 0
        else "This plan has no annual pharmacy deductible. Tier cost share applies from the "
        "first fill of the year."
    )
    return f"""\
# Benefit Summary: {plan.name}

Plan identifier: {plan.plan_id}
Covered drug list: {plan.formulary_id}

## Deductible
{ded}

## Cost share after the deductible is met
Tier 1 generic: ${c[DrugTier.GENERIC.value]:.2f} per 30 day supply.
Tier 2 preferred brand: ${c[DrugTier.PREFERRED_BRAND.value]:.2f} per 30 day supply.
Tier 3 non-preferred brand: ${c[DrugTier.NON_PREFERRED_BRAND.value]:.2f} per 30 day supply.
Tier 4 specialty: {plan.specialty_coinsurance:.0%} coinsurance of the medication cost.

## Home delivery
Home delivery reduces the member cost share by {plan.mail_order_discount:.0%} compared with the
same medication filled at retail.

## Out-of-network retail
Filling at an out-of-network retail pharmacy increases the member cost share by
{plan.out_of_network_penalty:.0%}, and some claims will reject entirely.

## Prior authorization
Medications flagged on the covered drug list and all tier 4 specialty medications
require prior authorization before the plan will cover them.
"""


# ==========================================================================
# MEMBER ARTEFACTS
# ==========================================================================

_OPENERS = [
    "Hi, I'm calling because", "Hello, I need help with", "Hi there, I'm trying to find out",
    "Good morning, I have a question about", "Hi, this is about",
]

_COST_PHRASES = [
    "it was {prev} last time and now they're telling me {curr}",
    "I paid {prev} in {month} and today the pharmacy said {curr}",
    "the price jumped from {prev} to {curr} and nobody could tell me why",
]

_FRUSTRATION = [
    "This is the second time I've called about this.",
    "I already explained all of this last week.",
    "I've been going back and forth between the pharmacy and you all.",
    "Nobody has been able to give me a straight answer.",
]

_ACCESS = [
    "I'm going to run out on Friday.",
    "I have about four days left.",
    "I've already missed two doses because of this.",
    "I need this before I travel next week.",
]


class DocumentGenerator:
    def __init__(self, rng: random.Random, drugs: dict[str, Drug],
                 formulary: dict[str, FormularyEntry]) -> None:
        self.rng = rng
        self.drugs = drugs
        self.formulary = formulary
        self._n = 0

    def _id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}-{self._n:07d}"

    def generate(self, sim: SimOutput, members: dict[str, Member]) -> list[Document]:
        """Attach free text to a sampled subset of interactions and events.

        Not every event gets a document. Sparse, uneven coverage is realistic
        and forces the extraction layer to cope with missing text.
        """
        docs: list[Document] = []
        rx_drug = {rx.rx_id: self.drugs[rx.drug_id] for rx in sim.prescriptions}
        cost_by_rx: dict[str, list[tuple]] = {}
        for e in sim.events:
            if e.event_type == EventType.COST_CHANGED and e.rx_id:
                cost_by_rx.setdefault(e.rx_id, []).append(e)

        # -- call transcripts / member messages on interactions --------
        events_by_member: dict[str, list] = {}
        for e in sim.events:
            events_by_member.setdefault(e.member_id, []).append(e)

        for interaction in sim.interactions:
            if self.rng.random() > 0.55:
                continue
            member = members[interaction.member_id]
            nearby = [
                e for e in events_by_member.get(member.member_id, [])
                if 0 <= (interaction.occurred_on - e.occurred_at).days <= 21
            ]
            drug = None
            for e in reversed(nearby):
                if e.rx_id and e.rx_id in rx_drug:
                    drug = rx_drug[e.rx_id]
                    break
            doc = self._interaction_document(interaction, member, drug, nearby)
            if doc:
                docs.append(doc)
                interaction.document_id = doc.document_id

        # -- letters on PA decisions and rejections ---------------------
        for pa in sim.authorizations:
            if pa.status not in (PAStatus.APPROVED, PAStatus.DENIED) or pa.decided_on is None:
                continue
            if self.rng.random() > 0.6:
                continue
            member = members[pa.member_id]
            drug = rx_drug.get(pa.rx_id)
            docs.append(self._pa_letter(pa, member, drug))

        # -- advocate notes on escalations ------------------------------
        for e in sim.events:
            if e.event_type == EventType.CASE_ESCALATED and self.rng.random() < 0.7:
                member = members[e.member_id]
                drug = rx_drug.get(e.rx_id) if e.rx_id else None
                docs.append(self._advocate_note(e, member, drug))
            elif e.event_type == EventType.RX_ABANDONED_AT_COUNTER and self.rng.random() < 0.35:
                member = members[e.member_id]
                drug = rx_drug.get(e.rx_id) if e.rx_id else None
                docs.append(self._pharmacy_note(e, member, drug))

        return docs

    # -- individual artefact builders ---------------------------------

    def _interaction_document(self, interaction, member: Member, drug, nearby) -> Document | None:
        rng = self.rng
        intent = interaction.intent
        drug_name = drug.name if drug else "my medication"

        recent_types = {e.event_type for e in nearby}
        body_parts: list[str] = []

        opener = rng.choice(_OPENERS)
        if intent == MemberIntent.WHY_DID_COST_CHANGE:
            cost_evt = next(
                (e for e in reversed(nearby) if e.event_type == EventType.COST_CHANGED), None
            )
            prev = f"${cost_evt.payload['previous']:.2f}" if cost_evt else "$20"
            curr = f"${cost_evt.payload['current']:.2f}" if cost_evt else "$65"
            month = (interaction.occurred_on - timedelta(days=32)).strftime("%B")
            body_parts.append(
                f"{opener} the cost of {drug_name}. "
                + rng.choice(_COST_PHRASES).format(prev=prev, curr=curr, month=month)
                + "."
            )
        elif intent == MemberIntent.AUTHORIZATION_STATUS:
            body_parts.append(
                f"{opener} the approval for {drug_name}. The pharmacy said it needs to be "
                "authorized and that my doctor had to send something in. I don't know if that "
                "happened or where it stands."
            )
        elif intent == MemberIntent.CLAIM_REJECTED_WHY:
            body_parts.append(
                f"{opener} why {drug_name} was rejected at the pharmacy. They just handed me a "
                "slip with a code on it and told me to call my insurance."
            )
        elif intent == MemberIntent.WHERE_IS_MY_PRESCRIPTION:
            body_parts.append(
                f"{opener} where {drug_name} is. It was supposed to be ready and the pharmacy "
                "says there's a hold on it."
            )
        elif intent == MemberIntent.EXPLAIN_LETTER:
            body_parts.append(
                f"{opener} a letter I received about {drug_name}. I've read it twice and I'm "
                "still not sure what it's asking me to do."
            )
        else:
            body_parts.append(
                f"{opener} {drug_name} and what I'm supposed to do next. The pharmacy told me "
                "to check with you and I don't know what the next step is."
            )

        if EventType.PA_INFO_REQUESTED in recent_types:
            body_parts.append(
                "Somebody mentioned they were waiting on paperwork from my doctor's office."
            )
        if not interaction.resolved or rng.random() < 0.3:
            body_parts.append(rng.choice(_FRUSTRATION))
        if rng.random() < 0.4:
            body_parts.append(rng.choice(_ACCESS))
        if member.is_caregiver and rng.random() < 0.4:
            body_parts.append("I'm handling this for my mother, so I'm going off what she tells me.")
        if rng.random() < 0.08:
            # a small share of contacts are genuinely clinical -- these must be
            # detected and routed, never answered by the assistant
            body_parts.append(
                "Also, is it normal to feel dizzy after taking this? Should I stop it?"
            )

        text = " ".join(body_parts)
        is_call = interaction.channel.value == "phone"
        if is_call:
            body = (
                f"[Call transcript excerpt -- {interaction.occurred_on:%Y-%m-%d %H:%M}]\n"
                f"MEMBER: {text}\n"
                f"ADVOCATE: Let me pull up your record and take a look at what's happening.\n"
            )
            dtype = DocumentType.CALL_TRANSCRIPT
            title = f"Call transcript {interaction.interaction_id}"
        else:
            body = text
            dtype = DocumentType.MEMBER_MESSAGE
            title = f"Member message {interaction.interaction_id}"

        return Document(
            document_id=self._id("DOC"),
            document_type=dtype,
            title=title,
            body=body,
            created_at=interaction.occurred_on,
            member_id=member.member_id,
            source_label="Member interaction record (synthetic)",
        )

    def _pa_letter(self, pa, member: Member, drug) -> Document:
        drug_name = drug.name if drug else "the requested medication"
        assert pa.decided_on is not None
        if pa.status == PAStatus.APPROVED:
            body = (
                f"Dear {member.full_name},\n\n"
                f"We reviewed the coverage request your prescriber submitted for {drug_name}. "
                f"The request has been approved effective {pa.decided_on:%B %d, %Y}. "
                "This approval is generally valid for 12 months from the approval date.\n\n"
                "You can now fill this prescription at a participating pharmacy. If your "
                "pharmacy already has the prescription on file, no further action is needed "
                "from you.\n\n"
                "If you have questions about this notice, contact member support."
            )
            title = f"Coverage approval notice -- {drug_name}"
        else:
            body = (
                f"Dear {member.full_name},\n\n"
                f"We reviewed the coverage request your prescriber submitted for {drug_name}. "
                f"The request was not approved. Reason: {pa.denial_reason}.\n\n"
                "You have the right to appeal this decision. An appeal must be filed within 60 "
                "calendar days of the date of this notice. Your prescriber may also request a "
                "peer-to-peer review.\n\n"
                "Your prescriber can discuss whether a covered alternative is appropriate for "
                "you. This notice is a coverage determination and is not medical advice."
            )
            title = f"Coverage denial notice -- {drug_name}"

        return Document(
            document_id=self._id("DOC"),
            document_type=DocumentType.MEMBER_LETTER,
            title=title,
            body=body,
            created_at=datetime.combine(pa.decided_on, datetime.min.time()).replace(hour=6),
            member_id=member.member_id,
            rx_id=pa.rx_id,
            source_label="Member correspondence (synthetic)",
        )

    def _advocate_note(self, event, member: Member, drug) -> Document:
        drug_name = drug.name if drug else "medication"
        rng = self.rng
        body = (
            f"Case escalated. Member contacted regarding {drug_name}. "
            + rng.choice(
                [
                    "Confirmed authorization is still open on the plan side; prescriber office "
                    "has not returned requested documentation.",
                    "Member reports two prior contacts with no resolution. Reviewed prior notes; "
                    "no outbound follow-up was logged after the first contact.",
                    "Cost at counter differs from member expectation. Verified deductible has "
                    "reset for the calendar year.",
                    "Pharmacy submitted at a location that is not in network. Member was not "
                    "aware of the network status.",
                ]
            )
            + " Advised member of next step and set follow-up. Escalating for ownership."
        )
        return Document(
            document_id=self._id("DOC"),
            document_type=DocumentType.ADVOCATE_NOTE,
            title=f"Advocate note -- escalation {event.event_id}",
            body=body,
            created_at=event.occurred_at,
            member_id=member.member_id,
            rx_id=event.rx_id,
            source_label="Advocate case notes (synthetic)",
        )

    def _pharmacy_note(self, event, member: Member, drug) -> Document:
        drug_name = drug.name if drug else "medication"
        cost = event.payload.get("member_cost")
        cost_s = f"${cost:.2f}" if isinstance(cost, (int, float)) else "the quoted amount"
        body = (
            f"Patient presented for {drug_name}. Copay quoted at {cost_s}. "
            "Patient declined to pick up and asked to be contacted if the price changes. "
            "Prescription returned to stock."
        )
        return Document(
            document_id=self._id("DOC"),
            document_type=DocumentType.PHARMACY_NOTE,
            title=f"Pharmacy note -- {event.event_id}",
            body=body,
            created_at=event.occurred_at,
            member_id=member.member_id,
            rx_id=event.rx_id,
            source_label="Pharmacy dispensing note (synthetic)",
        )


def reject_explanation(code: RejectCode) -> str:
    return REJECT_CODE_MEANING[code]
