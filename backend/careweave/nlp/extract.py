"""Unstructured-data intelligence.

Turns member messages, call transcripts, pharmacy notes and advocate notes into
a typed ``SignalSet`` the rest of the system can reason over.

Architecture note: the deterministic extractor is not a placeholder for the LLM
one -- it is the floor. It always runs. When an LLM is configured, its output is
*merged* with the deterministic result, and the deterministic result wins on the
two fields where a false negative is dangerous: ``clinical_question_detected``
and distress-level sentiment. A missed clinical question routes a safety
question to an automated answer, so that detection is never left solely to a
probabilistic component.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from careweave.config import SETTINGS
from careweave.domain.enums import BarrierType, MemberIntent, Sentiment, Urgency
from careweave.domain.models import Document, SignalSet
from careweave.llm.provider import LLMProvider, get_provider

EXTRACTOR_VERSION = "signal-extractor-v1"

# --- lexicons ---------------------------------------------------------------

_INTENT_PATTERNS: list[tuple[MemberIntent, list[str]]] = [
    (MemberIntent.WHY_DID_COST_CHANGE,
     [r"cost (of|more|change|went up)", r"price (jump|went|change)",
      r"paid \$", r"more expensive", r"copay.*(higher|different)",
      # two dollar amounts in one message is nearly always a before/after
      r"\$[\d,.]+.{0,80}\$[\d,.]+", r"charg(ed|ing)",
      r"cost .{0,15}more", r"paid .{0,15}more", r"went up", r"gone up"]),
    (MemberIntent.AUTHORIZATION_STATUS,
     [r"authoriz", r"approval", r"approved", r"pre-?auth", r"needs? to be authorized",
      r"waiting on (my )?doctor", r"paperwork"]),
    (MemberIntent.CLAIM_REJECTED_WHY,
     [r"reject", r"denied at the (pharmacy|counter)", r"slip with a code",
      r"didn.?t go through", r"wouldn.?t process"]),
    (MemberIntent.WHERE_IS_MY_PRESCRIPTION,
     [r"where (is|are) (my|the)", r"hold on it", r"supposed to be ready",
      r"hasn.?t (arrived|shipped)", r"status of my (prescription|refill)"]),
    (MemberIntent.LOWER_COST_OPTION,
     [r"cheaper", r"lower cost", r"generic", r"less expensive", r"afford"]),
    (MemberIntent.HOME_DELIVERY,
     [r"home delivery", r"mail order", r"delivered", r"by mail"]),
    (MemberIntent.EXPLAIN_LETTER,
     [r"got a letter", r"this letter", r"notice (i|we) received", r"letter.*mean"]),
    (MemberIntent.IS_THIS_COVERED,
     [r"is (this|it) covered", r"covered by my plan", r"on the (drug )?list",
      r"formulary"]),
    (MemberIntent.WHAT_HAPPENS_NEXT,
     [r"what.*(do|happens) next", r"what am i supposed to", r"next step",
      r"supposed to do"]),
    # Clinical is matched LAST for intent purposes: a clinical aside inside a
    # billing question should not relabel the whole contact. Clinical *detection*
    # is a separate boolean below and is unaffected by this ordering.
    (MemberIntent.CLINICAL_QUESTION,
     [r"side effect", r"is (it|this|that) (normal|safe|ok|okay|bad)",
      r"should i (stop|take|keep|switch|be)", r"safe to (take|use)",
      r"interact(s|ion)", r"dizzy", r"nause", r"allergic",
      # "dose" alone fires on "I missed two doses", which is an access signal,
      # not a clinical question. Require an intent to change the dose.
      r"(change|adjust|skip|double|split|increase|lower)\w*\s+(my\s+)?dos",
      r"my dos(e|age)"]),
]

_COST_TERMS = [r"\$\d", r"cost", r"price", r"copay", r"expensive", r"afford", r"deductible"]
_ACCESS_TERMS = [
    r"run out", r"running out", r"out of", r"days left", r"missed (a|two|three|\d+) dose",
    r"without (my|the) medication", r"can.?t get", r"travel",
]
_CONFUSION_TERMS = [
    r"don.?t (know|understand)", r"not sure", r"confus", r"nobody (could|has)",
    r"unclear", r"what does .* mean", r"no one (told|explained)",
]
_ADMIN_TERMS = [r"paperwork", r"authoriz", r"approval", r"form", r"documentation", r"reject"]
_REPEAT_TERMS = [
    r"second time", r"third time", r"already (called|explained|told)", r"again",
    r"back and forth", r"last (week|time)", r"still (waiting|haven)",
]
_FRUSTRATION_TERMS = [
    r"frustrat", r"ridiculous", r"nobody", r"no one", r"back and forth",
    r"straight answer", r"keep (getting|being)",
]
_DISTRESS_TERMS = [
    r"desperate", r"scared", r"panic", r"can.?t afford", r"missed (two|three|\d+) dose",
    r"without (my|the) medication for",
]
_URGENT_TERMS = [
    r"today", r"tomorrow", r"friday", r"run out", r"days left", r"urgent",
    r"emergency", r"missed .* dose",
]


#: Resolved once at import so clinical detection never depends on list order.
_CLINICAL_PATTERNS: list[str] = [
    p for intent, pats in _INTENT_PATTERNS if intent is MemberIntent.CLINICAL_QUESTION
    for p in pats
]


def _any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def _count(patterns: list[str], text: str) -> int:
    return sum(1 for p in patterns if re.search(p, text))


@dataclass
class ExtractionInput:
    text: str
    document_ids: list[str]


def deterministic_extract(payload: ExtractionInput, known_drugs: set[str]) -> SignalSet:
    """Rule-based extraction. Always runs; sets the safety floor."""
    text = payload.text.lower()

    intent = MemberIntent.OTHER
    for candidate, patterns in _INTENT_PATTERNS:
        if _any(patterns, text):
            intent = candidate
            break

    clinical = _any(_CLINICAL_PATTERNS, text)
    mentions_cost = _any(_COST_TERMS, text)
    mentions_access = _any(_ACCESS_TERMS, text)
    confusion = _any(_CONFUSION_TERMS, text)
    admin = _any(_ADMIN_TERMS, text)
    repeat = _any(_REPEAT_TERMS, text)

    if clinical:
        barrier = BarrierType.CLINICAL_CONCERN
    elif mentions_cost and intent in (
        MemberIntent.WHY_DID_COST_CHANGE, MemberIntent.LOWER_COST_OPTION
    ):
        barrier = BarrierType.COST
    elif mentions_access:
        barrier = BarrierType.ACCESS
    elif admin:
        barrier = BarrierType.ADMINISTRATIVE
    elif confusion:
        barrier = BarrierType.CONFUSION
    elif mentions_cost:
        barrier = BarrierType.COST
    else:
        barrier = BarrierType.NONE_DETECTED

    if _any(_DISTRESS_TERMS, text):
        sentiment = Sentiment.DISTRESSED
    elif _count(_FRUSTRATION_TERMS, text) >= 1 or repeat:
        sentiment = Sentiment.FRUSTRATED
    elif confusion:
        sentiment = Sentiment.CONFUSED
    else:
        sentiment = Sentiment.NEUTRAL

    urgent_hits = _count(_URGENT_TERMS, text)
    if urgent_hits >= 2 or sentiment == Sentiment.DISTRESSED:
        urgency = Urgency.HIGH
    elif urgent_hits == 1 or repeat:
        urgency = Urgency.MEDIUM
    else:
        urgency = Urgency.LOW

    mentioned = sorted({d for d in known_drugs if d.lower() in text})

    # confidence reflects how much the text actually gave us
    signal_hits = sum(
        [intent is not MemberIntent.OTHER, mentions_cost, mentions_access,
         confusion, admin, repeat, bool(mentioned)]
    )
    confidence = min(0.95, 0.28 + 0.11 * signal_hits)

    return SignalSet(
        intent=intent,
        barrier=barrier,
        urgency=urgency,
        sentiment=sentiment,
        mentioned_drugs=mentioned,
        mentions_cost=mentions_cost,
        mentions_access=mentions_access,
        repeat_contact_language=repeat,
        clinical_question_detected=clinical,
        source_document_ids=payload.document_ids,
        extractor_version=EXTRACTOR_VERSION,
        confidence=round(confidence, 3),
    )


_SYSTEM = """You extract structured signals from a pharmacy benefit member's own words.

You are not answering the member. You are not giving advice. You classify.

Rules:
- Use only the enum values given in the schema.
- If the member asks anything about whether a medication is safe, appropriate,
  causing a symptom, or whether to change a dose, set clinical_question_detected
  to true. Err toward true.
- Set confidence to reflect how much the text actually supports your labels.
"""

_SCHEMA_HINT = """{
  "intent": one of [where_is_my_prescription, why_did_cost_change, is_this_covered,
                    authorization_status, claim_rejected_why, lower_cost_option,
                    home_delivery, explain_letter, what_happens_next,
                    clinical_question, other],
  "barrier": one of [cost, access, confusion, administrative, clinical_concern, none_detected],
  "urgency": one of [low, medium, high],
  "sentiment": one of [neutral, confused, frustrated, distressed],
  "mentions_cost": boolean,
  "mentions_access": boolean,
  "repeat_contact_language": boolean,
  "clinical_question_detected": boolean,
  "confidence": number between 0 and 1
}"""


class SignalExtractor:
    def __init__(self, provider: LLMProvider | None = None,
                 known_drugs: set[str] | None = None) -> None:
        self.provider = provider or get_provider()
        self.known_drugs = known_drugs or set()

    def extract(self, documents: list[Document]) -> SignalSet:
        if not documents:
            return SignalSet(extractor_version=EXTRACTOR_VERSION, confidence=0.0)

        text = "\n\n".join(d.body for d in documents[:4])
        payload = ExtractionInput(text=text, document_ids=[d.document_id for d in documents[:4]])
        base = deterministic_extract(payload, self.known_drugs)

        if getattr(self.provider, "name", "stub") == "stub":
            return base

        user = (
            f"Member text:\n---\n{text[:4000]}\n---\n\n"
            f"FALLBACK:\n{json.dumps(base.model_dump(mode='json'))}"
        )
        result = self.provider.complete_json(
            system=_SYSTEM, user=user, schema_hint=_SCHEMA_HINT, max_tokens=500
        )
        if result.error or not result.parsed:
            return base
        return self._merge(base, result.parsed)

    @staticmethod
    def _merge(base: SignalSet, parsed: dict) -> SignalSet:
        """LLM refines, deterministic rules hold the safety floor."""
        def enum_or(cls, key, default):
            try:
                return cls(parsed[key])
            except (KeyError, ValueError):
                return default

        merged = base.model_copy(
            update={
                "intent": enum_or(MemberIntent, "intent", base.intent),
                "barrier": enum_or(BarrierType, "barrier", base.barrier),
                "urgency": enum_or(Urgency, "urgency", base.urgency),
                "sentiment": enum_or(Sentiment, "sentiment", base.sentiment),
                "mentions_cost": bool(parsed.get("mentions_cost", base.mentions_cost)),
                "mentions_access": bool(parsed.get("mentions_access", base.mentions_access)),
                "repeat_contact_language": bool(
                    parsed.get("repeat_contact_language", base.repeat_contact_language)
                ),
                # OR, never AND: a rule-detected clinical question cannot be
                # overridden by the model saying otherwise
                "clinical_question_detected": bool(
                    parsed.get("clinical_question_detected", False)
                ) or base.clinical_question_detected,
                "confidence": float(parsed.get("confidence", base.confidence)),
                "extractor_version": f"{EXTRACTOR_VERSION}+llm",
            }
        )
        # distress is never downgraded by the model
        if base.sentiment == Sentiment.DISTRESSED:
            merged = merged.model_copy(update={"sentiment": Sentiment.DISTRESSED})
        return merged
