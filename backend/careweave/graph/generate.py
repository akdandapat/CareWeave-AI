"""Grounded generation and verification.

The generator's contract is narrow on purpose. It receives:

* an action that has **already been selected** and passed the gate,
* a fixed set of grounded facts drawn from structured records,
* a fixed set of retrieved policy chunks,

and produces prose. It cannot select a different action, cannot introduce a fact
that is not in its inputs, and cannot reference a member attribute outside the
governance allowlist.

Enforcement is two-layer:

1. **A deterministic draft is always composed first**, from templates keyed to
   the action type and filled only from grounded facts. In the offline profile
   this draft *is* the answer. With an LLM configured, the draft is passed in as
   the content to be rewritten, so the model's job is fluency, not fact supply.

2. **The verifier checks the output back against the inputs.** Numbers, dates
   and dollar amounts in the response must appear in the grounded facts or the
   retrieved evidence. Anything else is an ungrounded span, and one regeneration
   is attempted before the response is withheld.

A number that appears in an answer but nowhere in the evidence is the single
most damaging failure mode a system like this has, because it is fluent,
specific and wrong. That is why the check is on numerals specifically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from careweave.config import SETTINGS
from careweave.domain.enums import ActionType, Channel
from careweave.domain.models import (
    ActionDecision,
    ContextFrame,
    EvidenceChunk,
    GeneratedResponse,
    MemberContext,
    SignalSet,
)
from careweave.governance.gate import MEMBER_FACING_FIELD_DENYLIST
from careweave.llm.provider import LLMProvider, get_provider

GENERATOR_VERSION = "responder-v1"

_NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")

_SYSTEM = """You write short messages to health plan members about their pharmacy benefits.

Hard constraints:
- Rewrite ONLY the draft you are given. Every fact, number, date and dollar
  amount in your output must already appear in the draft or the supporting
  evidence. Introduce nothing.
- Do not give clinical advice. Do not suggest starting, stopping, changing or
  substituting any medication.
- Do not mention risk scores, models, predictions, or anything about how this
  message was selected.
- Do not speculate about what will happen. State what is on the record and what
  the member can do.

Style: plain language, second person, 3-5 sentences, no greeting boilerplate, no
sign-off. Warm but not chatty. End with a concrete next step."""


@dataclass
class GenerationInputs:
    action: ActionType
    frame: ContextFrame
    ctx: MemberContext
    signals: SignalSet
    evidence: list[EvidenceChunk]
    decision: ActionDecision
    audience: str = "member"  # "member" | "advocate"


# ---------------------------------------------------------------------------
# Deterministic draft composition
# ---------------------------------------------------------------------------


def _strip_provenance(fact: str) -> str:
    return re.sub(r"\s*\[[a-z_]+:[^\]]+\]\s*$", "", fact).strip()


#: Which record types each action's explanation should draw on. A claim
#: rejection explanation that also recites unrelated cost history reads as a data
#: dump rather than an answer -- and the transcripts are explicit that more
#: information is not the same as better guidance.
_RELEVANT_SOURCES: dict[ActionType, tuple[str, ...]] = {
    ActionType.EXPLAIN_AUTH_STATUS: ("authorization", "claim", "prescription"),
    ActionType.REQUEST_MISSING_INFO: ("authorization", "prescription"),
    ActionType.EXPLAIN_CLAIM_STATUS: ("claim", "authorization", "prescription"),
    ActionType.EXPLAIN_COST_CHANGE: ("cost", "claim"),
    ActionType.EXPLAIN_COVERAGE: ("claim", "cost"),
    ActionType.SURFACE_ALTERNATIVE_OPTION: ("cost", "claim"),
    ActionType.PROVIDE_NEXT_STEP: ("authorization", "claim", "prescription"),
    ActionType.ESCALATE_CASE: ("interaction", "claim", "authorization"),
    ActionType.ROUTE_TO_ADVOCATE: ("interaction", "claim", "authorization"),
    ActionType.SCHEDULE_FOLLOW_UP: ("authorization", "prescription"),
}

_SOURCE_KIND = re.compile(r"\[([a-z_]+):[^\]]+\]\s*$")


def _fact_kind(fact: str) -> str:
    m = _SOURCE_KIND.search(fact)
    return m.group(1) if m else "unknown"


def _facts_text(frame: ContextFrame, action: ActionType, limit: int = 3) -> list[str]:
    """Facts the member needs for *this* answer, in priority order."""
    wanted = _RELEVANT_SOURCES.get(action)
    facts = frame.grounded_facts
    if wanted:
        ranked = sorted(
            (f for f in facts if _fact_kind(f) in wanted),
            key=lambda f: wanted.index(_fact_kind(f)),
        )
        facts = ranked or facts
    return [_strip_provenance(f) for f in facts[:limit]]


_CLOSERS: dict[ActionType, str] = {
    ActionType.EXPLAIN_AUTH_STATUS:
        "You do not need to do anything to move this along. If you are close to "
        "running out, your pharmacy can ask about a short emergency supply.",
    ActionType.REQUEST_MISSING_INFO:
        "The fastest way to move this forward is to ask your prescriber's office "
        "to send the requested documentation.",
    ActionType.EXPLAIN_CLAIM_STATUS:
        "A rejected claim is not a bill and you do not owe anything for it.",
    ActionType.EXPLAIN_COST_CHANGE:
        "If the amount is difficult, it is worth asking about home delivery or a "
        "90 day fill, which can lower what you pay per month.",
    ActionType.EXPLAIN_COVERAGE:
        "If you would like this checked against your specific prescription, an "
        "advocate can confirm it with you.",
    ActionType.SURFACE_ALTERNATIVE_OPTION:
        "An advocate can set any of this up with you if you would like.",
    ActionType.PROVIDE_NEXT_STEP:
        "If anything changes before then, you can reach support and this history "
        "will already be on your record.",
    ActionType.SCHEDULE_FOLLOW_UP:
        "We will check back on this so you do not have to.",
    ActionType.ROUTE_TO_ADVOCATE:
        "An advocate will pick this up with the history already in front of them, "
        "so you will not need to start over.",
    ActionType.ROUTE_TO_PHARMACIST:
        "A pharmacist is the right person for this and will follow up with you "
        "directly.",
    ActionType.ESCALATE_CASE:
        "This has been escalated so that one person owns it through to a resolution.",
}

_OPENERS: dict[ActionType, str] = {
    ActionType.EXPLAIN_AUTH_STATUS: "Here is where your coverage review stands.",
    ActionType.REQUEST_MISSING_INFO: "Your coverage review is open and waiting on one thing.",
    ActionType.EXPLAIN_CLAIM_STATUS: "Here is why your prescription did not go through.",
    ActionType.EXPLAIN_COST_CHANGE: "Here is why the amount at the pharmacy changed.",
    ActionType.EXPLAIN_COVERAGE: "Here is how your plan covers this.",
    ActionType.SURFACE_ALTERNATIVE_OPTION: "There may be a way to lower what you pay.",
    ActionType.PROVIDE_NEXT_STEP: "Here is where things stand and what happens next.",
    ActionType.SCHEDULE_FOLLOW_UP: "We are keeping an eye on this for you.",
    ActionType.ROUTE_TO_ADVOCATE: "We are connecting you with an advocate.",
    ActionType.ROUTE_TO_PHARMACIST: "This is a question for a pharmacist.",
    ActionType.ESCALATE_CASE: "We have escalated this.",
}


def compose_draft(inputs: GenerationInputs) -> str:
    """Deterministic, fully grounded draft. The floor beneath every response."""
    action = inputs.action
    parts: list[str] = [_OPENERS.get(action, "Here is what we can see.")]

    facts = _facts_text(inputs.frame, action)
    if action == ActionType.ROUTE_TO_PHARMACIST:
        parts.append(
            "Questions about how a medication is affecting you, or whether to keep "
            "taking it, are answered by a pharmacist rather than by this assistant."
        )
    elif facts:
        parts.append(" ".join(f"{f}." for f in facts))
    else:
        parts.append("Nothing on your record needs your attention right now.")

    # one supporting policy sentence, quoted structurally rather than verbatim
    if inputs.evidence and action not in (ActionType.ROUTE_TO_PHARMACIST,):
        top = inputs.evidence[0]
        # policy chunks are hard-wrapped markdown; collapse to one line before use
        flat = " ".join(top.text.split())
        first_sentence = flat.split(". ")[0].strip()
        if first_sentence and len(first_sentence) < 260:
            parts.append(f"Under your plan, {first_sentence[0].lower()}{first_sentence[1:]}.")

    parts.append(_CLOSERS.get(action, "Support can help if you have questions."))
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _numbers(text: str) -> set[str]:
    out = set()
    for m in _NUMBER.findall(text):
        cleaned = m.replace(",", "").replace("$", "").rstrip("%").rstrip(".")
        if cleaned:
            out.add(cleaned)
    return out


#: Numerals so common in benefit language that requiring evidence for them
#: produces noise rather than safety (ordinals, small counts in phrases like
#: "one thing"). Kept explicit and short.
_ALLOWED_NUMERALS = {"1", "2", "3", "0"}


def verify_grounding(
    text: str, inputs: GenerationInputs, *, draft: str | None = None
) -> tuple[bool, list[str]]:
    """Check every numeral in the response against the supplied evidence.

    ``draft`` is the deterministic composition, which is grounded by
    construction: it is assembled only from structured records and from static
    template constants whose numerals are asserted against the policy corpus in
    tests/test_templates_grounded.py. Including it here means an LLM that
    faithfully rewrites the draft passes, while one that invents a figure the
    draft never contained does not.
    """
    evidence_text = " ".join(
        [*inputs.frame.grounded_facts, *(e.text for e in inputs.evidence),
         inputs.frame.what_is_happening, inputs.frame.what_could_happen_next,
         draft or ""]
    )
    allowed = _numbers(evidence_text) | _ALLOWED_NUMERALS
    ungrounded = sorted(n for n in _numbers(text) if n not in allowed)

    # denylisted attributes must never surface, regardless of numerals
    lowered = text.lower()
    leaked = [
        field for field in MEMBER_FACING_FIELD_DENYLIST
        if field.replace("_", " ") in lowered
    ]
    problems = [f"ungrounded number: {n}" for n in ungrounded]
    problems += [f"denylisted attribute referenced: {f}" for f in leaked]
    return (not problems), problems


# ---------------------------------------------------------------------------
# Responder
# ---------------------------------------------------------------------------


class Responder:
    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider or get_provider()

    @property
    def version(self) -> str:
        return f"{GENERATOR_VERSION}/{SETTINGS.llm.version_tag}"

    def generate(self, inputs: GenerationInputs) -> GeneratedResponse:
        draft = compose_draft(inputs)
        text = draft
        attempts = 0

        if getattr(self.provider, "name", "stub") != "stub":
            for attempts in range(1, 3):
                result = self.provider.complete(
                    system=_SYSTEM,
                    user=self._user_prompt(inputs, draft, strict=attempts > 1),
                    max_tokens=SETTINGS.llm.max_tokens,
                )
                if result.error or not result.text:
                    text = draft  # degrade to the grounded draft
                    break
                ok, _ = verify_grounding(result.text, inputs, draft=draft)
                text = result.text
                if ok:
                    break
            else:  # pragma: no cover - loop always breaks
                text = draft

        grounded, problems = verify_grounding(text, inputs, draft=draft)
        if not grounded:
            # never ship an unverified response; fall back to the draft, which is
            # grounded by construction
            text = draft
            grounded, problems = verify_grounding(text, inputs, draft=draft)

        return GeneratedResponse(
            text=text,
            citations=[e.chunk_id for e in inputs.evidence[:3]],
            generator_version=self.version,
            grounded=grounded,
            ungrounded_spans=problems,
        )

    @staticmethod
    def _user_prompt(inputs: GenerationInputs, draft: str, *, strict: bool) -> str:
        evidence_block = "\n".join(
            f"- [{e.chunk_id}] {e.title}: {e.text[:400]}" for e in inputs.evidence[:3]
        ) or "- (no policy evidence retrieved)"
        strict_note = (
            "\nYour previous attempt introduced a number that was not in the "
            "evidence. Use ONLY numbers that appear below.\n" if strict else ""
        )
        return (
            f"Selected action (fixed, do not change): {inputs.action.value}\n"
            f"Audience: {inputs.audience}\n\n"
            f"Supporting policy evidence:\n{evidence_block}\n\n"
            f"Grounded facts from the member's record:\n"
            + "\n".join(f"- {_strip_provenance(f)}" for f in inputs.frame.grounded_facts)
            + strict_note
            + f"\n\nDRAFT:\n{draft}"
        )


def advocate_summary(inputs: GenerationInputs, risk_note: str) -> str:
    """Internal-facing case summary.

    Deliberately a different surface from the member response: an advocate is
    authorised to see the risk score and the reason codes, a member is not. Same
    engine, different disclosure boundary, enforced by using a separate function
    rather than a prompt instruction.
    """
    lines = [
        f"WHAT IS HAPPENING: {inputs.frame.what_is_happening}",
        f"WHY IT MATTERS: {inputs.frame.why_it_matters}",
        f"LIKELY NEXT: {inputs.frame.what_could_happen_next}",
        f"RISK: {risk_note}",
        f"RECOMMENDED ACTION: {inputs.action.value} "
        f"(confidence {inputs.decision.confidence:.2f})",
    ]
    if inputs.frame.grounded_facts:
        lines.append("RECORD:")
        lines += [f"  - {f}" for f in inputs.frame.grounded_facts]
    if inputs.evidence:
        lines.append("POLICY BASIS:")
        lines += [f"  - [{e.chunk_id}] {e.title}" for e in inputs.evidence[:3]]
    rejected = {
        k: v for k, v in inputs.decision.rejected_alternatives.items()
        if k != "__notes__"
    }
    if rejected:
        lines.append("CONSIDERED AND REJECTED:")
        lines += [f"  - {k}: {v}" for k, v in list(rejected.items())[:4]]
    return "\n".join(lines)
