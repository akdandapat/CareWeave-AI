# Responsible AI

Implemented as code and as a queryable table, not as a section in a README.

## The governance gate

`governance/gate.py` is the last checkpoint before anything reaches a member.
Four properties define it.

**Fully deterministic.** No model call, no probability. Same inputs, same verdict,
forever. A safety control that is itself stochastic is not a control.

**It can only restrict.** ALLOW can become HOLD, SUPPRESS or ESCALATE. There is
no path in the other direction. A bug fails closed. `test_gate_can_only_restrict`
asserts this against every restrictive input combination.

**It runs on every path**, including the ones where the engine decided to do
nothing. Silence is a governed decision.

**Every verdict carries reason codes**, so "why did this member get this message
on this day" is answerable without re-running anything.

Rules are ordered by severity; the first restrictive match wins, so a clinical
escalation is never overridden by a frequency cap further down.

| Order | Rule | Verdict |
|---|---|---|
| 1 | Clinical question detected | ESCALATE, human required |
| 2 | Distress language detected | ESCALATE, human required |
| 3 | Action structurally requires a human | ESCALATE, human required |
| 4 | Retrieved evidence is contradictory | HOLD |
| 5 | Evidence insufficient to assert | HOLD |
| 6 | Action confidence below floor | HOLD |
| 7 | Explicit no-action | SUPPRESS |
| 8 | Proactive: action not eligible for unsolicited contact | SUPPRESS |
| 9 | Proactive: risk band low | SUPPRESS |
| 10 | Proactive: a case is already open with an advocate | SUPPRESS |
| 11 | Proactive: same topic sent within 14 days | SUPPRESS |
| 12 | Proactive: 3 messages already sent in 30 days | SUPPRESS |

Rule 10 exists because contact from two directions increases confusion rather
than reducing it — the failure mode "proactive service" most easily becomes.

## Clinical safety

Clinical questions never receive an automated answer at any confidence. Three
independent mechanisms, so no single component's failure is sufficient:

1. **Detection runs deterministically and always.** The rule-based extractor sets
   the floor and cannot be switched off.
2. **The language model cannot downgrade it.** The merge is an OR, never an AND.
   `test_llm_cannot_downgrade_clinical_detection` asserts this. Distress is
   likewise never downgraded.
3. **The gate blocks structurally.** `ROUTE_TO_PHARMACIST` and `ESCALATE_CASE` are
   in `HUMAN_REQUIRED_ACTIONS` and can never be automated regardless of score.

`SURFACE_ALTERNATIVE_OPTION` is deliberately non-clinical: home delivery, 90-day
fill, in-network pharmacy, generic-on-formulary. **No therapeutic substitution
suggestions**, ever — that is a prescriber's decision.

## Grounding

Every numeral in a generated response must appear in the grounded facts, the
retrieved evidence, or the deterministic draft. Anything else is an ungrounded
span; one regeneration is attempted, then the response falls back to the draft,
which is grounded by construction.

Numerals specifically, because a fabricated number is the most damaging failure
this system can produce: fluent, specific and wrong.

The draft's own template constants are trusted — and
`test_template_numerals_appear_in_the_policy_corpus` asserts that every number in
that static copy actually appears in the policy corpus, so the trust is verified
rather than assumed.

Retrieval is **plan-scoped**. Benefit summaries state different deductibles per
plan; an unscoped search returned another plan's numbers during development and
grounded an answer in them. `test_benefit_summaries_are_plan_scoped` prevents
regression.

## Disclosure boundaries

Members and advocates see different things, enforced by **separate endpoints
returning separate shapes** rather than a flag on one response. A field that is
never serialised cannot leak through a UI bug.

| Field | Member | Advocate |
|---|---|---|
| Explanation of their situation | yes | yes |
| Source document titles | yes | yes |
| Risk score and band | **no** | yes |
| Reason codes | **no** | yes |
| Rejected alternatives | **no** | yes |
| Model versions | **no** | yes |
| Gate reasoning | **no** | yes |

A member is told a person is involved; they are not told which internal rule
fired. There is also an explicit field allowlist and denylist for member-facing
text, checked by the verifier — the "understood, not watched" line enforced as
code rather than trusted to a prompt instruction.

## The Friction Ledger

Append-only by construction: no update, no delete. A human override is a *new*
record referencing the same `ledger_id`, so replaying the file always shows what
the system proposed before a person touched it.

Each record carries inputs, extracted signals, risk score and reasons, evidence
ids, selected action and confidence, rejected alternatives, gate verdict and
reasons, human decision, response text, citations, every component's version
string, and the full node-by-node execution trace.

Queryable views: per member, by verdict, by action, human review rate, override
rate and breakdown, top gate reasons, recurring root causes, friction trends.

## Human in the loop

Review is a graph primitive, not a wrapper. `interrupt()` suspends execution mid-
graph, persists state, and resumes from that exact node when a reviewer responds —
minutes or days later.

Reviewers accept, modify or reject. A modification can change the action or the
text. All three outcomes are recorded, so disagreement between humans and the
engine is measurable rather than anecdotal. That override rate is the metric that
would tell you, in production, whether the policy is drifting.

## Known gaps

**No fairness or subgroup analysis.** The synthetic population carries no
protected attributes, so none is performed. On real data this would be a release
blocker, not a follow-up. It is the largest gap in this project and is stated
plainly rather than buried.

**No adversarial testing** of the extraction layer — no prompt injection through
member text, no attempts to elicit ungrounded claims through crafted phrasing.

**Conflict detection is narrow.** It catches contradictory day counts, the one
failure this corpus can produce. A general contradiction detector is a research
project; a fake one would be worse than none.
