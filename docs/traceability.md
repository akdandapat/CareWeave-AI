# Transcript traceability

Every feature maps to a problem stated in the source material. Evidence tiers:
**[T]** stated in a transcript · **[I]** engineering inference · **[K]** general
technical knowledge.

## Problems identified

| # | Problem | Source |
|---|---|---|
| P1 | Healthcare asks the consumer to do too much: interpret letters, understand coverage, figure out cost, check status, call, repeat themselves — and still leave without a clear answer | T1 **[T]** |
| P2 | Answers exist somewhere but are not easy to find, easy to understand, or available at the right moment | T1 **[T]** |
| P3 | Service is reactive; the member must hit a wall before anything happens | T1, T2 **[T]** |
| P4 | Affordability is not the dollar amount — it is whether the member understands why it changed and what options exist | T1 **[T]** |
| P5 | Non-adherence has distinct causes (cost, side effects, access, confusion, life) and generic outreach treats them identically | T1 **[T]** |
| P6 | Employees search across multiple systems and start conversations under-informed | T1, T2 **[T]** |
| P7 | Prior authorization and claims are manual, multi-source and error-prone; denials cause real distress | T2 **[T]** |
| P8 | Claims need encounter, benefit and eligibility context assembled to process first-time-right | T2 **[T]** |
| P9 | Members lack visibility into their financial liability | T2 **[T]** |
| P10 | The experience is disconnected moments: a letter, a call, a counter issue, a portal message | T1 **[T]** |

## Feature to problem mapping

| Feature | Addresses | Basis |
|---|---|---|
| Journey store with unified event stream | P10, P2 | **[T]** fragmentation is named repeatedly; **[I]** the event-stream form is my design |
| `member_context()` — one call, one timestamp, every source | P6, P8 | **[T]** advocates should not make members repeat themselves |
| Friction risk model | P3 | **[T]** "AI can help us see those patterns, the timing, the signals and likely needs earlier than we could in a purely manual process" |
| Barrier-type extraction (cost / access / confusion / administrative / clinical) | P5 | **[T]** the transcript enumerates these causes almost exactly |
| Context engine's five questions | P1, P2 | **[I]** my formalisation of "turn complexity into guidance" **[T]** |
| Plan-scoped policy retrieval | P4, P9 | **[T]** answers must be grounded or trust erodes; **[I]** plan scoping is my fix for a real bug |
| Cost-change explanation with deductible-phase reasoning | P4, P9 | **[T]** affordability is about understanding, not the number |
| Closed twelve-action space | — | **[I]** entirely my design. The transcripts demand guard rails **[T]**; a closed enum is my implementation |
| Governance gate | P7 | **[T]** privacy, security, bias, explainability, accountability named as central |
| Suppression and frequency caps | P3 | **[T]** "proactive does not mean sending people a bunch of random messages" |
| Clinical routing to a pharmacist | P5 | **[T]** clinical questions are explicitly out of scope for advocates and assistants |
| Distress detection routing to a person | — | **[I]** my extension; the transcript's concern is AI becoming "a wall" **[T]** |
| Human-in-the-loop interrupt/resume | P7 | **[T]** "clear guard rails for when human or clinical expertise needs to be involved" |
| Advocate brief with risk and rejected alternatives | P6 | **[T]** AI should help people be more prepared and summarise what has happened |
| Friction Ledger | — | **[T]** accountability and explainability named; **[I]** the append-only ledger design is mine |
| Operations recurring-root-cause view | — | **[T]** ground-up innovation surfacing recurring issues |
| Member/advocate disclosure split | — | **[I]** my design decision; supports "helpful, not creepy" **[T]** |

## What I did *not* build, and why

- **Revenue cycle, coding, ambient transcription** (T2 **[T]**) — real and
  important, but a second domain. Building both shallowly would have been worse
  than building one properly. The same architecture applies; that claim is
  discussed in `architecture.md` and not demonstrated.
- **Telehealth and remote monitoring** (T2 **[T]**) — outside the pharmacy
  journey.
- **Anything about Optum's actual systems, datasets, models or metrics** — the
  transcripts describe problems and principles, not implementations. Every system
  detail here is my own construction over synthetic data.

## Claims I deliberately avoided

The transcripts contain a ballpark figure about clinician time split between
patients and administration, explicitly flagged in the source as illustrative
rather than measured. It is not used as a benchmark anywhere in this project, and
should not be cited as one.
