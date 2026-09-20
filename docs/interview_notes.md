# Interview notes

Short answers to the questions this design invites. Each is defensible from the
code, not from the pitch.

**Why isn't this a chatbot?**
A chatbot decides what to say. This decides what to *do*, from twelve enum-typed
actions, using rules plus a calibrated score plus evidence contracts — then hands
the chosen action to a language model to phrase. Delete the language model and
the system still works; it just writes worse prose. That is the test.

**Why ML and an LLM together?**
They answer different questions. "Is this member about to have a problem" has no
right answer in any document — it is a probabilistic ranking over interacting
features, which is ML. "What is this member actually asking" and "how do I say
this to a worried person" are language problems. Using either for the other's job
is the mistake.

**Where is deterministic logic, and why?**
Formulary lookups, PA requirements, suppression windows, frequency caps,
confidence floors, clinical blocking, the entire governance gate. Anything with a
right answer, and anything that is a safety control. A control that is itself
stochastic is not a control.

**Why RAG rather than putting policy in the prompt?**
Citations. The member sees which document the answer came from, and the verifier
checks assertions against retrieved text. Also plan scoping: benefit summaries
differ per plan, and an unscoped answer grounded in the wrong plan's deductible
is confidently wrong. That was a real bug, caught in development and now covered
by a test.

**Why agents?**
Not for the label. A prior authorization opens Tuesday and resolves the following
Monday — the workflow has to suspend for days and survive a process restart.
Human review has to pause mid-execution and resume from the same node. Those two
properties need durable orchestration. The rest of the graph is a state machine,
and I say so.

**How do you stop it hallucinating?**
Layered. The action is chosen before generation, so the model cannot invent one.
Facts come from structured records with provenance. A deterministic draft is
composed first and is the answer in the offline profile. Every numeral in the
output is checked against the evidence; failures fall back to the draft. And the
static templates' own numbers are asserted against the policy corpus, so the
trusted baseline is verified rather than assumed.

**What happens when sources conflict?**
The gate holds and routes to a human. Detection is narrow on purpose — it catches
contradictory day counts, which is the failure this corpus can actually produce.
A general contradiction detector would be a research project; a fake one would be
worse than none.

**Your extraction accuracy is 99%. Really?**
No. It is optimistically biased and the report says so in a field called
`VALIDITY_CAVEAT`. The text is templated and the rules were tuned after seeing
failures on that text. It demonstrates the pipeline and the harness. The number I
would defend is clinical-question recall, and even that is 209 examples.

**Why is PR-AUC only 0.38?**
Because the simulator draws friction from a hazard that includes latent traits
the model never sees, plus noise. That ceiling is deliberate. A 0.95 on this task
would be evidence of a leak. Against a 0.147 base rate this is 2.58× lift, and
the top 5% queue runs at 53.6% precision — which is the number operations would
actually care about.

**How do you know there's no leakage?**
Four mechanisms. One time filter, in `JourneyStore`, that everything goes
through. A source-level test asserting the feature builder never calls
`future_events`. A purge gap of one full horizon between folds. And status
rollback — a prior authorization decided after `as_of` reads as pending, which is
the subtlest leak available here and the one a test specifically covers.

**What if the model is wrong about someone?**
It suppresses more often than it acts: 64% of cases land in the low band and most
terminate without contact. When it does act, the action is bounded, the message
is grounded, and there is always an unblocked path to a person. High-impact and
low-confidence cases never reach the member at all.

**What does this cost to run?**
The offline profile costs nothing — no API calls at all. That is why it is the
default. With a model configured, one call per case for extraction and up to two
for generation, and only on cases that pass the risk gate, which is about a third
of them.

**What would break first at real scale?**
Retrieval. Sixty-one chunks is small enough that BM25 with synonym expansion does
well; at ten thousand documents lexical retrieval degrades and dense embeddings
stop being optional. Second would be the in-process suppression ledger, which
needs to be a shared table the moment there is more than one worker.

**What's the biggest weakness?**
No fairness analysis. The synthetic population has no protected attributes, so
there is no subgroup evaluation. On real data, checking whether friction
predictions differ systematically by language, disability status or geography
would be a release blocker rather than a follow-up. I would rather name that than
have it found.

**Why should anyone believe results on data you generated?**
They shouldn't, entirely — and `docs/limitations.md` opens with that. What the
synthetic data does support is that the *pipeline* is correct: leakage-safe
features, honest splits, calibrated probabilities, hand-labelled retrieval
evaluation, ground-truth extraction scoring, and tests for the invariants that
would otherwise fail silently. Those transfer. The numbers do not.
