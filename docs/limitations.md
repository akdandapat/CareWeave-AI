# Limitations

Read this before believing any number in this repository.

## The data is synthetic, and I wrote the generator

This is the limitation that qualifies every other result. The friction the model
predicts was produced by a hazard function I designed. The relationship the model
recovers is, at bottom, a relationship I put there.

Three deliberate choices make that recovery non-trivial rather than circular:

- **Latent traits are unobservable.** Health literacy, contact propensity, cost
  sensitivity, persistence and digital engagement drive behaviour but never
  appear as features. The model has to infer from behavioural proxies.
- **A confounder works against the obvious signal.** Digitally engaged members
  self-serve, so at equal operational severity they contact support *less*. A
  model keyed only on severity is miscalibrated for them.
- **Friction is drawn from a probability, not a rule.** No feature is a giveaway,
  and irreducible noise sets a real ceiling.

That is why test PR-AUC is 0.379 and not 0.95. But it remains a simulation.
**Nothing here estimates performance on real claims data.**

## Specific things the numbers do not mean

**Intent accuracy of 99% is not extraction accuracy.** Member text is generated
from a finite template set, and the extraction rules were tuned after seeing
failures on that text. It shows the pipeline and the evaluation harness work.
Real member language is far more varied, contains typos, code-switching, partial
sentences and domain vocabulary members invent. Honest validation needs held-out
human-written text, which this project does not have.

**Retrieval scores are on a 61-chunk corpus.** Ten documents is small enough that
BM25 with synonym expansion does well. At ten thousand documents, lexical
retrieval degrades and dense embeddings stop being optional. The hybrid interface
exists for that reason, but the dense half is untested at scale here.

**Clinical-question recall of 100% is on 209 examples from one template.** The
rules catch the phrasings in the test set and the phrasings I thought to add. A
real deployment would find phrasings neither of us anticipated. The detector is a
floor, not a guarantee — which is why the governance gate treats it as one input
among several rather than the sole control.

**Calibration error of 0.048 is measured on one temporal fold.** No repeated
backtesting across multiple periods, so the stability of that calibration over
time is unknown.

## Architectural simplifications

- **In-memory storage.** Parquet and JSONL loaded into a process. Real scale needs
  Postgres with pgvector, and the point-in-time guarantee becomes a query concern
  rather than a Python filter.
- **`MemorySaver` checkpointing.** Suspended cases do not survive a process
  restart. `SqliteSaver` or `PostgresSaver` is a one-line swap and is what any
  real deployment would use.
- **The outreach suppression ledger is in-process.** In production it is a shared
  table, and concurrent workers would need it to be.
- **No authentication or authorisation.** The member/advocate disclosure boundary
  is enforced by separate endpoints returning separate shapes, which is the right
  structure, but there is no identity layer behind it.
- **Contribution estimates are leave-one-out against a population median**, not
  SHAP. Faithful to what the model does with a given row, cheaper, and adequate
  for an advocate sanity-check — but not a formal attribution method.
- **Conflict detection is narrow by design.** It catches the specific failure this
  corpus can produce: two chunks quoting different day counts for the same
  concept. A general contradiction detector is a research project, and a fake one
  would be worse than none.

## What real healthcare data would change

- **Class imbalance would be far worse.** A 17% positive rate is generous.
- **Labels would be noisy and delayed.** "Did the member contact support" is clean
  here; in reality it spans channels, systems and identity-resolution problems.
- **Fairness auditing would be mandatory, not optional.** This project has no
  protected attributes in its synthetic population and therefore performs no
  subgroup analysis. On real data, measuring whether friction predictions differ
  systematically by race, language, disability status or geography would be a
  release blocker, not a follow-up. Its absence here is a genuine gap.
- **Regulatory scope.** Anything touching coverage determinations sits under
  utilisation-management rules. This prototype explains determinations; it does
  not make them, and that boundary would need legal review before it moved.
- **Privacy.** No PHI handling, de-identification, minimum-necessary analysis, or
  audit controls beyond the ledger.

## What I would do next, in order

1. Fairness and subgroup analysis — the largest gap.
2. Held-out human-written text for extraction validation.
3. Repeated temporal backtesting for calibration stability.
4. Postgres and pgvector, with the point-in-time guarantee expressed in SQL.
5. A counterfactual replay harness: run the population with the engine off and
   on, and measure avoidable contacts and time-to-clarity. The design is in
   `docs/roadmap.md`; it is the evaluation that would actually test the thesis.
