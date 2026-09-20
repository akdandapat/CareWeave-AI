# ML methodology

## The target

Predict whether a **member-visible friction event** occurs in the 14 days after a
scoring moment:

```
label = 1 if any of {member_contacted_support, case_escalated,
                     rx_abandoned_at_counter} in (as_of, as_of + 14 days]
```

Claim rejection is deliberately *not* in the label. It is an operational
precursor, and it is used as a feature. The label is "the member had to do
something about it, or gave up" — which is what the system exists to prevent.

**This is not a clinical prediction.** It predicts service-workflow friction. No
part of it estimates health outcomes, and the model is never positioned as
clinical decision support.

## Observation points

Scoring happens where the engine would actually run: a refill comes due, an
authorization is requested or blocked, a claim rejects, a cost changes, a member
makes contact, a prescription is written. Plus three sampled routine points per
member, so the model sees quiet periods and is not trained exclusively on crises.

Heavy utilisers are capped at 12 triggered points so they cannot dominate.

Result: **27,237 observations, 16.98% positive**.

Positive rate by trigger — a sanity check that the observation design is sensible:

| Trigger | n | Friction rate |
|---|---|---|
| Authorization blocked on information | 143 | 35.0% |
| Authorization requested | 967 | 34.2% |
| Member contacted support | 1,612 | 26.6% |
| Claim rejected | 2,900 | 26.4% |
| Cost changed | 1,032 | 21.0% |
| Prescription written | 1,995 | 19.0% |
| Refill due | 12,588 | 15.4% |
| Routine portal view | 6,000 | 8.6% |

## Splitting

**Chronological with a 14-day purge between folds.**

Why not member-disjoint? Every member is present across the whole simulation
window, so requiring disjointness would empty the later folds — and it models the
wrong deployment. In production the engine scores members it has already seen,
repeatedly, over time. The honest question is "can it predict the future for a
known member", not "can it generalise to a stranger".

The genuine risk in that setup is **adjacency**: a training row whose 14-day label
window overlaps a test row's feature window shares outcome information. A purge
gap of one full horizon removes it. 1,625 rows are discarded to the gap.

A member-disjoint split is reported *separately* as a cold-start robustness check
(PR-AUC 0.433 on a 0.184 base rate). It answers a different question and is not
directly comparable to the primary figure.

## Why PR-AUC is the headline

At a 15–17% positive rate, ROC-AUC flatters. Precision-recall reflects what an
outreach queue actually experiences: of the cases you flag, how many were real.

| Model | Test PR-AUC | Test ROC-AUC |
|---|---|---|
| Majority baseline | 0.147 | 0.500 |
| Logistic regression | 0.329 | 0.771 |
| LightGBM | 0.398 | 0.788 |

The linear baseline is reported because it is informative: most of the signal is
linear, and gradient boosting adds about 0.07 PR-AUC on top. A project that only
reports its best model hides that comparison.

## Calibration

The governance gate thresholds on probability, so an uncalibrated score makes
those thresholds meaningless. Isotonic regression is fitted on the validation
fold and evaluated on the untouched test fold.

Calibration costs a little discrimination (PR-AUC 0.398 → 0.379) and buys usable
probabilities: **ECE 0.048, Brier 0.110**. That trade is correct for this system —
a well-ranked but wrongly-scaled score would cause the gate to fire at the wrong
rate.

Band behaviour on held-out data:

| Band | Threshold | Share | Observed friction |
|---|---|---|---|
| Low | < 0.25 | 64.4% | 6.0% |
| Medium | 0.25–0.50 | 31.5% | 26.8% |
| High | ≥ 0.50 | 4.1% | 58.1% |

Monotonic and well-separated, which is the property the gate depends on.

## Operating points

Because capacity is finite, the decision-relevant question is the precision of a
queue you can staff:

| Queue | Precision | Recall | Lift |
|---|---|---|---|
| Top 5% | 53.6% | 18.2% | 3.65× |
| Top 10% | 45.5% | 30.9% | 3.09× |
| Top 20% | 34.0% | 46.2% | 2.31× |
| Top 30% | 32.2% | 65.6% | 2.19× |

## Explainability, and what it is not

Two separate mechanisms, deliberately:

- **Reason codes** are derived from the feature row by explicit rules,
  independent of the model. They explain the *situation*, stay stable across
  retraining, and are what an advocate and an auditor need.
- **Contributions** are leave-one-out perturbations against a population median.
  Model-agnostic, cheap, and faithful to what the model does with that row. **This
  is not SHAP** and is not a formal attribution method.

Top features by permutation importance in average precision:

| Feature | Δ AP |
|---|---|
| Unresolved contacts, 90 days | +0.047 |
| Days since last rejection | +0.040 |
| Days since last contact | +0.039 |
| Portal views, 30 days | +0.033 |
| Trigger type | +0.016 |
| Prior-authorization rejections, 90 days | +0.016 |
| Refill runway days | +0.012 |

Portal views ranking fourth is the confounder working as designed: engaged
members self-serve and contact support less at equal severity.

## Why the ceiling is where it is

The simulator draws friction from a hazard that mixes observable operational
severity with **latent traits the model never sees** — health literacy, contact
propensity, cost sensitivity, persistence, digital engagement — plus Gaussian
noise. That irreducible component is deliberate. A model scoring 0.95 on this
task would be evidence of a leak, not of skill.
