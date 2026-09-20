# Roadmap

Ordered by what would most change the strength of the project's claims.

## 1. Fairness and subgroup analysis
The largest gap. Add protected attributes to the synthetic population, then
measure whether friction predictions, action selection and gate outcomes differ
systematically by subgroup. Report disparate impact alongside PR-AUC. On real
data this is a release blocker; here it is missing entirely.

## 2. Counterfactual replay
The evaluation that would actually test the product thesis. Replay the population
twice — engine off, engine on — and measure avoidable contacts, time-to-clarity,
and abandonment. Requires modelling how a member responds to an intervention,
which is a modelling commitment that should be stated explicitly rather than
buried. Design sketch:

- For each proactive message the engine would send, resample the friction hazard
  with a response term whose magnitude is a declared parameter.
- Report results as a sensitivity curve across that parameter, not a single
  number, because the single number would be an artefact of the assumption.

## 3. Held-out human-written text
Collect or write member messages independently of the templates, label them, and
re-score extraction. The current 99% is not a number to defend.

## 4. Repeated temporal backtesting
Calibration is measured on one fold. Roll the split forward across several
periods and report calibration stability over time.

## 5. Postgres and pgvector
Move the store to SQL with the point-in-time guarantee expressed as a `WHERE`
clause, and the retriever to hybrid dense/lexical at a corpus size where the
dense half matters. Swap `MemorySaver` for `PostgresSaver` so suspended cases
survive restarts.

## 6. Adversarial testing
Prompt injection through member text, attempts to elicit ungrounded claims
through crafted phrasing, and attempts to route around clinical detection.

## 7. Engineering debt
- Register domain models with LangGraph's msgpack serde rather than suppressing
  the deserialization notices.
- Cache the median row used for contribution estimates across scorer instances;
  it currently costs about 350ms per scored case.
- Streaming node-by-node execution to the UI is implemented in the engine
  (`CareWeaveEngine.stream`) but the frontend currently renders the trace after
  completion rather than live.
