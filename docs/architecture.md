# Architecture

## Layers

| Layer | Module | Responsibility |
|---|---|---|
| Domain | `domain/enums.py`, `domain/models.py` | Closed vocabularies and typed shapes. The only things that cross module boundaries. |
| Data | `data/generator/`, `data/store.py` | Synthetic ecosystem; point-in-time access. |
| Learning | `ml/features.py`, `ml/train.py`, `ml/score.py` | Leakage-safe features, training, runtime scoring with reason codes. |
| Retrieval | `rag/retriever.py`, `rag/evaluate.py` | Plan-scoped policy grounding with citations. |
| Understanding | `nlp/extract.py` | Free text to a typed `SignalSet`. |
| Reasoning | `graph/context_engine.py`, `graph/policy.py` | Five typed questions; action selection over the closed set. |
| Control | `governance/gate.py`, `governance/ledger.py` | Deterministic gate; append-only audit. |
| Orchestration | `graph/nodes.py`, `graph/build.py` | LangGraph state machine with checkpointing and interrupts. |
| Interface | `api/main.py`, `frontend/index.html` | Three surfaces over one intelligence layer. |

## The one invariant everything depends on

```python
store.events(member_id, as_of)   # never returns anything after as_of
```

Every consumer — feature builder, context engine, advocate view, evaluation
harness — goes through `JourneyStore`. Centralising the time filter means leakage
is prevented in one auditable place rather than reimplemented, and eventually
mis-implemented, in five.

The subtlest case is authorization status. A prior authorization decided *after*
`as_of` must read as pending, not with its eventual outcome. `authorizations_asof`
rolls the record back. Without it, a feature built before a decision would encode
the result of that decision, and every metric downstream would be inflated by an
amount nobody could see.

## Data flow

```
generator ──► parquet + jsonl ──► JourneyStore (in memory)
                                        │
   ┌────────────────────────────────────┼──────────────────────────┐
   ▼                                    ▼                          ▼
build_features(as_of)          member_context(as_of)      documents(as_of)
   │                                    │                          │
   ▼                                    ▼                          ▼
FrictionScorer ─────────────► ContextEngine ◄──────────── SignalExtractor
                                        │
                            PolicyRetriever (plan-scoped)
                                        │
                                  ActionPolicy
                                        │
                                GovernanceGate
                          ┌─────────────┼─────────────┐
                     suppress        escalate       allow
                          │             │             │
                          │        interrupt()    Responder
                          │             │             │
                          └──────► FrictionLedger ◄───┘
```

## Design decisions worth defending

**Nodes are thin.** Every graph node reads state, calls one plain-Python
component, writes state. All real logic lives in importable modules, so the test
suite exercises decision logic without a graph runtime, and swapping the
orchestrator would be mechanical.

**Candidate generation is separate from selection.** The context engine proposes
generously; the policy narrows with evidence contracts. This is what lets the
ledger record what was considered *and rejected*, not only what was chosen.

**The policy knows the channel.** Proactive outreach has a narrower permitted set,
filtered at selection time rather than suppressed afterwards. Choosing an
undeliverable action and then suppressing it produces a silent no-op and an
audit trail that misrepresents the reasoning.

**Two corpora, never merged.** Policy documents are retrievable and citable.
Member letters, transcripts and notes are not in that index — they reach the
engine through the structured store, where they are time-filtered and
access-controlled. Merging them would let a generated answer cite one member's
correspondence while answering another's question.

**Generation is the last step and the narrowest.** The responder receives a fixed
action, fixed facts and fixed evidence. It cannot select, substitute or invent.
The deterministic draft is composed first and *is* the answer in the offline
profile; with an LLM configured, the draft becomes the content to be rewritten,
so the model supplies fluency rather than facts.

## What changes in production

| Here | Production | Why it is a swap, not a rewrite |
|---|---|---|
| Parquet in memory | Postgres | `JourneyStore` is the only reader; `as_of` becomes a `WHERE` clause |
| BM25 + numpy | pgvector hybrid | `PolicyRetriever.retrieve` keeps its signature |
| `MemorySaver` | `PostgresSaver` | one line in `build_graph` |
| In-process outreach ledger | shared table | `OutreachLedger` has three methods |
| JSONL ledger | append-only table | `FrictionLedger` already has no update path |
| No auth | identity + RBAC | surfaces are already separate endpoints with separate shapes |

## Generalisation claim, stated honestly

The revenue-cycle and coding workflows described in the second source share this
shape: fragmented state, a silent countdown, a decision that needs multiple
records reconciled, and a point where a human must own the outcome. The same
architecture would apply — journey store, risk model, evidence contracts, closed
action space, gate, ledger.

That is an argument, not a demonstration. This repository builds the pharmacy
benefit journey and only that. Building two domains shallowly would have been
worse than building one properly.
