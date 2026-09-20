"""Extraction evaluation against generator ground truth.

The simulator recorded the true intent behind every interaction before any text
was written, and knows exactly which documents had a clinical question inserted.
That gives real labels rather than model-graded ones, so these numbers mean
something.

Clinical-question recall is reported separately and treated as the metric that
matters: a missed clinical question routes a safety question into an automated
answer, which is the highest-severity failure this component can produce.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from typing import Any

from careweave.config import SETTINGS
from careweave.data.store import JourneyStore, get_store
from careweave.domain.enums import DocumentType, MemberIntent
from careweave.nlp.extract import ExtractionInput, deterministic_extract

#: The exact sentence the generator inserts when a contact carries a clinical
#: question. Used as ground truth only -- the extractor never sees it.
_CLINICAL_MARKER = "is it normal to feel dizzy"


def evaluate_extraction(store: JourneyStore, *, limit: int | None = None) -> dict[str, Any]:
    known_drugs = {d.name for d in store.drugs.values()}

    interactions_by_doc: dict[str, Any] = {}
    for mid in store.member_ids:
        for i in store._interactions_by_member.get(mid, []):
            if i.document_id:
                interactions_by_doc[i.document_id] = i

    rows: list[dict[str, Any]] = []
    for doc in store.member_documents.values():
        if doc.document_type not in (
            DocumentType.MEMBER_MESSAGE, DocumentType.CALL_TRANSCRIPT
        ):
            continue
        interaction = interactions_by_doc.get(doc.document_id)
        if interaction is None:
            continue

        signals = deterministic_extract(
            ExtractionInput(text=doc.body, document_ids=[doc.document_id]), known_drugs
        )
        true_clinical = _CLINICAL_MARKER in doc.body.lower()
        rows.append(
            {
                "document_id": doc.document_id,
                "true_intent": interaction.intent.value,
                "pred_intent": signals.intent.value,
                "intent_correct": int(signals.intent == interaction.intent),
                "true_clinical": int(true_clinical),
                "pred_clinical": int(signals.clinical_question_detected),
                "barrier": signals.barrier.value,
                "urgency": signals.urgency.value,
                "sentiment": signals.sentiment.value,
                "confidence": signals.confidence,
            }
        )
        if limit and len(rows) >= limit:
            break

    n = len(rows)
    # a document carrying a clinical question has two valid intent readings; the
    # primary reason for contact and the clinical aside. Scored leniently on
    # intent, strictly on clinical detection.
    scored = [r for r in rows if not r["true_clinical"]]
    intent_acc = sum(r["intent_correct"] for r in scored) / max(1, len(scored))

    tp = sum(1 for r in rows if r["true_clinical"] and r["pred_clinical"])
    fn = sum(1 for r in rows if r["true_clinical"] and not r["pred_clinical"])
    fp = sum(1 for r in rows if not r["true_clinical"] and r["pred_clinical"])
    clinical_recall = tp / max(1, tp + fn)
    clinical_precision = tp / max(1, tp + fp)

    per_intent: dict[str, dict[str, float]] = {}
    for intent in {r["true_intent"] for r in scored}:
        sub = [r for r in scored if r["true_intent"] == intent]
        per_intent[intent] = {
            "n": len(sub),
            "accuracy": round(sum(r["intent_correct"] for r in sub) / len(sub), 4),
        }

    confusions = Counter(
        (r["true_intent"], r["pred_intent"]) for r in scored if not r["intent_correct"]
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "extractor": "deterministic (offline profile)",
        "VALIDITY_CAVEAT": (
            "This number is optimistically biased and should not be quoted as a "
            "general extraction accuracy. The member text is generated from a "
            "finite set of templates, and the rule patterns were tuned after "
            "seeing failures on that same text. It measures whether the "
            "extractor handles the distribution it was built for; it does not "
            "estimate performance on real member language, which is far more "
            "varied. The honest reading is that the pipeline and the evaluation "
            "harness work, not that extraction is solved. Real validation needs "
            "held-out human-written text -- see docs/limitations.md."
        ),
        "n_documents": n,
        "intent_accuracy": round(intent_acc, 4),
        "intent_accuracy_note": (
            "scored on documents without an inserted clinical aside, which have "
            "a single unambiguous ground-truth intent"
        ),
        "clinical_question": {
            "recall": round(clinical_recall, 4),
            "precision": round(clinical_precision, 4),
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "note": (
                "Recall is the metric that matters. A false positive routes a "
                "member to a pharmacist unnecessarily; a false negative lets an "
                "automated answer respond to a safety question."
            ),
        },
        "per_intent": dict(sorted(per_intent.items(), key=lambda kv: -kv[1]["n"])),
        "top_confusions": [
            {"true": t, "predicted": p, "n": c} for (t, p), c in confusions.most_common(8)
        ],
        "barrier_distribution": dict(Counter(r["barrier"] for r in rows)),
        "urgency_distribution": dict(Counter(r["urgency"] for r in rows)),
        "sentiment_distribution": dict(Counter(r["sentiment"] for r in rows)),
        "mean_confidence": round(sum(r["confidence"] for r in rows) / max(1, n), 4),
    }


def main() -> None:  # pragma: no cover
    store = get_store()
    report = evaluate_extraction(store)
    out = SETTINGS.paths.artifacts / "extraction_report.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"Documents evaluated: {report['n_documents']:,}")
    print(f"Intent accuracy:     {report['intent_accuracy']:.3f}  "
          "(optimistically biased -- see VALIDITY_CAVEAT in the report)")
    c = report["clinical_question"]
    print(f"Clinical recall:     {c['recall']:.3f}  (precision {c['precision']:.3f}, "
          f"{c['false_negatives']} missed of {c['true_positives'] + c['false_negatives']})")
    print("\nPer-intent accuracy:")
    for intent, m in report["per_intent"].items():
        print(f"  {intent:<28} n={m['n']:>5}  acc={m['accuracy']:.3f}")
    print("\nTop confusions:")
    for c2 in report["top_confusions"][:5]:
        print(f"  {c2['true']} -> {c2['predicted']}  ({c2['n']})")
    print(f"\nWritten to {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
