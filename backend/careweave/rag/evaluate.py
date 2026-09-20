"""RAG evaluation.

The question set is hand-written and hand-labelled against the corpus. That is
laborious and it is the point: a retrieval metric computed against
model-generated ground truth measures agreement with the model, not correctness.

Questions are phrased the way members and advocates actually phrase them --
"why does this cost more", not "describe the deductible provision" -- so the
metric reflects the vocabulary gap the retriever has to cross.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from careweave.config import SETTINGS
from careweave.rag.retriever import PolicyRetriever


@dataclass(frozen=True)
class RAGQuestion:
    qid: str
    question: str
    #: document ids that contain an acceptable answer
    relevant_documents: tuple[str, ...]
    #: section headings that specifically answer it (stricter check)
    relevant_sections: tuple[str, ...]
    persona: str  # "member" or "advocate"


EVAL_SET: list[RAGQuestion] = [
    RAGQuestion("Q01", "why does my medication cost more this month than last month",
                ("FAQ-001", "POL-FRM-001"),
                ("Why did my medication cost more this month than last month?", "Deductible"),
                "member"),
    RAGQuestion("Q02", "how long does it take to get a prior authorization decision",
                ("POL-PA-001", "FAQ-001"),
                ("Standard review timeframe",), "member"),
    RAGQuestion("Q03", "can I get an emergency supply while waiting for approval",
                ("POL-PA-001", "FAQ-001"), ("Interim supply",), "member"),
    RAGQuestion("Q04", "my pharmacy says the prescription needs approval, who submits it",
                ("POL-PA-001", "FAQ-001"), ("How a request is started",), "member"),
    RAGQuestion("Q05", "how long do I have to appeal a denial",
                ("POL-PA-001",), ("If a request is denied",), "member"),
    RAGQuestion("Q06", "does my claim rejection mean I owe money",
                ("POL-CLM-001", "FAQ-001"), ("What a rejection is",), "member"),
    RAGQuestion("Q07", "why was my refill rejected as too soon",
                ("POL-CLM-001",), ("Refill too soon",), "member"),
    RAGQuestion("Q08", "how long does home delivery take to arrive",
                ("POL-NET-001", "FAQ-001"), ("Home delivery",), "member"),
    RAGQuestion("Q09", "is home delivery cheaper than my retail pharmacy",
                ("POL-NET-001", "BEN-PLN-STANDARD", "BEN-PLN-VALUE", "BEN-PLN-PREMIER"),
                ("Home delivery",), "member"),
    RAGQuestion("Q10", "what happens if I fill at a pharmacy that is out of network",
                ("POL-NET-001",), ("Out-of-network retail",), "member"),
    RAGQuestion("Q11", "if I switch pharmacies do I need a new authorization",
                ("POL-NET-001", "FAQ-001"), ("Transferring a prescription",), "member"),
    RAGQuestion("Q12", "how long is my approval good for",
                ("POL-PA-001",), ("Approval duration",), "member"),
    RAGQuestion("Q13", "what is step therapy",
                ("POL-FRM-001",), ("Step therapy",), "member"),
    RAGQuestion("Q14", "my drug is not on the covered list, what are my options",
                ("POL-FRM-001",), ("Non-covered medications",), "member"),
    RAGQuestion("Q15", "why is my specialty medication a percentage instead of a flat copay",
                ("POL-FRM-001",), ("How cost share is calculated",), "member"),
    RAGQuestion("Q16", "the pharmacy said quantity limit exceeded",
                ("POL-FRM-001", "POL-CLM-001"), ("Quantity limits",), "member"),
    RAGQuestion("Q17", "can someone tell me if this medication will cause side effects",
                ("SOP-ADV-001", "FAQ-001"), ("Clinical questions",), "member"),
    RAGQuestion("Q18", "why did my costs go up in January",
                ("POL-FRM-001", "FAQ-001"), ("Deductible",), "member"),
    RAGQuestion("Q19", "how do I get money back if I paid out of pocket",
                ("POL-CLM-001",), ("Reprocessing",), "member"),
    RAGQuestion("Q20", "can I get a faster review if this is urgent",
                ("POL-PA-001",), ("Expedited review",), "member"),
    RAGQuestion("Q21", "what documentation does the plan ask for on a pending authorization",
                ("POL-PA-001",), ("Pending additional information",), "advocate"),
    RAGQuestion("Q22", "when should a case be escalated",
                ("SOP-ADV-001",), ("Escalation criteria",), "advocate"),
    RAGQuestion("Q23", "how should I handle a member who has already called about this",
                ("SOP-ADV-001",), ("Repeat contact",), "advocate"),
    RAGQuestion("Q24", "what do I need to review before opening a case",
                ("SOP-ADV-001",), ("Preparation",), "advocate"),
    RAGQuestion("Q25", "when is proactive outreach not allowed",
                ("SOP-OUT-001",), ("Suppression rules", "Escalation"), "advocate"),
    RAGQuestion("Q26", "how many outreach messages can a member get in a month",
                ("SOP-OUT-001",), ("Suppression rules",), "advocate"),
    RAGQuestion("Q27", "what must an outreach message contain",
                ("SOP-OUT-001",), ("Content standards",), "advocate"),
    RAGQuestion("Q28", "who resolves a prior authorization rejection, us or the prescriber",
                ("POL-CLM-001",), ("Who resolves what",), "advocate"),
    RAGQuestion("Q29", "what are the common reasons a pharmacy claim rejects",
                ("POL-CLM-001",), ("Common rejection reasons",), "advocate"),
    RAGQuestion("Q30", "which medications require prior authorization",
                ("POL-PA-001",), ("When prior authorization applies",), "advocate"),
    RAGQuestion("Q31", "what is the tier structure of the covered drug list",
                ("POL-FRM-001",), ("Tier structure",), "advocate"),
    RAGQuestion("Q32", "can the covered drug list change during the year",
                ("POL-FRM-001",), ("Changes to the covered drug list",), "advocate"),
    RAGQuestion("Q33", "how does the specialty pharmacy contact members",
                ("POL-NET-001",), ("Specialty pharmacy",), "advocate"),
    RAGQuestion("Q34", "what should be documented on every case",
                ("SOP-ADV-001",), ("Documentation",), "advocate"),
    RAGQuestion("Q35", "can a member submit their own prior authorization request",
                ("POL-PA-001",), ("How a request is started",), "advocate"),
    RAGQuestion("Q36", "what is a peer to peer review",
                ("POL-PA-001",), ("If a request is denied",), "advocate"),
    RAGQuestion("Q37", "how much is the deductible on the value plan",
                ("BEN-PLN-VALUE",), ("Deductible",), "advocate"),
    RAGQuestion("Q38", "what is the generic copay on the premier plan",
                ("BEN-PLN-PREMIER",), ("Cost share after the deductible is met",), "advocate"),
    RAGQuestion("Q39", "can a member ask for an early refill before travelling",
                ("POL-CLM-001",), ("Refill too soon",), "member"),
    RAGQuestion("Q40", "does an outreach ever deliver a denial decision",
                ("SOP-OUT-001",), ("Escalation",), "advocate"),
]


def _hit_documents(evidence, relevant: tuple[str, ...]) -> bool:
    return any(e.document_id in relevant for e in evidence)


def _hit_sections(evidence, relevant: tuple[str, ...]) -> bool:
    wanted = {s.lower() for s in relevant}
    return any(
        any(w in e.title.lower() for w in wanted) for e in evidence
    )


def _reciprocal_rank(evidence, relevant: tuple[str, ...]) -> float:
    for i, e in enumerate(evidence, start=1):
        if e.document_id in relevant:
            return 1.0 / i
    return 0.0


def evaluate_retrieval(retriever: PolicyRetriever, k_values=(1, 3, 5)) -> dict[str, Any]:
    per_question: list[dict[str, Any]] = []
    agg: dict[str, list[float]] = {}

    for q in EVAL_SET:
        top = retriever.retrieve(q.question, top_k=max(k_values))
        row: dict[str, Any] = {
            "qid": q.qid,
            "question": q.question,
            "persona": q.persona,
            "retrieved": [
                {"document_id": e.document_id, "section": e.title, "score": e.score}
                for e in top
            ],
            "mrr": _reciprocal_rank(top, q.relevant_documents),
        }
        for k in k_values:
            row[f"doc_hit@{k}"] = float(_hit_documents(top[:k], q.relevant_documents))
            row[f"section_hit@{k}"] = float(_hit_sections(top[:k], q.relevant_sections))
        sufficient, _ = retriever.sufficiency(top)
        row["sufficient"] = float(sufficient)
        per_question.append(row)

    numeric_keys = [k for k in per_question[0] if isinstance(per_question[0][k], float)]
    for key in numeric_keys:
        agg[key] = [r[key] for r in per_question]

    summary = {k: round(sum(v) / len(v), 4) for k, v in agg.items()}
    by_persona: dict[str, dict[str, float]] = {}
    for persona in {q.persona for q in EVAL_SET}:
        rows = [r for r in per_question if r["persona"] == persona]
        by_persona[persona] = {
            k: round(sum(r[k] for r in rows) / len(rows), 4) for k in numeric_keys
        }

    failures = [
        {"qid": r["qid"], "question": r["question"],
         "top_retrieved": r["retrieved"][:2]}
        for r in per_question
        if r[f"doc_hit@{max(k_values)}"] == 0.0
    ]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "index_version": retriever.index_version,
        "n_questions": len(EVAL_SET),
        "n_chunks": len(retriever.chunks),
        "n_documents": len(retriever.documents),
        "summary": summary,
        "by_persona": by_persona,
        "failures": failures,
        "per_question": per_question,
    }


def main() -> None:  # pragma: no cover
    from careweave.rag.retriever import load_retriever

    r = load_retriever()
    report = evaluate_retrieval(r)
    out = SETTINGS.paths.artifacts / "rag_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    s = report["summary"]
    print(f"Corpus: {report['n_documents']} documents, {report['n_chunks']} chunks")
    print(f"Questions: {report['n_questions']} hand-labelled\n")
    for k in (1, 3, 5):
        print(f"  doc_hit@{k}      {s[f'doc_hit@{k}']:.3f}     "
              f"section_hit@{k}  {s[f'section_hit@{k}']:.3f}")
    print(f"  MRR            {s['mrr']:.3f}")
    print(f"  sufficiency    {s['sufficient']:.3f}")
    print("\nBy persona:")
    for persona, m in report["by_persona"].items():
        print(f"  {persona:<9} doc_hit@3={m['doc_hit@3']:.3f} MRR={m['mrr']:.3f}")
    if report["failures"]:
        print(f"\n{len(report['failures'])} retrieval failures:")
        for f in report["failures"]:
            print(f"  {f['qid']}: {f['question']}")
    print(f"\nWritten to {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
