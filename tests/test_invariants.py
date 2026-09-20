"""Invariant tests.

These are not coverage tests. Each one asserts a property that, if it broke,
would make the system unsafe or its numbers dishonest -- and would break
silently, without any exception being raised. That is the category of failure
worth spending tests on.
"""

from __future__ import annotations

import inspect
import re
from datetime import datetime, timedelta

import pytest

from careweave.data.store import get_store
from careweave.domain.enums import (
    HUMAN_REQUIRED_ACTIONS,
    PROACTIVE_ELIGIBLE_ACTIONS,
    ActionType,
    EventType,
    GateVerdict,
    MemberIntent,
    PAStatus,
    ReasonCode,
    RiskBand,
    Sentiment,
)
from careweave.domain.models import (
    ActionDecision,
    MemberContext,
    RiskAssessment,
    SignalSet,
)
from careweave.governance.gate import GovernanceGate, OutreachLedger
from careweave.ml import features as feature_module


@pytest.fixture(scope="session")
def store():
    return get_store()


# ===========================================================================
# 1. Temporal leakage
# ===========================================================================


class TestNoLeakage:
    def test_feature_builder_never_reads_the_future(self):
        """``build_features`` must not call ``future_events``, directly or not.

        Source-level assertion rather than a behavioural one: a future read
        introduced by a later edit would still produce plausible numbers, so
        nothing else would catch it.
        """
        src = inspect.getsource(feature_module.build_features)
        assert "future_events" not in src
        assert "future" not in src.lower().replace("future_events", "")

    def test_only_label_builder_reads_the_future(self):
        module_src = inspect.getsource(feature_module)
        # actual call sites only -- prose mentions in docstrings are fine and are
        # in fact where the rule is documented
        callers = [
            line.strip() for line in module_src.splitlines()
            if re.search(r"\.future_events\s*\(", line)
        ]
        assert len(callers) == 1, f"unexpected future_events call sites: {callers}"
        assert "future_events" in inspect.getsource(feature_module.build_label)

    def test_latent_traits_are_not_loadable_from_the_store(self, store):
        """The generator's latent traits must be unreachable from the store."""
        assert not hasattr(store, "traits")
        store_src = inspect.getsource(type(store))
        assert "_latent_traits" not in store_src

    def test_latent_trait_names_are_not_features(self, store):
        from careweave.ml.features import ObservationPoint, build_features

        point = ObservationPoint(
            store.member_ids[0], datetime(2026, 1, 15), EventType.REFILL_DUE, None
        )
        keys = set(build_features(store, point))
        forbidden = {
            "health_literacy", "cost_sensitivity", "digital_engagement",
            "contact_propensity", "persistence",
        }
        assert not (keys & forbidden)

    def test_events_respects_as_of(self, store):
        mid = next(
            m for m in store.member_ids if len(store._events_by_member.get(m, [])) > 20
        )
        cutoff = datetime(2025, 9, 1)
        assert all(e.occurred_at <= cutoff for e in store.events(mid, cutoff))

    def test_authorization_status_is_rolled_back(self, store):
        """A PA decided after ``as_of`` must read as pending, not decided.

        The subtlest leak in the system: without the rollback, a feature built
        before a decision would encode the outcome of that decision.
        """
        for mid in store.member_ids[:300]:
            for pa in store._pa_by_member.get(mid, []):
                if pa.requested_on is None or pa.decided_on is None:
                    continue
                if pa.decided_on <= pa.requested_on:
                    continue
                mid_point = datetime.combine(
                    pa.requested_on + (pa.decided_on - pa.requested_on) / 2,
                    datetime.min.time(),
                )
                rolled = [
                    a for a in store.authorizations_asof(mid, mid_point)
                    if a.pa_id == pa.pa_id
                ]
                if rolled:
                    assert rolled[0].status == PAStatus.PENDING
                    assert rolled[0].decided_on is None
                    assert rolled[0].denial_reason is None
                    return
        pytest.skip("no straddling authorization found in the sampled members")


# ===========================================================================
# 2. The closed action space
# ===========================================================================


class TestClosedActionSpace:
    def test_every_action_has_an_evidence_contract(self):
        from careweave.graph.policy import CONTRACTS

        assert set(CONTRACTS) == set(ActionType)

    def test_generator_has_copy_for_every_action(self):
        from careweave.graph.generate import _CLOSERS, _OPENERS

        for action in ActionType:
            if action == ActionType.SUPPRESS_OUTREACH:
                continue  # suppression produces no member-facing text by design
            assert action in _OPENERS, f"no opener for {action}"
            assert action in _CLOSERS, f"no closer for {action}"

    def test_policy_only_returns_enum_actions(self, store):
        from careweave.graph.context_engine import ContextEngine
        from careweave.graph.policy import ActionPolicy
        from careweave.ml.score import FrictionScorer

        engine = ContextEngine(store)
        scorer = FrictionScorer(store)
        policy = ActionPolicy()
        checked = 0
        for mid in store.member_ids[:40]:
            as_of = datetime(2026, 3, 1)
            ctx = store.member_context(mid, as_of)
            risk, _ = scorer.score(mid, as_of, EventType.REFILL_DUE)
            signals = SignalSet()
            frame, _ = engine.build_frame(
                ctx, signals, risk, EventType.REFILL_DUE, as_of
            )
            decision = policy.select(
                frame=frame, ctx=ctx, signals=signals, risk=risk, evidence=[]
            )
            assert isinstance(decision.action, ActionType)
            checked += 1
        assert checked == 40

    def test_proactive_eligible_is_a_strict_subset(self):
        assert PROACTIVE_ELIGIBLE_ACTIONS < set(ActionType)
        # nothing requiring a human may be sent as unsolicited outreach
        assert not (PROACTIVE_ELIGIBLE_ACTIONS & HUMAN_REQUIRED_ACTIONS)


# ===========================================================================
# 3. Governance gate
# ===========================================================================


def _ctx(store, mid="MBR-000001", as_of=datetime(2026, 3, 1)) -> MemberContext:
    return store.member_context(mid, as_of)


def _risk(band=RiskBand.MEDIUM, score=0.35) -> RiskAssessment:
    return RiskAssessment(
        score=score, band=band, reasons=[], model_version="test", calibrated=True
    )


def _decision(action=ActionType.EXPLAIN_AUTH_STATUS, confidence=0.9) -> ActionDecision:
    return ActionDecision(action=action, confidence=confidence, reasons=[])


class TestGovernanceGate:
    def test_clinical_question_always_escalates(self, store):
        gate = GovernanceGate()
        result = gate.evaluate(
            decision=_decision(confidence=0.99),
            ctx=_ctx(store),
            signals=SignalSet(clinical_question_detected=True, confidence=0.9),
            risk=_risk(RiskBand.LOW, 0.02),
            evidence_sufficient=True, evidence_conflict=False,
            as_of=datetime(2026, 3, 1), is_proactive=False,
        )
        assert result.verdict == GateVerdict.ESCALATE
        assert result.requires_human
        assert ReasonCode.CLINICAL_TOPIC_DETECTED in result.reasons

    def test_distress_escalates(self, store):
        gate = GovernanceGate()
        result = gate.evaluate(
            decision=_decision(), ctx=_ctx(store),
            signals=SignalSet(sentiment=Sentiment.DISTRESSED),
            risk=_risk(), evidence_sufficient=True, evidence_conflict=False,
            as_of=datetime(2026, 3, 1), is_proactive=False,
        )
        assert result.verdict == GateVerdict.ESCALATE
        assert result.requires_human

    def test_insufficient_evidence_holds(self, store):
        gate = GovernanceGate()
        result = gate.evaluate(
            decision=_decision(), ctx=_ctx(store), signals=SignalSet(),
            risk=_risk(), evidence_sufficient=False, evidence_conflict=False,
            as_of=datetime(2026, 3, 1), is_proactive=False,
        )
        assert result.verdict == GateVerdict.HOLD
        assert ReasonCode.EVIDENCE_INSUFFICIENT in result.reasons

    def test_low_confidence_holds(self, store):
        gate = GovernanceGate()
        result = gate.evaluate(
            decision=_decision(confidence=0.2), ctx=_ctx(store), signals=SignalSet(),
            risk=_risk(), evidence_sufficient=True, evidence_conflict=False,
            as_of=datetime(2026, 3, 1), is_proactive=False,
        )
        assert result.verdict == GateVerdict.HOLD

    def test_human_required_actions_cannot_be_automated(self, store):
        gate = GovernanceGate()
        for action in HUMAN_REQUIRED_ACTIONS:
            result = gate.evaluate(
                decision=_decision(action, confidence=0.99), ctx=_ctx(store),
                signals=SignalSet(), risk=_risk(RiskBand.HIGH, 0.9),
                evidence_sufficient=True, evidence_conflict=False,
                as_of=datetime(2026, 3, 1), is_proactive=False,
            )
            assert result.requires_human, f"{action} was automated"

    def test_frequency_cap_suppresses(self, store):
        ledger = OutreachLedger()
        gate = GovernanceGate(ledger)
        as_of = datetime(2026, 3, 1)
        ctx = _ctx(store)
        for i in range(3):
            ledger.record(
                ctx.member_id, as_of - timedelta(days=i + 1),
                ActionType.PROVIDE_NEXT_STEP,
            )
        result = gate.evaluate(
            decision=_decision(ActionType.EXPLAIN_COST_CHANGE), ctx=ctx,
            signals=SignalSet(), risk=_risk(RiskBand.HIGH, 0.8),
            evidence_sufficient=True, evidence_conflict=False,
            as_of=as_of, is_proactive=True,
        )
        assert result.verdict == GateVerdict.SUPPRESS

    def test_topic_suppression_window(self, store):
        ledger = OutreachLedger()
        gate = GovernanceGate(ledger)
        as_of = datetime(2026, 3, 1)
        ctx = _ctx(store)
        ledger.record(ctx.member_id, as_of - timedelta(days=3),
                      ActionType.EXPLAIN_COST_CHANGE)
        result = gate.evaluate(
            decision=_decision(ActionType.EXPLAIN_COST_CHANGE), ctx=ctx,
            signals=SignalSet(), risk=_risk(RiskBand.HIGH, 0.8),
            evidence_sufficient=True, evidence_conflict=False,
            as_of=as_of, is_proactive=True,
        )
        assert result.verdict == GateVerdict.SUPPRESS
        assert ReasonCode.SUPPRESSION_WINDOW_ACTIVE in result.reasons

    def test_gate_can_only_restrict(self, store):
        """No input combination may turn a restrictive verdict into ALLOW.

        The gate's safety argument depends on it having no upgrade path, so the
        property is asserted rather than assumed.
        """
        gate = GovernanceGate()
        restrictive_inputs = [
            dict(signals=SignalSet(clinical_question_detected=True)),
            dict(signals=SignalSet(sentiment=Sentiment.DISTRESSED)),
            dict(evidence_sufficient=False),
            dict(evidence_conflict=True),
            dict(decision=_decision(confidence=0.1)),
        ]
        for override in restrictive_inputs:
            kwargs = dict(
                decision=_decision(confidence=0.99), ctx=_ctx(store),
                signals=SignalSet(), risk=_risk(RiskBand.HIGH, 0.95),
                evidence_sufficient=True, evidence_conflict=False,
                as_of=datetime(2026, 3, 1), is_proactive=False,
            )
            kwargs.update(override)
            assert gate.evaluate(**kwargs).verdict != GateVerdict.ALLOW


# ===========================================================================
# 4. Grounding
# ===========================================================================


class TestGrounding:
    def test_template_numerals_appear_in_the_policy_corpus(self, store):
        """Static copy may not state a number the corpus does not support.

        The deterministic draft is treated as grounded by construction, so its
        template constants have to actually be true of the corpus. Without this
        test that assumption is unverified.
        """
        from careweave.graph.generate import _CLOSERS, _OPENERS

        corpus = " ".join(d.body for d in store.policy_documents)
        corpus_numbers = set(re.findall(r"\d+", corpus))
        for action, text in {**_OPENERS, **_CLOSERS}.items():
            for number in re.findall(r"\d+", text):
                assert number in corpus_numbers, (
                    f"template for {action} states '{number}', which appears "
                    "nowhere in the policy corpus"
                )

    def test_ungrounded_number_is_rejected(self, store):
        from careweave.domain.models import ContextFrame
        from careweave.graph.generate import GenerationInputs, verify_grounding

        frame = ContextFrame(
            what_is_happening="A review has been open for 12 days.",
            why_it_matters="", relevant_evidence_ids=[],
            what_could_happen_next="", candidate_next_steps=[],
            grounded_facts=["A review has been open for 12 days. [authorization:PA-1]"],
        )
        inputs = GenerationInputs(
            action=ActionType.EXPLAIN_AUTH_STATUS, frame=frame,
            ctx=_ctx(store), signals=SignalSet(), evidence=[],
            decision=_decision(),
        )
        ok, problems = verify_grounding("Your review has been open for 47 days.", inputs)
        assert not ok
        assert any("47" in p for p in problems)

        ok, _ = verify_grounding("Your review has been open for 12 days.", inputs)
        assert ok

    def test_denylisted_attributes_never_surface(self, store):
        from careweave.domain.models import ContextFrame
        from careweave.graph.generate import GenerationInputs, verify_grounding

        frame = ContextFrame(
            what_is_happening="", why_it_matters="", relevant_evidence_ids=[],
            what_could_happen_next="", candidate_next_steps=[], grounded_facts=[],
        )
        inputs = GenerationInputs(
            action=ActionType.PROVIDE_NEXT_STEP, frame=frame, ctx=_ctx(store),
            signals=SignalSet(), evidence=[], decision=_decision(),
        )
        ok, problems = verify_grounding(
            "Because your risk score is elevated, we are reaching out.", inputs
        )
        assert not ok
        assert any("risk_score" in p for p in problems)


# ===========================================================================
# 5. Extraction safety
# ===========================================================================


class TestExtractionSafety:
    @pytest.mark.parametrize(
        "text",
        [
            "Is it normal to feel dizzy after taking this?",
            "Should I stop taking my medication?",
            "Are there side effects I should know about?",
            "Can I double my dose if I miss one?",
            "Is this safe to take with my other prescription?",
        ],
    )
    def test_clinical_questions_are_detected(self, text):
        from careweave.nlp.extract import ExtractionInput, deterministic_extract

        signals = deterministic_extract(ExtractionInput(text, []), set())
        assert signals.clinical_question_detected, f"missed: {text}"

    @pytest.mark.parametrize(
        "text",
        [
            "I've already missed two doses because of this.",
            "The pharmacy said my claim was rejected.",
            "Why did my copay go up this month?",
        ],
    )
    def test_non_clinical_text_is_not_flagged(self, text):
        from careweave.nlp.extract import ExtractionInput, deterministic_extract

        signals = deterministic_extract(ExtractionInput(text, []), set())
        assert not signals.clinical_question_detected, f"false positive: {text}"

    def test_llm_cannot_downgrade_clinical_detection(self):
        """The merge is an OR on clinical detection, never an AND."""
        from careweave.nlp.extract import SignalExtractor

        base = SignalSet(clinical_question_detected=True, confidence=0.8)
        merged = SignalExtractor._merge(base, {"clinical_question_detected": False})
        assert merged.clinical_question_detected

    def test_llm_cannot_downgrade_distress(self):
        from careweave.nlp.extract import SignalExtractor

        base = SignalSet(sentiment=Sentiment.DISTRESSED)
        merged = SignalExtractor._merge(base, {"sentiment": "neutral"})
        assert merged.sentiment == Sentiment.DISTRESSED

    def test_malformed_llm_output_falls_back(self):
        from careweave.nlp.extract import SignalExtractor

        base = SignalSet(intent=MemberIntent.IS_THIS_COVERED, confidence=0.7)
        merged = SignalExtractor._merge(base, {"intent": "not_a_real_intent"})
        assert merged.intent == MemberIntent.IS_THIS_COVERED


# ===========================================================================
# 6. Retrieval scoping
# ===========================================================================


class TestRetrievalScoping:
    def test_benefit_summaries_are_plan_scoped(self, store):
        """Retrieval must not cite another plan's numbers.

        Benefit summaries state different deductibles per plan, so an unscoped
        result can ground an answer in the wrong figures -- fluent, specific and
        wrong.
        """
        from careweave.rag.retriever import load_retriever

        retriever = load_retriever(store)
        results = retriever.retrieve(
            "what is my deductible and generic copay", top_k=8, plan_id="PLN-VALUE"
        )
        wrong = [
            e for e in results
            if e.document_id.startswith("BEN-") and not e.document_id.endswith("PLN-VALUE")
        ]
        assert not wrong, f"leaked other plans' benefit summaries: {wrong}"

    def test_member_documents_are_not_in_the_policy_index(self, store):
        from careweave.rag.retriever import load_retriever

        retriever = load_retriever(store)
        member_doc_ids = set(store.member_documents)
        indexed = {c.document_id for c in retriever.chunks}
        assert not (indexed & member_doc_ids)


# ===========================================================================
# 7. Ledger
# ===========================================================================


class TestLedger:
    def test_every_terminal_path_writes_a_record(self, tmp_path, store):
        """Including the deliberate no-action path."""
        from careweave.governance.ledger import FrictionLedger
        from careweave.graph.build import CareWeaveEngine

        ledger = FrictionLedger(tmp_path / "ledger.jsonl")
        engine = CareWeaveEngine(store=store, ledger=ledger)

        before = len(ledger.records)
        for i, mid in enumerate(store.member_ids[:6]):
            engine.run(
                member_id=mid, as_of=datetime(2026, 3, 1),
                trigger=EventType.REFILL_DUE, case_id=f"T-{i}",
            )
        assert len(ledger.records) - before == 6

    def test_suppressed_decisions_are_audited(self, tmp_path, store):
        from careweave.governance.ledger import FrictionLedger
        from careweave.graph.build import CareWeaveEngine

        ledger = FrictionLedger(tmp_path / "l.jsonl")
        engine = CareWeaveEngine(store=store, ledger=ledger)
        for i, mid in enumerate(store.member_ids[:25]):
            engine.run(
                member_id=mid, as_of=datetime(2026, 2, 1),
                trigger=EventType.REFILL_DUE, is_proactive=True, case_id=f"S-{i}",
            )
        verdicts = {r.gate_verdict for r in ledger.records}
        assert GateVerdict.SUPPRESS in verdicts
        for record in ledger.records:
            assert record.gate_verdict is not None
            assert record.model_versions
