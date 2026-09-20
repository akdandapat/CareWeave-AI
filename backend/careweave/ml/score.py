"""Runtime scoring.

Wraps the trained artefact so the graph never touches sklearn directly. Three
responsibilities beyond calling ``predict_proba``:

* **Band assignment** from calibrated probability, using the frozen thresholds.
  The gate keys on bands, so this is where a probability becomes a decision.
* **Reason codes**, derived from the feature vector rather than from the model.
  A SHAP value explains the model; a reason code explains the *situation*, which
  is what an advocate and an auditor actually need.
* **Per-feature contributions** for the advocate UI, so a human can see which
  parts of the record drove the score and disagree with it.

If the artefact is missing the scorer degrades to a transparent rule-based
fallback rather than raising. An engine that stops working because a pickle is
absent is worse than one that says so and keeps going with a weaker signal.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from careweave.config import SETTINGS
from careweave.data.store import JourneyStore
from careweave.domain.enums import ReasonCode, RiskBand
from careweave.domain.models import RiskAssessment
from careweave.ml.features import ObservationPoint, build_features


def _band(p: float) -> RiskBand:
    if p >= SETTINGS.ml.band_high:
        return RiskBand.HIGH
    if p >= SETTINGS.ml.band_medium:
        return RiskBand.MEDIUM
    return RiskBand.LOW


#: (feature, comparison, threshold, reason). Evaluated against the point-in-time
#: feature row, independent of the model, so reasons stay stable if the model
#: is retrained.
_REASON_RULES: list[tuple[str, str, float, ReasonCode]] = [
    ("max_pa_pending_days", ">", 7, ReasonCode.PA_PENDING_PAST_SLA),
    ("any_pa_pending_info", ">", 0, ReasonCode.PA_PENDING_INFO),
    ("refill_runway_days", "<", 7, ReasonCode.REFILL_RUNWAY_SHORT),
    ("days_since_last_rejection", "<", 30, ReasonCode.RECENT_CLAIM_REJECTION),
    ("n_contacts_30d", ">=", 2, ReasonCode.REPEAT_CONTACT),
    ("cost_delta", ">", 15, ReasonCode.COST_DELTA_LARGE),
    ("in_deductible_phase", ">", 0, ReasonCode.DEDUCTIBLE_PHASE_TRANSITION),
    ("n_specialty_rx", ">", 0, ReasonCode.SPECIALTY_DRUG),
    ("n_abandonments_180d", ">", 0, ReasonCode.PRIOR_ABANDONMENT),
    ("reject_pharmacy_out_of_network_90d", ">", 0, ReasonCode.OUT_OF_NETWORK_PHARMACY),
]

_OPS = {
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
}


def derive_reasons(features: dict[str, Any]) -> list[ReasonCode]:
    out: list[ReasonCode] = []
    for key, op, threshold, reason in _REASON_RULES:
        value = features.get(key)
        if value is None:
            continue
        try:
            if _OPS[op](float(value), threshold):
                out.append(reason)
        except (TypeError, ValueError):
            continue
    return out


class FrictionScorer:
    def __init__(self, store: JourneyStore, artifact_path: Path | None = None) -> None:
        self.store = store
        path = artifact_path or (SETTINGS.paths.artifacts / "friction_model.pkl")
        self.available = path.exists()
        self.model = None
        self.feature_columns: list[str] = []
        self.model_version = SETTINGS.ml.model_version
        self.algorithm = "unavailable"
        if self.available:
            with path.open("rb") as fh:
                bundle = pickle.load(fh)
            self.model = bundle["model"]
            self.feature_columns = bundle["feature_columns"]
            self.model_version = bundle["model_version"]
            self.algorithm = bundle["selected_algorithm"]

    # -- scoring ---------------------------------------------------------

    def score_point(self, point: ObservationPoint) -> tuple[RiskAssessment, dict[str, Any]]:
        features = build_features(self.store, point)
        reasons = derive_reasons(features)

        if not self.available or self.model is None:
            score = self._fallback_score(reasons)
            return (
                RiskAssessment(
                    score=score, band=_band(score), reasons=reasons,
                    model_version=f"{self.model_version}(fallback)",
                    contributions={}, calibrated=False,
                ),
                features,
            )

        row = pd.DataFrame([{c: features.get(c) for c in self.feature_columns}])
        score = float(self.model.predict_proba(row)[0, 1])
        return (
            RiskAssessment(
                score=round(score, 4),
                band=_band(score),
                reasons=reasons,
                model_version=self.model_version,
                contributions=self._contributions(row, score),
                calibrated=True,
            ),
            features,
        )

    def score(
        self, member_id: str, as_of: datetime, trigger
    ) -> tuple[RiskAssessment, dict[str, Any]]:
        return self.score_point(ObservationPoint(member_id, as_of, trigger, None))

    # -- explanation ------------------------------------------------------

    def _contributions(self, row: pd.DataFrame, score: float, top_n: int = 6) -> dict[str, float]:
        """Leave-one-out contributions against a population-median baseline.

        Model-agnostic and cheap. Each value answers "how much did this feature's
        actual value move the score, relative to a typical member". Not SHAP, and
        the docs say so -- but it is faithful to what the model does with this
        row, which is what an advocate needs to sanity-check a recommendation.
        """
        if self.model is None:
            return {}
        baseline = self._median_row()
        if baseline is None:
            return {}

        out: dict[str, float] = {}
        for col in self.feature_columns:
            if col == "trigger_event":
                continue
            probe = row.copy()
            probe.loc[:, col] = baseline.get(col, probe[col].iloc[0])
            try:
                without = float(self.model.predict_proba(probe)[0, 1])
            except Exception:  # noqa: BLE001
                continue
            delta = score - without
            if abs(delta) > 1e-4:
                out[col] = round(delta, 4)
        ranked = sorted(out.items(), key=lambda kv: -abs(kv[1]))[:top_n]
        return dict(ranked)

    _median_cache: dict[str, Any] | None = None

    def _median_row(self) -> dict[str, Any] | None:
        if FrictionScorer._median_cache is not None:
            return FrictionScorer._median_cache
        path = SETTINGS.paths.artifacts / "features.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        numeric = df[[c for c in self.feature_columns if c in df.columns
                      and pd.api.types.is_numeric_dtype(df[c])]]
        FrictionScorer._median_cache = numeric.median().to_dict()
        return FrictionScorer._median_cache

    @staticmethod
    def _fallback_score(reasons: list[ReasonCode]) -> float:
        """Transparent additive fallback used only when the artefact is missing."""
        weights = {
            ReasonCode.PA_PENDING_INFO: 0.18,
            ReasonCode.PA_PENDING_PAST_SLA: 0.12,
            ReasonCode.REFILL_RUNWAY_SHORT: 0.15,
            ReasonCode.RECENT_CLAIM_REJECTION: 0.14,
            ReasonCode.REPEAT_CONTACT: 0.16,
            ReasonCode.COST_DELTA_LARGE: 0.10,
            ReasonCode.PRIOR_ABANDONMENT: 0.08,
            ReasonCode.SPECIALTY_DRUG: 0.05,
        }
        return float(min(0.95, 0.06 + sum(weights.get(r, 0.0) for r in reasons)))


_SCORER: FrictionScorer | None = None


def get_scorer(store: JourneyStore | None = None) -> FrictionScorer:
    global _SCORER
    if _SCORER is None:
        from careweave.data.store import get_store

        _SCORER = FrictionScorer(store or get_store())
    return _SCORER
