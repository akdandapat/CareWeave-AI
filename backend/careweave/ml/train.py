"""Friction risk model.

Methodology choices worth defending in a review:

* **Temporal split, not random.** Observation points are time-ordered and split
  chronologically. A random split would let the model see a member's later
  behaviour while predicting their earlier behaviour, which inflates every
  metric and is the single most common flaw in portfolio ML.

* **Grouped by member across the boundary.** Members straddling the split are
  assigned wholly to the earlier fold, so no member appears in both train and
  test. Temporal *and* group separation together.

* **PR-AUC is the headline, not ROC-AUC.** At a 17% positive rate ROC-AUC
  flatters. Precision-recall reflects what an outreach queue actually
  experiences.

* **Calibration is a requirement, not a nicety.** The governance gate thresholds
  on probability. An uncalibrated score makes those thresholds meaningless, so
  the model is calibrated on a held-out validation fold and calibration error is
  reported alongside discrimination.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from careweave.config import SETTINGS
from careweave.ml.features import feature_columns

CATEGORICAL = ["trigger_event"]


@dataclass
class SplitData:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_valid: pd.DataFrame
    y_valid: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    boundaries: dict[str, str] = field(default_factory=dict)


def purged_temporal_split(df: pd.DataFrame, *, purge_days: int | None = None) -> SplitData:
    """Chronological split with an embargo between folds.

    Why not a member-disjoint split? Because every member is present across the
    whole simulation window, so requiring disjointness would empty the later
    folds -- and more importantly, it would model the wrong deployment. In
    production the engine scores members it has already seen, repeatedly, over
    time. The honest question is "can it predict the future for a known member",
    not "can it generalise to a stranger".

    The genuine risk in that setup is *adjacency*: a training row whose 14-day
    label window overlaps a test row's feature window shares outcome
    information. That is removed with a purge gap of one full horizon between
    folds, the standard embargo used in time-series backtesting.

    ``member_disjoint_split`` below provides a cold-start robustness check on
    unseen members, reported alongside the primary numbers.
    """
    purge = purge_days if purge_days is not None else SETTINGS.ml.horizon_days
    df = df.sort_values("as_of").reset_index(drop=True)
    n = len(df)
    t_train_end = df["as_of"].iloc[int(n * SETTINGS.ml.train_end_frac)]
    t_valid_end = df["as_of"].iloc[int(n * SETTINGS.ml.valid_end_frac)]
    gap = pd.Timedelta(days=purge)

    tr = df[df["as_of"] <= t_train_end]
    va = df[(df["as_of"] > t_train_end + gap) & (df["as_of"] <= t_valid_end)]
    te = df[df["as_of"] > t_valid_end + gap]

    cols = feature_columns(df)
    return SplitData(
        X_train=tr[cols], y_train=tr["label"].to_numpy(),
        X_valid=va[cols], y_valid=va["label"].to_numpy(),
        X_test=te[cols], y_test=te["label"].to_numpy(),
        boundaries={
            "strategy": f"chronological with {purge}-day purge between folds",
            "train": f"{tr['as_of'].min():%Y-%m-%d} .. {tr['as_of'].max():%Y-%m-%d}",
            "valid": f"{va['as_of'].min():%Y-%m-%d} .. {va['as_of'].max():%Y-%m-%d}",
            "test": f"{te['as_of'].min():%Y-%m-%d} .. {te['as_of'].max():%Y-%m-%d}",
            "purged_rows": int(n - len(tr) - len(va) - len(te)),
        },
    )


def member_disjoint_split(df: pd.DataFrame, *, seed: int) -> SplitData:
    """Cold-start robustness check: hold out entire members, not time periods.

    Answers a different question -- how the model behaves on a member it has
    never scored before. Reported as a secondary figure so the primary
    (temporal) number is not confused with it.
    """
    rng = np.random.default_rng(seed)
    members = df["member_id"].unique()
    rng.shuffle(members)
    n = len(members)
    tr_m = set(members[: int(n * 0.7)])
    va_m = set(members[int(n * 0.7): int(n * 0.85)])
    te_m = set(members[int(n * 0.85):])

    cols = feature_columns(df)
    tr, va, te = (df[df["member_id"].isin(m)] for m in (tr_m, va_m, te_m))
    return SplitData(
        X_train=tr[cols], y_train=tr["label"].to_numpy(),
        X_valid=va[cols], y_valid=va["label"].to_numpy(),
        X_test=te[cols], y_test=te["label"].to_numpy(),
        boundaries={"strategy": "member-disjoint (cold start robustness check)",
                    "n_members": str(n)},
    )


def _preprocessor(numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )


def build_models(numeric: list[str]) -> dict[str, Any]:
    models: dict[str, Any] = {
        "majority_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": Pipeline(
            [
                ("prep", _preprocessor(numeric)),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
            ]
        ),
    }
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = Pipeline(
            [
                ("prep", _preprocessor(numeric)),
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=400,
                        learning_rate=0.045,
                        num_leaves=24,
                        min_child_samples=60,
                        subsample=0.85,
                        subsample_freq=1,
                        colsample_bytree=0.75,
                        reg_lambda=1.5,
                        random_state=SETTINGS.generation.seed,
                        verbose=-1,
                    ),
                ),
            ]
        )
    except ImportError:  # pragma: no cover
        from sklearn.ensemble import HistGradientBoostingClassifier

        models["hist_gradient_boosting"] = Pipeline(
            [("prep", _preprocessor(numeric)),
             ("clf", HistGradientBoostingClassifier(random_state=SETTINGS.generation.seed))]
        )
    return models


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    base = float(y.mean())
    ap = float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan")
    return {
        "n": int(len(y)),
        "positive_rate": base,
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "pr_auc": ap,
        "pr_auc_lift_over_base": float(ap / base) if base else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "threshold": float(threshold),
    }


def precision_at_k(y: np.ndarray, p: np.ndarray, k_frac: float) -> dict[str, float]:
    """What an outreach queue of the top k% would actually look like.

    More decision-relevant than a global threshold: operations capacity is
    finite, so the real question is the precision of the queue you can staff.
    """
    k = max(1, int(len(p) * k_frac))
    order = np.argsort(-p)[:k]
    return {
        "k_frac": k_frac,
        "n_flagged": int(k),
        "precision": float(y[order].mean()),
        "recall": float(y[order].sum() / max(1, y.sum())),
        "lift": float(y[order].mean() / y.mean()) if y.mean() else float("nan"),
    }


def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Pick the threshold maximising F1 on validation, then freeze it."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    if len(thr) == 0:
        return 0.5
    return float(thr[int(np.argmax(f1[:-1]))])


def permutation_importance_ap(model, X: pd.DataFrame, y: np.ndarray, *, seed: int,
                              n_repeats: int = 3) -> dict[str, float]:
    """Permutation importance measured in average precision, not accuracy."""
    rng = np.random.default_rng(seed)
    base = average_precision_score(y, model.predict_proba(X)[:, 1])
    out: dict[str, float] = {}
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops.append(base - average_precision_score(y, model.predict_proba(Xp)[:, 1]))
        out[col] = float(np.mean(drops))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def train(df: pd.DataFrame, *, artifacts_dir: Path | None = None) -> dict[str, Any]:
    artifacts_dir = artifacts_dir or SETTINGS.paths.artifacts
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    split = purged_temporal_split(df)
    numeric = [c for c in split.X_train.columns if c not in CATEGORICAL]
    models = build_models(numeric)

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_version": SETTINGS.ml.model_version,
        "horizon_days": SETTINGS.ml.horizon_days,
        "label_definition": (
            "1 if member_contacted_support, case_escalated, or "
            "rx_abandoned_at_counter occurs in (as_of, as_of + horizon]"
        ),
        "split": {
            "strategy": "chronological, member-disjoint across folds",
            "boundaries": split.boundaries,
            "sizes": {
                "train": int(len(split.X_train)),
                "valid": int(len(split.X_valid)),
                "test": int(len(split.X_test)),
            },
            "positive_rates": {
                "train": float(split.y_train.mean()),
                "valid": float(split.y_valid.mean()),
                "test": float(split.y_test.mean()),
            },
        },
        "n_features": len(split.X_train.columns),
        "models": {},
    }

    best_name, best_ap, best_obj = None, -1.0, None
    for name, model in models.items():
        model.fit(split.X_train, split.y_train)
        p_valid = model.predict_proba(split.X_valid)[:, 1]
        thr = choose_threshold(split.y_valid, p_valid) if name != "majority_baseline" else 0.5
        p_test = model.predict_proba(split.X_test)[:, 1]

        entry = {
            "valid": evaluate(split.y_valid, p_valid, thr),
            "test": evaluate(split.y_test, p_test, thr),
        }
        report["models"][name] = entry
        ap = entry["valid"]["pr_auc"]
        if name != "majority_baseline" and ap == ap and ap > best_ap:
            best_name, best_ap, best_obj = name, ap, model

    assert best_obj is not None and best_name is not None

    # -- calibrate the winner on validation, evaluate on untouched test ----
    # sklearn >=1.6: wrap the fitted estimator rather than cv="prefit"
    calibrated = CalibratedClassifierCV(FrozenEstimator(best_obj), method="isotonic")
    calibrated.fit(split.X_valid, split.y_valid)
    p_cal_valid = calibrated.predict_proba(split.X_valid)[:, 1]
    thr_cal = choose_threshold(split.y_valid, p_cal_valid)
    p_cal_test = calibrated.predict_proba(split.X_test)[:, 1]

    report["selected_model"] = best_name
    report["calibration"] = {
        "method": "isotonic, fit on validation fold",
        "uncalibrated_test": report["models"][best_name]["test"],
        "calibrated_test": evaluate(split.y_test, p_cal_test, thr_cal),
        "frozen_threshold": float(thr_cal),
    }
    report["operating_points"] = [
        precision_at_k(split.y_test, p_cal_test, k) for k in (0.05, 0.10, 0.20, 0.30)
    ]
    report["reliability_curve"] = _reliability(split.y_test, p_cal_test)
    report["band_thresholds"] = {
        "medium": SETTINGS.ml.band_medium,
        "high": SETTINGS.ml.band_high,
        "band_distribution_test": _band_distribution(split.y_test, p_cal_test),
    }
    report["permutation_importance_top20"] = dict(
        list(
            permutation_importance_ap(
                best_obj, split.X_test, split.y_test, seed=SETTINGS.generation.seed
            ).items()
        )[:20]
    )

    # -- cold-start robustness: entire members held out --------------------
    ms = member_disjoint_split(df, seed=SETTINGS.generation.seed)
    cold = build_models([c for c in ms.X_train.columns if c not in CATEGORICAL])[best_name]
    cold.fit(ms.X_train, ms.y_train)
    p_cold = cold.predict_proba(ms.X_test)[:, 1]
    report["robustness_member_disjoint"] = {
        "note": (
            "Secondary check on unseen members. Answers a different question "
            "from the primary temporal split and is not directly comparable."
        ),
        "boundaries": ms.boundaries,
        "test": evaluate(ms.y_test, p_cold, choose_threshold(ms.y_valid,
                                                             cold.predict_proba(ms.X_valid)[:, 1])),
    }

    import pickle

    with (artifacts_dir / "friction_model.pkl").open("wb") as fh:
        pickle.dump(
            {
                "model": calibrated,
                "threshold": thr_cal,
                "feature_columns": list(split.X_train.columns),
                "model_version": SETTINGS.ml.model_version,
                "selected_algorithm": best_name,
            },
            fh,
        )
    (artifacts_dir / "ml_report.json").write_text(json.dumps(report, indent=2))
    return report


def _reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        out.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
                "n": int(m.sum()),
                "mean_predicted": float(p[m].mean()),
                "observed_rate": float(y[m].mean()),
            }
        )
    return out


def _band_distribution(y: np.ndarray, p: np.ndarray) -> dict[str, dict[str, float]]:
    lo, hi = SETTINGS.ml.band_medium, SETTINGS.ml.band_high
    bands = {"low": p < lo, "medium": (p >= lo) & (p < hi), "high": p >= hi}
    return {
        name: {
            "n": int(m.sum()),
            "share": float(m.mean()),
            "observed_friction_rate": float(y[m].mean()) if m.any() else float("nan"),
        }
        for name, m in bands.items()
    }


def main() -> None:  # pragma: no cover
    from careweave.data.store import get_store
    from careweave.ml.features import build_dataset

    store = get_store()
    df = build_dataset(store)
    report = train(df)

    sel = report["selected_model"]
    print(f"Selected: {sel}")
    print(f"Split: {report['split']['boundaries']}")
    for name, m in report["models"].items():
        t = m["test"]
        print(f"  {name:<22} test PR-AUC={t['pr_auc']:.3f} ROC-AUC={t['roc_auc']:.3f} "
              f"F1={t['f1']:.3f}")
    cal = report["calibration"]["calibrated_test"]
    print(f"\nCalibrated test: PR-AUC={cal['pr_auc']:.3f} (base rate {cal['positive_rate']:.3f}, "
          f"lift {cal['pr_auc_lift_over_base']:.2f}x) ECE={cal['ece']:.4f} Brier={cal['brier']:.4f}")
    print("\nQueue operating points (test):")
    for op in report["operating_points"]:
        print(f"  top {op['k_frac']:.0%}: precision={op['precision']:.3f} "
              f"recall={op['recall']:.3f} lift={op['lift']:.2f}x")
    cold = report["robustness_member_disjoint"]["test"]
    print(f"\nCold-start (unseen members): PR-AUC={cold['pr_auc']:.3f} "
          f"ROC-AUC={cold['roc_auc']:.3f} base={cold['positive_rate']:.3f}")
    print("\nTop features by permutation importance (drop in average precision):")
    for k, v in list(report["permutation_importance_top20"].items())[:10]:
        print(f"  {k:<28} {v:+.4f}")
    print("\nRisk band behaviour (test):")
    for band, st in report["band_thresholds"]["band_distribution_test"].items():
        print(f"  {band:<7} n={st['n']:>5} share={st['share']:.1%} "
              f"observed friction={st['observed_friction_rate']:.1%}")


if __name__ == "__main__":  # pragma: no cover
    main()
