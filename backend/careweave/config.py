"""Configuration. Environment-driven, with sane local defaults.

The default profile runs entirely offline with no external API keys, which
keeps the demo reproducible for anyone who clones the repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class Paths:
    root: Path = REPO_ROOT
    data: Path = REPO_ROOT / "data"
    synthetic: Path = REPO_ROOT / "data" / "synthetic"
    documents: Path = REPO_ROOT / "data" / "documents"
    artifacts: Path = REPO_ROOT / "data" / "artifacts"
    docs: Path = REPO_ROOT / "docs"

    def ensure(self) -> None:
        for p in (self.data, self.synthetic, self.documents, self.artifacts, self.docs):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = _env_int("CW_SEED", 20260911)
    n_members: int = _env_int("CW_N_MEMBERS", 2000)
    #: simulation window
    start_date: str = os.environ.get("CW_START_DATE", "2025-01-01")
    end_date: str = os.environ.get("CW_END_DATE", "2026-06-30")


@dataclass(frozen=True)
class MLConfig:
    #: prediction horizon in days for the friction label
    horizon_days: int = _env_int("CW_HORIZON_DAYS", 14)
    #: temporal split boundaries (fractions of the simulation window)
    train_end_frac: float = _env_float("CW_TRAIN_END", 0.70)
    valid_end_frac: float = _env_float("CW_VALID_END", 0.85)
    #: probability thresholds after calibration
    band_medium: float = _env_float("CW_BAND_MEDIUM", 0.25)
    band_high: float = _env_float("CW_BAND_HIGH", 0.50)
    model_version: str = os.environ.get("CW_MODEL_VERSION", "friction-risk-v1")


@dataclass(frozen=True)
class GovernanceConfig:
    #: minimum action confidence to act without a human
    min_action_confidence: float = _env_float("CW_MIN_ACTION_CONF", 0.55)
    #: minimum extractor confidence for signals to be trusted
    min_signal_confidence: float = _env_float("CW_MIN_SIGNAL_CONF", 0.40)
    #: no two proactive outreaches on the same topic within this many days
    topic_suppression_days: int = _env_int("CW_TOPIC_SUPPRESSION_DAYS", 14)
    #: max proactive outreaches per member per rolling 30 days
    frequency_cap_30d: int = _env_int("CW_FREQ_CAP_30D", 3)
    gate_version: str = os.environ.get("CW_GATE_VERSION", "gate-v1")


@dataclass(frozen=True)
class RAGConfig:
    chunk_tokens: int = _env_int("CW_CHUNK_TOKENS", 180)
    chunk_overlap: int = _env_int("CW_CHUNK_OVERLAP", 40)
    top_k: int = _env_int("CW_TOP_K", 5)
    #: lexical/dense blend weight; 1.0 == pure BM25
    lexical_weight: float = _env_float("CW_LEXICAL_WEIGHT", 0.65)
    #: retrieval score floor below which evidence is deemed insufficient
    min_evidence_score: float = _env_float("CW_MIN_EVIDENCE_SCORE", 0.12)
    index_version: str = os.environ.get("CW_INDEX_VERSION", "rag-index-v1")


@dataclass(frozen=True)
class LLMConfig:
    #: "stub" runs deterministic offline behaviour; "anthropic" calls the API
    provider: str = os.environ.get("CW_LLM_PROVIDER", "stub")
    model: str = os.environ.get("CW_LLM_MODEL", "claude-sonnet-4-6")
    max_tokens: int = _env_int("CW_LLM_MAX_TOKENS", 1200)
    temperature: float = _env_float("CW_LLM_TEMPERATURE", 0.2)

    @property
    def version_tag(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


SETTINGS = Settings()
