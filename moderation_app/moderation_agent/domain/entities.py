from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .enums import Decision, MessageStatus, ReviewKind


@dataclass(frozen=True)
class Message:
    id: int
    text: str
    created_at: str
    status: MessageStatus
    report_count: int = 0


@dataclass(frozen=True)
class Prediction:
    message_id: int
    baseline_scores: Dict[str, float]
    learned_scores: Dict[str, float]
    final_scores: Dict[str, float]
    top_label: str
    top_score: float
    decision: Decision
    new_status: MessageStatus
    model_version_id: Optional[int] = None


@dataclass(frozen=True)
class Review:
    id: int
    message_id: int
    created_at: str
    kind: ReviewKind
    labels: List[str]
    reporter_hash: str
    confidence: float
    used_for_training: bool
    note: str


@dataclass(frozen=True)
class ModelVersion:
    id: int
    created_at: str
    trained_on_reports: int
    is_active: bool


@dataclass(frozen=True)
class SystemSettings:
    """Small mutable 'world state' the retrain agent senses."""

    enabled: bool = True
    retrain_gold_threshold: int = 50
    new_gold_since_last_train: int = 0

    # Threshold rules (kept in shared, not web)
    allow_max: float = 0.35
    pending_min_default: float = 0.35
    delete_min_default: float = 0.80

    pending_min_by_label: Dict[str, float] = field(default_factory=dict)
    delete_min_by_label: Dict[str, float] = field(default_factory=dict)

    violent_threat_delete_min: float = 0.80
