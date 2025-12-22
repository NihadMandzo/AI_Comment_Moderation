from __future__ import annotations

import threading
from typing import Dict, List, Tuple

from moderation_agent.ml.baseline import LABELS, baseline_model
from moderation_agent.ml.online import OnlineLearner
from moderation_agent.text.heuristics import (
    nonviolent_threat_score,
    profanity_score,
    spam_score,
    threat_score,
    violent_threat_score,
)
from moderation_agent.text.normalize import normalize_text, two_channel

from .classifier import ITextClassifier


class ModerationClassifier(ITextClassifier):
    """Wraps the existing baseline + online learner into a single black box."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._online: OnlineLearner | None = None

    @property
    def labels(self) -> List[str]:
        return list(LABELS)

    def loads_active_model(self, blob: bytes) -> None:
        with self._lock:
            self._online = OnlineLearner.loads(blob)

    def dumps_active_model(self) -> bytes:
        with self._lock:
            assert self._online is not None
            return self._online.dumps()

    def _ensure_online(self) -> OnlineLearner:
        with self._lock:
            if self._online is None:
                self._online = OnlineLearner.fresh()
            return self._online

    def predict_scores(self, text_raw: str, text_norm: str) -> Dict[str, float]:
        raw = (text_raw or "").strip()
        norm = (text_norm or "").strip()
        if not raw and not norm:
            return {k: 0.0 for k in LABELS}

        ch = two_channel(raw)
        baseline = baseline_model.predict(ch.raw)

        online = self._ensure_online()
        with self._lock:
            learned = online.predict_proba(norm or ch.norm)

        # Return both signals; application combines + applies overrides.
        out = {f"baseline:{k}": float(baseline.get(k, 0.0)) for k in LABELS}
        out.update({f"learned:{k}": float(learned.get(k, 0.0)) for k in LABELS})
        return out

    def update_online(self, text_norm: str, selected_labels: list[str], *, kind: str, weight: float) -> None:
        online = self._ensure_online()
        with self._lock:
            online.update_weighted(normalize_text(text_norm), selected_labels, kind=kind, weight=float(weight))

    def rebuild_from_training_rows(self, rows: List[Tuple[str, List[str], float, str]]) -> int:
        learner = OnlineLearner.fresh()
        used = 0
        for kind, labels, weight, text_norm in rows:
            learner.update_weighted(normalize_text(text_norm), labels, kind=kind, weight=float(weight))
            used += 1
        with self._lock:
            self._online = learner
        return used


def rule_overrides(text_norm: str) -> tuple[float, Dict[str, float]]:
    """Shared rule-based overrides (still shared, not web)."""

    vth = float(violent_threat_score(text_norm))
    nth = float(nonviolent_threat_score(text_norm))
    overrides = {
        "profanity": float(profanity_score(text_norm)),
        "threat": float(max(vth, nth, float(threat_score(text_norm)))),
        "spam": float(spam_score(text_norm)),
    }
    return vth, overrides
