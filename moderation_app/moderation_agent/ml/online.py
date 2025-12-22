from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


LABELS: List[str] = ["hate_speech", "abuse", "threat", "profanity", "spam"]


def _new_vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        n_features=2**18,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        analyzer="char_wb",
        ngram_range=(3, 5),
    )


def _new_classifier() -> SGDClassifier:
    return SGDClassifier(loss="log_loss", alpha=1e-5, random_state=42)


@dataclass
class OnlineLearner:
    vectorizer: HashingVectorizer
    classifiers: Dict[str, SGDClassifier]
    is_initialized: bool

    @classmethod
    def fresh(cls) -> "OnlineLearner":
        vectorizer = _new_vectorizer()
        classifiers = {label: _new_classifier() for label in LABELS}
        return cls(vectorizer=vectorizer, classifiers=classifiers, is_initialized=False)

    def predict_proba(self, text: str) -> Dict[str, float]:
        x = self.vectorizer.transform([text])
        out: Dict[str, float] = {}
        for label, clf in self.classifiers.items():
            if not self.is_initialized:
                out[label] = 0.0
                continue
            proba = clf.predict_proba(x)[0]
            out[label] = float(proba[1])
        return out

    def update(self, text: str, selected_labels: Iterable[str], kind: str = "BAD") -> None:
        self.update_weighted(text, selected_labels, kind=kind, weight=1.0)

    def update_weighted(
        self,
        text: str,
        selected_labels: Iterable[str],
        *,
        kind: str = "BAD",
        weight: float = 1.0,
    ) -> None:
        kind_norm = (kind or "BAD").upper()
        selected = set(selected_labels)
        x = self.vectorizer.transform([text])
        w = float(max(0.0, weight))
        sample_weight = np.array([w], dtype=float)

        if kind_norm == "GOOD":
            y = {label: 0 for label in LABELS}
        else:
            y = {label: int(label in selected) for label in LABELS}

        if not self.is_initialized:
            for label, clf in self.classifiers.items():
                clf.partial_fit(
                    x,
                    np.array([y[label]]),
                    classes=np.array([0, 1]),
                    sample_weight=sample_weight,
                )
            self.is_initialized = True
            return

        for label, clf in self.classifiers.items():
            clf.partial_fit(x, np.array([y[label]]), sample_weight=sample_weight)

    def dumps(self) -> bytes:
        payload = {
            "vectorizer": self.vectorizer,
            "classifiers": self.classifiers,
            "is_initialized": self.is_initialized,
        }
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def loads(cls, blob: bytes) -> "OnlineLearner":
        if not blob or blob == b"\x00":
            return cls.fresh()
        payload = pickle.loads(blob)
        return cls(
            vectorizer=payload["vectorizer"],
            classifiers=payload["classifiers"],
            is_initialized=bool(payload.get("is_initialized", False)),
        )
