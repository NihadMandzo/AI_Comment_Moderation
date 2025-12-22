from __future__ import annotations

import threading
from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from moderation_agent.text.heuristics import abuse_score, profanity_score, spam_score, threat_score


LABELS = ["hate_speech", "abuse", "threat", "profanity", "spam"]


class BaselineModel:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._pipe_hate = None
        self._toxic_tok = None
        self._toxic_model = None

    def _load(self) -> None:
        hate_tok = AutoTokenizer.from_pretrained(
            "classla/bcms-bertic-frenk-hate",
            use_fast=False,
        )
        self._pipe_hate = pipeline(
            task="text-classification",
            model="classla/bcms-bertic-frenk-hate",
            tokenizer=hate_tok,
            top_k=None,
        )

        self._toxic_tok = AutoTokenizer.from_pretrained(
            "unitary/multilingual-toxic-xlm-roberta",
            use_fast=False,
        )
        self._toxic_model = AutoModelForSequenceClassification.from_pretrained(
            "unitary/multilingual-toxic-xlm-roberta",
        )
        self._toxic_model.eval()

        self._loaded = True

    def predict(self, text: str) -> Dict[str, float]:
        t = (text or "").strip()
        if not t:
            return {k: 0.0 for k in LABELS}

        with self._lock:
            if not self._loaded:
                try:
                    self._load()
                except Exception as e:  # noqa: BLE001
                    print(f"[baseline] WARNING: could not load models: {e}")
                    return {
                        "hate_speech": 0.0,
                        "abuse": 0.0,
                        "threat": 0.0,
                        "profanity": 0.0,
                        "spam": float(spam_score(t)),
                    }

        bertic_hate = 0.0
        try:
            out = self._pipe_hate(t)
            items = out[0] if out and isinstance(out[0], list) else out
            for item in items:
                label = str(item.get("label", "")).lower()
                score = float(item.get("score", 0.0))
                if "hate" in label or "offen" in label or "toxic" in label:
                    bertic_hate = max(bertic_hate, score)
        except Exception as e:  # noqa: BLE001
            print(f"[baseline] WARNING: bertic inference failed: {e}")

        detox = {
            "toxic": 0.0,
            "severe_toxic": 0.0,
            "obscene": 0.0,
            "threat": 0.0,
            "insult": 0.0,
            "identity_hate": 0.0,
        }
        try:
            assert self._toxic_tok is not None
            assert self._toxic_model is not None

            enc = self._toxic_tok(t, return_tensors="pt", truncation=True)
            with torch.no_grad():
                out = self._toxic_model(**enc)

            logits = out.logits
            probs = torch.sigmoid(logits).squeeze(0).tolist()

            id2label = getattr(self._toxic_model.config, "id2label", {}) or {}
            for i, p in enumerate(probs):
                label = str(id2label.get(i, i)).lower()
                if label in detox:
                    detox[label] = max(detox[label], float(p))
        except Exception as e:  # noqa: BLE001
            print(f"[baseline] WARNING: toxic inference failed: {e}")

        hate_speech = max(bertic_hate, detox["identity_hate"])
        threat = max(detox["threat"], float(threat_score(t)))
        profanity = max(float(profanity_score(t)), 0.35 * float(detox["obscene"]))
        abuse = max(detox["insult"], detox["toxic"], detox["severe_toxic"], float(abuse_score(t)))
        spam = float(spam_score(t))

        return {
            "hate_speech": float(hate_speech),
            "abuse": float(abuse),
            "threat": float(threat),
            "profanity": float(profanity),
            "spam": float(spam),
        }


baseline_model = BaselineModel()
