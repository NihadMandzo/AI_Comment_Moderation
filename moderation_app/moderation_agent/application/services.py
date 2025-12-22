from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from moderation_agent.text.normalize import normalize_text, two_channel

from moderation_agent.domain import (
    Decision,
    Message,
    MessageStatus,
    ModerationRules,
    Prediction,
    ReviewKind,
)
from moderation_agent.infrastructure import ModelStore, SettingsStore, connect
from moderation_agent.ml.moderation_classifier import ModerationClassifier, rule_overrides


LABELS = ["hate_speech", "abuse", "threat", "profanity", "spam"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def max_label(scores: Dict[str, float]) -> tuple[str, float]:
    if not scores:
        return "", 0.0
    k = max(scores, key=lambda x: scores[x])
    return k, float(scores[k])


@dataclass(frozen=True)
class QueueItem:
    message: Message
    text_norm: str


class QueueService:
    def enqueue(self, text: str) -> int:
        t = (text or "").strip()
        if not t:
            raise ValueError("text required")

        with connect() as conn:
            conn.execute(
                "INSERT INTO comments(text, created_at, status, baseline_labels_json, learned_labels_json, final_labels_json, final_max_label, final_max_score, report_count) "
                "VALUES (?,?,?,?,?,?,?,?,0)",
                (t, now_iso(), MessageStatus.QUEUED.value, "{}", "{}", "{}", "", 0.0),
            )
            conn.commit()
            row = conn.execute("SELECT last_insert_rowid() AS id").fetchone()
            return int(row["id"])

    def dequeue_next(self) -> Optional[QueueItem]:
        """Atomically claim one QUEUED message.

        Idempotence: we flip QUEUED→PROCESSING in a single update.
        """
        with connect() as conn:
            row = conn.execute(
                "SELECT id, text, created_at, status, report_count FROM comments WHERE status=? ORDER BY id ASC LIMIT 1",
                (MessageStatus.QUEUED.value,),
            ).fetchone()
            if not row:
                return None

            message_id = int(row["id"])
            updated = conn.execute(
                "UPDATE comments SET status=? WHERE id=? AND status=?",
                (MessageStatus.PROCESSING.value, message_id, MessageStatus.QUEUED.value),
            ).rowcount
            conn.commit()

            if updated != 1:
                return None

            text = str(row["text"] or "")
            ch = two_channel(text)
            msg = Message(
                id=message_id,
                text=text,
                created_at=str(row["created_at"] or ""),
                status=MessageStatus.PROCESSING,
                report_count=int(row["report_count"] or 0),
            )
            return QueueItem(message=msg, text_norm=ch.norm)

    def set_status(self, message_id: int, status: MessageStatus) -> None:
        with connect() as conn:
            conn.execute("UPDATE comments SET status=? WHERE id=?", (status.value, int(message_id)))
            conn.commit()


class ScoringService:
    def __init__(
        self,
        *,
        classifier: ModerationClassifier,
        settings_store: SettingsStore,
        rules: ModerationRules,
    ) -> None:
        self._classifier = classifier
        self._settings_store = settings_store
        self._rules = rules

    def _training_count(self) -> int:
        with connect() as conn:
            row = conn.execute("SELECT COUNT(1) AS n FROM reports WHERE used_for_training=1").fetchone()
            return int(row["n"] if row else 0)

    def _combine_scores(self, baseline: Dict[str, float], learned: Dict[str, float], *, training_count: int) -> Dict[str, float]:
        lw = float(min(0.5, (float(training_count) / 2000.0) * 0.5))
        bw = 1.0 - lw
        final: Dict[str, float] = {}
        for label in LABELS:
            b = float(baseline.get(label, 0.0))
            l = float(learned.get(label, 0.0))
            final[label] = float(max(0.0, min(1.0, bw * b + lw * l)))
        return final

    def score_and_decide(self, *, item: QueueItem) -> Prediction:
        settings = self._settings_store.load()
        training_count = self._training_count()

        raw = item.message.text
        norm = item.text_norm

        # THINK: model scores
        signals = self._classifier.predict_scores(raw, norm)
        baseline = {k: float(signals.get(f"baseline:{k}", 0.0)) for k in LABELS}
        learned = {k: float(signals.get(f"learned:{k}", 0.0)) for k in LABELS}

        final = self._combine_scores(baseline, learned, training_count=training_count)

        # Shared rule overrides (not web)
        violent_threat, overrides = rule_overrides(norm)
        for k, v in overrides.items():
            final[k] = float(max(final.get(k, 0.0), float(v)))

        top_label, top_score = max_label(final)
        decision, new_status = self._rules.decide(final, settings=settings, violent_threat_score=violent_threat)

        return Prediction(
            message_id=item.message.id,
            baseline_scores=baseline,
            learned_scores=learned,
            final_scores=final,
            top_label=top_label,
            top_score=float(top_score),
            decision=decision,
            new_status=new_status,
            model_version_id=self._active_model_version_id(),
        )

    def persist_prediction(self, *, prediction: Prediction) -> None:
        with connect() as conn:
            conn.execute(
                "UPDATE comments SET status=?, baseline_labels_json=?, learned_labels_json=?, final_labels_json=?, final_max_label=?, final_max_score=? WHERE id=?",
                (
                    prediction.new_status.value,
                    json.dumps(prediction.baseline_scores, ensure_ascii=False),
                    json.dumps(prediction.learned_scores, ensure_ascii=False),
                    json.dumps(prediction.final_scores, ensure_ascii=False),
                    prediction.top_label,
                    float(prediction.top_score),
                    int(prediction.message_id),
                ),
            )
            conn.commit()

    def rescore_all(self) -> int:
        with connect() as conn:
            rows = conn.execute("SELECT id, text, status FROM comments ORDER BY id ASC").fetchall()

        updated = 0
        for r in rows:
            message_id = int(r["id"])
            text = str(r["text"] or "")
            status = str(r["status"] or "")
            ch = two_channel(text)

            item = QueueItem(
                message=Message(
                    id=message_id,
                    text=text,
                    created_at="",
                    status=MessageStatus(status) if status in MessageStatus._value2member_map_ else MessageStatus.APPROVED,
                ),
                text_norm=ch.norm,
            )

            pred = self.score_and_decide(item=item)
            if status == MessageStatus.DELETED.value:
                pred = Prediction(
                    **{**pred.__dict__, "new_status": MessageStatus.DELETED},
                )

            self.persist_prediction(prediction=pred)
            updated += 1

        return updated

    def _active_model_version_id(self) -> Optional[int]:
        with connect() as conn:
            row = conn.execute("SELECT active_model_version_id FROM training_state WHERE id=1").fetchone()
            if not row:
                return None
            v = row["active_model_version_id"]
            return int(v) if v is not None else None


class ReviewService:
    def __init__(
        self,
        *,
        classifier: ModerationClassifier,
        settings_store: SettingsStore,
        model_store: ModelStore,
        consensus_min_reporters: int = 5,
    ) -> None:
        self._classifier = classifier
        self._settings_store = settings_store
        self._model_store = model_store
        self._consensus_min_reporters = int(consensus_min_reporters)

    def insert_report(
        self,
        *,
        message_id: int,
        kind: ReviewKind,
        labels: List[str],
        reporter_hash: str,
        confidence: float,
        used_for_training: bool,
        note: str,
    ) -> None:
        labels_json = json.dumps({l: 1 for l in labels}, ensure_ascii=False)
        with connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO reports(comment_id, created_at, kind, labels_json, reporter_hash, confidence, used_for_training, note) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        int(message_id),
                        now_iso(),
                        kind.value,
                        labels_json,
                        reporter_hash or "",
                        float(confidence),
                        1 if used_for_training else 0,
                        note,
                    ),
                )
            except Exception:
                return

            conn.execute("UPDATE comments SET report_count = report_count + 1 WHERE id=?", (int(message_id),))
            conn.commit()

    def maybe_train_from_consensus(self, *, message_id: int, text: str, kind: ReviewKind, labels: List[str]) -> bool:
        if kind != ReviewKind.BAD:
            return False

        text_norm = normalize_text(text)

        labels_json = json.dumps({l: 1 for l in labels}, ensure_ascii=False)
        with connect() as conn:
            already = conn.execute(
                "SELECT COUNT(DISTINCT reporter_hash) AS n "
                "FROM reports WHERE comment_id=? AND kind=? AND labels_json=? AND used_for_training=1 AND reporter_hash!=''",
                (int(message_id), kind.value, labels_json),
            ).fetchone()
            already_n = int(already["n"] if already else 0)
            if already_n >= self._consensus_min_reporters:
                return False

            total = conn.execute(
                "SELECT COUNT(DISTINCT reporter_hash) AS n "
                "FROM reports WHERE comment_id=? AND kind=? AND labels_json=? AND reporter_hash!=''",
                (int(message_id), kind.value, labels_json),
            ).fetchone()
            total_n = int(total["n"] if total else 0)
            if total_n < self._consensus_min_reporters:
                return False

            conn.execute(
                "UPDATE reports SET used_for_training=1 "
                "WHERE comment_id=? AND kind=? AND labels_json=? AND used_for_training=0 AND reporter_hash!=''",
                (int(message_id), kind.value, labels_json),
            )
            conn.commit()

            rows = conn.execute(
                "SELECT reporter_hash, MAX(confidence) AS c "
                "FROM reports WHERE comment_id=? AND kind=? AND labels_json=? AND reporter_hash!='' "
                "GROUP BY reporter_hash",
                (int(message_id), kind.value, labels_json),
            ).fetchall()

        n_reporters = int(len(rows))
        sum_conf = float(sum(float(r["c"] or 0.0) for r in rows))
        weight = float(max(sum_conf, 1.0 + 0.5 * max(0, n_reporters - self._consensus_min_reporters)))
        weight = float(max(0.1, min(2.0, weight)))

        self._classifier.update_online(text_norm, labels, kind=kind.value, weight=weight)
        self._model_store.save_active_model_blob(self._classifier.dumps_active_model())
        return True

    def moderator_mark_good(self, *, message_id: int, text: str) -> None:
        text_norm = normalize_text(text)
        self.insert_report(
            message_id=message_id,
            kind=ReviewKind.GOOD,
            labels=[],
            reporter_hash="moderator",
            confidence=1.0,
            used_for_training=True,
            note="moderation:good",
        )
        self._classifier.update_online(text_norm, [], kind="GOOD", weight=1.0)
        self._model_store.save_active_model_blob(self._classifier.dumps_active_model())
        self._settings_store.increment_gold_counter(1)

    def moderator_mark_bad(self, *, message_id: int, text: str, labels: List[str], note: str) -> None:
        text_norm = normalize_text(text)
        self.insert_report(
            message_id=message_id,
            kind=ReviewKind.BAD,
            labels=labels,
            reporter_hash="moderator",
            confidence=1.0,
            used_for_training=True,
            note=note or "moderation:bad",
        )
        self._classifier.update_online(text_norm, labels, kind="BAD", weight=1.0)
        self._model_store.save_active_model_blob(self._classifier.dumps_active_model())
        self._settings_store.increment_gold_counter(1)


class TrainingService:
    def __init__(
        self,
        *,
        classifier: ModerationClassifier,
        settings_store: SettingsStore,
        model_store: ModelStore,
    ) -> None:
        self._classifier = classifier
        self._settings_store = settings_store
        self._model_store = model_store

    def _load_training_rows(self) -> List[Tuple[str, List[str], float, str]]:
        with connect() as conn:
            rows = conn.execute(
                "SELECT r.kind, r.labels_json, r.confidence, c.text "
                "FROM reports r JOIN comments c ON c.id=r.comment_id "
                "WHERE r.used_for_training=1 "
                "ORDER BY r.id ASC"
            ).fetchall()

        out: List[Tuple[str, List[str], float, str]] = []
        for r in rows:
            kind = str(r["kind"] or "BAD").upper()
            labels_json = r["labels_json"] or "{}"
            conf = float(r["confidence"] or 1.0)
            try:
                d = json.loads(labels_json)
                labels = [k for k, v in (d or {}).items() if v]
            except Exception:  # noqa: BLE001
                labels = []
            text_norm = normalize_text(str(r["text"] or ""))
            out.append((kind, labels, conf, text_norm))
        return out

    def train_model(self, *, activate: bool) -> Optional[int]:
        rows = self._load_training_rows()
        used = self._classifier.rebuild_from_training_rows(rows)
        blob = self._classifier.dumps_active_model()

        version_id: Optional[int] = None
        with connect() as conn:
            conn.execute("UPDATE model_versions SET is_active=0 WHERE is_active=1")
            conn.execute(
                "INSERT INTO model_versions(created_at, trained_on_reports, is_active, model_blob) VALUES (?,?,?,?)",
                (now_iso(), int(used), 1 if activate else 0, blob),
            )
            row = conn.execute("SELECT last_insert_rowid() AS id").fetchone()
            version_id = int(row["id"]) if row else None

            conn.execute(
                "UPDATE training_state SET model_blob=?, active_model_version_id=? WHERE id=1",
                (blob, version_id),
            )
            conn.commit()

        # Also keep a simple direct write path (for future storage options)
        self._model_store.save_active_model_blob(blob)
        self._model_store.set_active_model_version(version_id)

        if activate:
            self._settings_store.reset_gold_counter()

        return version_id
