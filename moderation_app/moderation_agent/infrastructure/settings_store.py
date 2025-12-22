from __future__ import annotations

from dataclasses import replace
import json

from moderation_agent.domain import SystemSettings

from .db import connect


class SettingsStore:
    def load(self) -> SystemSettings:
        with connect() as conn:
            row = conn.execute(
                "SELECT enabled, retrain_gold_threshold, new_gold_since_last_train, settings_json "
                "FROM system_settings WHERE id=1"
            ).fetchone()

        if not row:
            return SystemSettings()

        try:
            extra = json.loads(row["settings_json"] or "{}")
        except Exception:  # noqa: BLE001
            extra = {}

        base = SystemSettings(
            enabled=bool(int(row["enabled"])),
            retrain_gold_threshold=int(row["retrain_gold_threshold"]),
            new_gold_since_last_train=int(row["new_gold_since_last_train"]),
        )

        overrides = {
            k: extra[k]
            for k in [
                "allow_max",
                "pending_min_default",
                "delete_min_default",
                "pending_min_by_label",
                "delete_min_by_label",
                "violent_threat_delete_min",
            ]
            if k in (extra or {})
        }

        out = replace(base, **overrides)

        # Project defaults (kept here so Web doesn't define business rules)
        if not out.pending_min_by_label:
            out = replace(
                out,
                pending_min_by_label={
                    "spam": 0.60,
                    "profanity": 0.55,
                    "abuse": 0.60,
                    "hate_speech": 0.70,
                    "threat": 0.50,
                },
            )
        if not out.delete_min_by_label:
            out = replace(
                out,
                delete_min_by_label={
                    "spam": 0.95,
                    "profanity": 0.95,
                    "abuse": 0.90,
                },
            )

        return out

    def save(self, settings: SystemSettings) -> None:
        extra = {
            "allow_max": settings.allow_max,
            "pending_min_default": settings.pending_min_default,
            "delete_min_default": settings.delete_min_default,
            "pending_min_by_label": settings.pending_min_by_label,
            "delete_min_by_label": settings.delete_min_by_label,
            "violent_threat_delete_min": settings.violent_threat_delete_min,
        }

        with connect() as conn:
            conn.execute(
                "UPDATE system_settings SET enabled=?, retrain_gold_threshold=?, new_gold_since_last_train=?, settings_json=? WHERE id=1",
                (
                    1 if settings.enabled else 0,
                    int(settings.retrain_gold_threshold),
                    int(settings.new_gold_since_last_train),
                    json.dumps(extra, ensure_ascii=False),
                ),
            )
            conn.commit()

    def increment_gold_counter(self, delta: int = 1) -> int:
        with connect() as conn:
            conn.execute(
                "UPDATE system_settings SET new_gold_since_last_train = new_gold_since_last_train + ? WHERE id=1",
                (int(delta),),
            )
            conn.commit()
            row = conn.execute("SELECT new_gold_since_last_train FROM system_settings WHERE id=1").fetchone()
            return int(row["new_gold_since_last_train"] if row else 0)

    def reset_gold_counter(self) -> None:
        with connect() as conn:
            conn.execute("UPDATE system_settings SET new_gold_since_last_train=0 WHERE id=1")
            conn.commit()
