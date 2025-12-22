from __future__ import annotations

import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data" / "moderation.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with connect() as conn:
        # Ensure WAL/SHM live next to the DB file under ./data
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass
        conn.executescript(schema)

        cols = [r["name"] for r in conn.execute("PRAGMA table_info(reports)").fetchall()]
        if "kind" not in cols:
            conn.execute("ALTER TABLE reports ADD COLUMN kind TEXT NOT NULL DEFAULT 'BAD'")
        if "reporter_hash" not in cols:
            conn.execute("ALTER TABLE reports ADD COLUMN reporter_hash TEXT")
        if "confidence" not in cols:
            conn.execute("ALTER TABLE reports ADD COLUMN confidence REAL NOT NULL DEFAULT 0.3")
        if "used_for_training" not in cols:
            conn.execute("ALTER TABLE reports ADD COLUMN used_for_training INTEGER NOT NULL DEFAULT 0")

        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_reports_dedupe "
            "ON reports(comment_id, reporter_hash, kind, labels_json)"
        )

        conn.execute(
            "CREATE TABLE IF NOT EXISTS system_settings ("
            "id INTEGER PRIMARY KEY CHECK (id=1),"
            "enabled INTEGER NOT NULL DEFAULT 1,"
            "retrain_gold_threshold INTEGER NOT NULL DEFAULT 50,"
            "new_gold_since_last_train INTEGER NOT NULL DEFAULT 0,"
            "settings_json TEXT"
            ")"
        )
        conn.execute(
            "INSERT OR IGNORE INTO system_settings(id, enabled, retrain_gold_threshold, new_gold_since_last_train, settings_json) "
            "VALUES (1, 1, 50, 0, '{}')"
        )

        conn.execute(
            "CREATE TABLE IF NOT EXISTS model_versions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "created_at TEXT NOT NULL,"
            "trained_on_reports INTEGER NOT NULL,"
            "is_active INTEGER NOT NULL DEFAULT 0,"
            "model_blob BLOB NOT NULL"
            ")"
        )

        cols_ts = [r["name"] for r in conn.execute("PRAGMA table_info(training_state)").fetchall()]
        if "active_model_version_id" not in cols_ts:
            try:
                conn.execute("ALTER TABLE training_state ADD COLUMN active_model_version_id INTEGER")
            except sqlite3.OperationalError:
                pass

        conn.commit()
