from __future__ import annotations

import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "moderation.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with connect() as conn:
        conn.executescript(schema)

        # Lightweight migrations for existing DBs
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(reports)").fetchall()]
        if "kind" not in cols:
            conn.execute("ALTER TABLE reports ADD COLUMN kind TEXT NOT NULL DEFAULT 'BAD'")

        if "reporter_hash" not in cols:
            conn.execute("ALTER TABLE reports ADD COLUMN reporter_hash TEXT")

        if "confidence" not in cols:
            conn.execute("ALTER TABLE reports ADD COLUMN confidence REAL NOT NULL DEFAULT 0.3")

        if "used_for_training" not in cols:
            conn.execute("ALTER TABLE reports ADD COLUMN used_for_training INTEGER NOT NULL DEFAULT 0")

        conn.commit()
