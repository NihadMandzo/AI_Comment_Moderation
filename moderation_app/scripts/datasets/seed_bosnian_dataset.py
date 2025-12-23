from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# Allow running as: python scripts/datasets/seed_bosnian_dataset.py
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from moderation_agent.ml import LABELS  # noqa: E402
from moderation_agent.ml.online import OnlineLearner  # noqa: E402

# Prefer shared infrastructure entrypoint (Clean Architecture):
from moderation_agent.infrastructure import connect, init_db  # noqa: E402


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


GOOD_COMMENTS = [
    "Prelijepo izlaganje, svaka čast!",
    "Baš si mi lijep.",
    "Lijep si.",
    "Hvala ti na pomoći, mnogo znači.",
    "Odlično objašnjenje, sve mi je jasno.",
    "Slažem se s tobom, dobar argument.",
    "Super ideja, podržavam!",
    "Molim te pošalji detalje kada stigneš.",
    "Bravo, odličan rad.",
    "Zanimljivo mišljenje, hvala što dijeliš.",
]

# Each bad example includes labels you want the learner to associate.
BAD_EXAMPLES = [
    {
        "text": "Jebem ti mater, glup si.",
        "labels": ["profanity", "abuse"],
    },
    {
        "text": "Ubiću te ako još jednom to kažeš.",
        "labels": ["threat"],
    },
    {
        "text": "Mrzim vas sve, vi ste odvratni.",
        "labels": ["abuse"],
    },
    {
        "text": "Kretenu jedan, začepi.",
        "labels": ["abuse"],
    },
    {
        "text": "Klikni ovdje http://spammy.site i zaradi 1000€ odmah!!!",
        "labels": ["spam"],
    },
    {
        "text": "www.super-ponuda.biz najbolja ponuda ponuda ponuda ponuda",
        "labels": ["spam"],
    },
    {
        "text": "Zaklaću te, kunem se.",
        "labels": ["threat"],
    },
]


def insert_comment(text: str, status: str) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO comments(text, created_at, status, baseline_labels_json, learned_labels_json, final_labels_json, final_max_label, final_max_score, report_count) "
            "VALUES (?,?,?,?,?,?,?,?,0)",
            (
                text,
                now_iso(),
                status,
                json.dumps({k: 0.0 for k in LABELS}, ensure_ascii=False),
                json.dumps({k: 0.0 for k in LABELS}, ensure_ascii=False),
                json.dumps({k: 0.0 for k in LABELS}, ensure_ascii=False),
                "",
                0.0,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def insert_report(comment_id: int, kind: str, labels: list[str], note: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO reports(comment_id, created_at, kind, labels_json, note) VALUES (?,?,?,?,?)",
            (comment_id, now_iso(), kind, json.dumps({l: 1 for l in labels}, ensure_ascii=False), note),
        )
        conn.execute("UPDATE comments SET report_count = report_count + 1 WHERE id=?", (comment_id,))
        conn.commit()


def retrain_from_reports() -> int:
    learner = OnlineLearner.fresh()
    with connect() as conn:
        rows = conn.execute(
            "SELECT r.kind, r.labels_json, c.text FROM reports r JOIN comments c ON c.id=r.comment_id ORDER BY r.id ASC"
        ).fetchall()

    used = 0
    for r in rows:
        kind = str(r["kind"] or "BAD")
        try:
            d = json.loads(r["labels_json"] or "{}")
            selected = [k for k, v in (d or {}).items() if v and k in LABELS]
        except Exception:
            selected = []
        learner.update(str(r["text"] or ""), selected, kind=kind)
        used += 1

    with connect() as conn:
        conn.execute("UPDATE training_state SET model_blob=? WHERE id=1", (learner.dumps(),))
        conn.commit()

    return used


def main() -> None:
    init_db()

    # Insert GOOD comments as APPROVED + add GOOD reports (negative training examples)
    for text in GOOD_COMMENTS:
        cid = insert_comment(text, status="APPROVED")
        insert_report(cid, kind="GOOD", labels=[], note="seed:good")

    # Insert BAD comments as DELETED (not public) + add BAD reports with labels
    for ex in BAD_EXAMPLES:
        cid = insert_comment(ex["text"], status="DELETED")
        insert_report(cid, kind="BAD", labels=ex["labels"], note="seed:bad")

    used = retrain_from_reports()
    print(f"Seeded {len(GOOD_COMMENTS)} good and {len(BAD_EXAMPLES)} bad examples. Retrained from {used} reports.")


if __name__ == "__main__":
    main()
