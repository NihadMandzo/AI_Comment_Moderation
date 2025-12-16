from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .db import connect, init_db
from .ml_baseline import LABELS, baseline_model
from .ml_online import OnlineLearner
from .rules import profanity_score, spam_score, threat_score


BASELINE_WEIGHT = 0.70
LEARNED_WEIGHT = 0.30

# Per-label decision thresholds (helps avoid false positives on mild "abuse" scores).
LABEL_HIDE_THRESHOLDS: Dict[str, float] = {
    "hate_speech": 0.75,
    "abuse": 0.75,
    "threat": 0.65,
    "profanity": 0.65,
    "spam": 0.65,
}

# Simple in-memory rate limit for reports per IP
REPORTS_PER_MINUTE = 10


app = FastAPI(title="Moderation App")
APP_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


_model_lock = threading.Lock()
_online: Optional[OnlineLearner] = None

_rate_lock = threading.Lock()
_rate_state: Dict[str, List[float]] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rate_limit_ok(ip: str) -> bool:
    if not ip:
        return True
    cutoff = time.time() - 60
    with _rate_lock:
        times = [t for t in _rate_state.get(ip, []) if t >= cutoff]
        if len(times) >= REPORTS_PER_MINUTE:
            _rate_state[ip] = times
            return False
        times.append(time.time())
        _rate_state[ip] = times
        return True


def load_online() -> OnlineLearner:
    global _online
    with _model_lock:
        if _online is not None:
            return _online
        with connect() as conn:
            row = conn.execute("SELECT model_blob FROM training_state WHERE id=1").fetchone()
            blob = bytes(row["model_blob"]) if row else b"\x00"
        _online = OnlineLearner.loads(blob)
        return _online


def combine_scores(baseline: Dict[str, float], learned: Dict[str, float]) -> Dict[str, float]:
    final: Dict[str, float] = {}
    for label in LABELS:
        b = float(baseline.get(label, 0.0))
        l = float(learned.get(label, 0.0))
        final[label] = float(max(0.0, min(1.0, BASELINE_WEIGHT * b + LEARNED_WEIGHT * l)))
    return final


def max_label(scores: Dict[str, float]) -> tuple[str, float]:
    if not scores:
        return "", 0.0
    m_label = max(scores, key=lambda k: scores[k])
    return m_label, float(scores[m_label])


def insert_report(comment_id: int, kind: str, selected: List[str], note: str) -> None:
    kind_norm = (kind or "BAD").upper()
    labels_json = json.dumps({l: 1 for l in selected}, ensure_ascii=False)
    with connect() as conn:
        conn.execute(
            "INSERT INTO reports(comment_id, created_at, kind, labels_json, note) VALUES (?,?,?,?,?)",
            (comment_id, now_iso(), kind_norm, labels_json, note),
        )
        conn.execute("UPDATE comments SET report_count = report_count + 1 WHERE id=?", (comment_id,))
        conn.commit()


def retrain_from_all_reports() -> int:
    """Rebuild the online learner from all stored reports.

    This is useful after seeding data or when you want a clean training pass.
    Returns the number of report rows used.
    """
    learner = OnlineLearner.fresh()
    with connect() as conn:
        rows = conn.execute(
            "SELECT r.kind, r.labels_json, c.text "
            "FROM reports r JOIN comments c ON c.id=r.comment_id "
            "ORDER BY r.id ASC"
        ).fetchall()

    used = 0
    for r in rows:
        kind = str(r["kind"] or "BAD")
        labels_json = r["labels_json"] or "{}"
        try:
            d = json.loads(labels_json)
            selected = [k for k, v in (d or {}).items() if v and k in LABELS]
        except Exception:  # noqa: BLE001
            selected = []
        learner.update(str(r["text"] or ""), selected, kind=kind)
        used += 1

    with connect() as conn:
        conn.execute("UPDATE training_state SET model_blob=? WHERE id=1", (learner.dumps(),))
        conn.commit()

    global _online
    with _model_lock:
        _online = learner
    return used


def rescore_existing_comments() -> int:
    """Recompute baseline/learned/final scores for existing comments.

    Keeps DELETED comments as DELETED (still used for training if they have reports).
    Returns number of rescored rows.
    """
    learner = load_online()
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, text, status FROM comments ORDER BY id ASC"
        ).fetchall()

    updated = 0
    for r in rows:
        cid = int(r["id"])
        text = str(r["text"] or "")
        status = str(r["status"] or "")

        baseline = baseline_model.predict(text)
        learned = learner.predict_proba(text)
        final = combine_scores(baseline, learned)

        # Hard rule overrides
        rule_overrides = {
            "profanity": float(profanity_score(text)),
            "threat": float(threat_score(text)),
            "spam": float(spam_score(text)),
        }
        for k, v in rule_overrides.items():
            final[k] = float(max(final.get(k, 0.0), v))

        top_label, top_score = max_label(final)

        new_status = status
        if status != "DELETED":
            new_status = "APPROVED"
            for label, thr in LABEL_HIDE_THRESHOLDS.items():
                if float(final.get(label, 0.0)) >= float(thr):
                    new_status = "HIDDEN"
                    break

        with connect() as conn:
            conn.execute(
                "UPDATE comments SET status=?, baseline_labels_json=?, learned_labels_json=?, final_labels_json=?, final_max_label=?, final_max_score=? WHERE id=?",
                (
                    new_status,
                    json.dumps(baseline, ensure_ascii=False),
                    json.dumps(learned, ensure_ascii=False),
                    json.dumps(final, ensure_ascii=False),
                    top_label,
                    float(top_score),
                    cid,
                ),
            )
            conn.commit()
        updated += 1

    return updated


@app.on_event("startup")
def _startup() -> None:
    init_db()
    load_online()


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, text, created_at, final_labels_json, final_max_label, final_max_score, report_count "
            "FROM comments WHERE status='APPROVED' ORDER BY id DESC LIMIT 200"
        ).fetchall()
    comments = []
    for r in rows:
        final = json.loads(r["final_labels_json"]) if r["final_labels_json"] else {}
        comments.append(
            {
                "id": r["id"],
                "text": r["text"],
                "created_at": r["created_at"],
                "final": final,
                "final_max_label": r["final_max_label"],
                "final_max_score": r["final_max_score"],
                "report_count": r["report_count"],
            }
        )
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "labels": LABELS,
            "comments": comments,
        },
    )


@app.post("/comments")
def create_comment(text: str = Form(...)) -> RedirectResponse:
    t = (text or "").strip()
    if not t:
        return RedirectResponse(url="/", status_code=303)

    baseline = baseline_model.predict(t)
    online = load_online()
    learned = online.predict_proba(t)
    final = combine_scores(baseline, learned)

    # Hard, uni-project-safe rule overrides: obvious cases should work immediately.
    rule_overrides = {
        "profanity": float(profanity_score(t)),
        "threat": float(threat_score(t)),
        "spam": float(spam_score(t)),
    }
    for k, v in rule_overrides.items():
        final[k] = float(max(final.get(k, 0.0), v))

    top_label, top_score = max_label(final)

    status = "APPROVED"
    for label, thr in LABEL_HIDE_THRESHOLDS.items():
        if float(final.get(label, 0.0)) >= float(thr):
            status = "HIDDEN"
            break

    print(f"[moderation] status={status} top={top_label}:{top_score:.3f} text={t[:80]!r}")

    with connect() as conn:
        conn.execute(
            "INSERT INTO comments(text, created_at, status, baseline_labels_json, learned_labels_json, final_labels_json, "
            "final_max_label, final_max_score, report_count) VALUES (?,?,?,?,?,?,?,?,0)",
            (
                t,
                now_iso(),
                status,
                json.dumps(baseline, ensure_ascii=False),
                json.dumps(learned, ensure_ascii=False),
                json.dumps(final, ensure_ascii=False),
                top_label,
                top_score,
            ),
        )
        conn.commit()

    return RedirectResponse(url="/", status_code=303)


@app.post("/report/{comment_id}")
async def report_comment(comment_id: int, request: Request) -> JSONResponse:
    ip = request.client.host if request.client else ""
    if not rate_limit_ok(ip):
        return JSONResponse({"ok": False, "error": "rate_limited"}, status_code=429)

    payload = await request.json()
    labels = payload.get("labels") or []
    note = (payload.get("note") or "").strip()

    selected = [l for l in labels if l in LABELS]

    with connect() as conn:
        row = conn.execute("SELECT id, text FROM comments WHERE id=? AND status='APPROVED'", (comment_id,)).fetchone()
        if not row:
            return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
        text = row["text"]

    insert_report(comment_id=comment_id, kind="BAD", selected=selected, note=note)

    # Online update
    online = load_online()
    with _model_lock:
        online.update(text, selected, kind="BAD")
        with connect() as conn:
            conn.execute("UPDATE training_state SET model_blob=? WHERE id=1", (online.dumps(),))
            conn.commit()

    return JSONResponse({"ok": True})


@app.get("/moderation", response_class=HTMLResponse)
def moderation(request: Request) -> HTMLResponse:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, text, created_at, status, final_labels_json, final_max_label, final_max_score, report_count "
            "FROM comments ORDER BY id DESC LIMIT 300"
        ).fetchall()

    grouped: Dict[str, List[Dict]] = {"APPROVED": [], "HIDDEN": [], "DELETED": []}
    all_items: List[Dict] = []
    for r in rows:
        final = json.loads(r["final_labels_json"]) if r["final_labels_json"] else {}
        item = {
            "id": r["id"],
            "text": r["text"],
            "created_at": r["created_at"],
            "status": r["status"],
            "final": final,
            "final_max_label": r["final_max_label"],
            "final_max_score": r["final_max_score"],
            "report_count": r["report_count"],
        }
        all_items.append(item)
        if item["status"] in grouped:
            grouped[item["status"]].append(item)

    return templates.TemplateResponse(
        "moderation.html",
        {
            "request": request,
            "labels": LABELS,
            "all_comments": all_items,
            "approved_comments": grouped["APPROVED"],
            "hidden_comments": grouped["HIDDEN"],
            "deleted_comments": grouped["DELETED"],
        },
    )


@app.post("/moderation/{comment_id}/approve")
def approve(comment_id: int) -> RedirectResponse:
    with connect() as conn:
        conn.execute("UPDATE comments SET status='APPROVED' WHERE id=?", (comment_id,))
        conn.commit()
    return RedirectResponse(url="/moderation", status_code=303)


@app.post("/moderation/{comment_id}/good")
def mark_good(comment_id: int) -> RedirectResponse:
    # Moderator says: this hidden comment is actually OK.
    with connect() as conn:
        row = conn.execute("SELECT id, text FROM comments WHERE id=?", (comment_id,)).fetchone()
        if not row:
            return RedirectResponse(url="/moderation", status_code=303)
        text = row["text"]

        conn.execute("UPDATE comments SET status='APPROVED' WHERE id=?", (comment_id,))
        conn.commit()

    insert_report(comment_id=comment_id, kind="GOOD", selected=[], note="moderation:good")

    online = load_online()
    with _model_lock:
        online.update(text, [], kind="GOOD")
        with connect() as conn:
            conn.execute("UPDATE training_state SET model_blob=? WHERE id=1", (online.dumps(),))
            conn.commit()

    return RedirectResponse(url="/moderation", status_code=303)


@app.post("/moderation/{comment_id}/bad")
async def mark_bad(comment_id: int, request: Request) -> JSONResponse:
    # Moderator says: this comment is bad, with multi-labels.
    payload = await request.json()
    labels = payload.get("labels") or []
    note = (payload.get("note") or "").strip()
    selected = [l for l in labels if l in LABELS]

    with connect() as conn:
        row = conn.execute("SELECT id, text FROM comments WHERE id=?", (comment_id,)).fetchone()
        if not row:
            return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
        text = row["text"]

        conn.execute("UPDATE comments SET status='DELETED' WHERE id=?", (comment_id,))
        conn.commit()

    insert_report(comment_id=comment_id, kind="BAD", selected=selected, note=(note or "moderation:bad"))

    online = load_online()
    with _model_lock:
        online.update(text, selected, kind="BAD")
        with connect() as conn:
            conn.execute("UPDATE training_state SET model_blob=? WHERE id=1", (online.dumps(),))
            conn.commit()

    return JSONResponse({"ok": True})


@app.post("/moderation/train")
def train_button(request: Request):
    used = retrain_from_all_reports()
    rescored = rescore_existing_comments()

    accept = (request.headers.get("accept") or "").lower()
    wants_html = "text/html" in accept
    if wants_html:
        return RedirectResponse(url="/moderation", status_code=303)

    return JSONResponse({"ok": True, "reports_used": used, "comments_rescored": rescored})


@app.post("/moderation/{comment_id}/delete")
def delete(comment_id: int) -> RedirectResponse:
    with connect() as conn:
        conn.execute("UPDATE comments SET status='DELETED' WHERE id=?", (comment_id,))
        conn.commit()
    return RedirectResponse(url="/moderation", status_code=303)
