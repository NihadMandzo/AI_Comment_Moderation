from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
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
from .normalize import normalize_text, two_channel
from .rules import nonviolent_threat_score, profanity_score, spam_score, threat_score, violent_threat_score


# 3-zone moderation: ALLOW / PENDING_REVIEW / DELETE
ALLOW_MAX = 0.35
PENDING_MIN = 0.35
DELETE_MIN_DEFAULT = 0.80

PENDING_MIN_BY_LABEL: Dict[str, float] = {
    "spam": 0.60,
    "profanity": 0.55,
    "abuse": 0.60,
    "hate_speech": 0.70,
    "threat": 0.50,
}

# Conservative deletes for weaker categories.
DELETE_MIN_BY_LABEL: Dict[str, float] = {
    "spam": 0.95,
    "profanity": 0.95,
    "abuse": 0.90,
}

STATUS_APPROVED = "APPROVED"
STATUS_PENDING = "PENDING_REVIEW"
STATUS_DELETED = "DELETED"

# True hard rule: violent threats get deleted immediately.
VIOLENT_THREAT_DELETE_MIN = 0.80

# Simple in-memory rate limit for reports per IP
REPORTS_PER_MINUTE = 10

# Weak-label learning gate
CONSENSUS_MIN_REPORTERS = 2

# Server-side salt file to hash reporters without storing raw IP/UA.
SALT_ENV_VAR = "MODERATION_SALT"


app = FastAPI(title="Moderation App")
APP_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


_model_lock = threading.Lock()
_online: Optional[OnlineLearner] = None

_rate_lock = threading.Lock()
_rate_state: Dict[str, List[float]] = {}

_training_count_lock = threading.Lock()
_training_count_cache: Optional[int] = None


def _server_salt_path() -> Path:
    return APP_DIR / "_server_salt.txt"


def _get_server_salt() -> str:
    env = (os.environ.get(SALT_ENV_VAR) or "").strip()
    if env:
        return env

    p = _server_salt_path()
    try:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
        salt = secrets.token_hex(32)
        p.write_text(salt, encoding="utf-8")
        return salt
    except Exception:  # noqa: BLE001
        # If filesystem is read-only, still provide a best-effort salt (non-persistent).
        return secrets.token_hex(32)


def _reporter_hash(ip: str, user_agent: str, ts: Optional[datetime] = None) -> str:
    """Hash IP+UA with monthly rotation, so we can count 'independent reporters'."""
    ip = (ip or "").strip()
    ua = (user_agent or "").strip()
    if not ip:
        return ""

    dt = ts or datetime.now(timezone.utc)
    month = dt.strftime("%Y-%m")

    base = _get_server_salt()
    rot = hashlib.sha256(f"{base}:{month}".encode("utf-8")).hexdigest()
    return hashlib.sha256(f"{rot}:{ip}:{ua}".encode("utf-8")).hexdigest()


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


def _training_count_db() -> int:
    with connect() as conn:
        row = conn.execute("SELECT COUNT(1) AS n FROM reports WHERE used_for_training=1").fetchone()
        return int(row["n"] if row else 0)


def refresh_training_count() -> int:
    global _training_count_cache
    n = _training_count_db()
    with _training_count_lock:
        _training_count_cache = n
    return n


def get_training_count() -> int:
    with _training_count_lock:
        cached = _training_count_cache
    if cached is not None:
        return int(cached)
    return refresh_training_count()


def _learned_weight(training_count: int) -> float:
    # Increase learned influence as we accumulate gold/consensus labels.
    # Caps at 0.5.
    return float(min(0.5, (float(training_count) / 2000.0) * 0.5))


def combine_scores(
    baseline: Dict[str, float],
    learned: Dict[str, float],
    *,
    training_count: int,
) -> Dict[str, float]:
    final: Dict[str, float] = {}
    lw = _learned_weight(training_count)
    bw = 1.0 - lw
    for label in LABELS:
        b = float(baseline.get(label, 0.0))
        l = float(learned.get(label, 0.0))
        final[label] = float(max(0.0, min(1.0, bw * b + lw * l)))
    return final


def max_label(scores: Dict[str, float]) -> tuple[str, float]:
    if not scores:
        return "", 0.0
    m_label = max(scores, key=lambda k: scores[k])
    return m_label, float(scores[m_label])


def insert_report(
    *,
    comment_id: int,
    kind: str,
    selected: List[str],
    note: str,
    reporter_hash: str,
    confidence: float,
    used_for_training: int,
) -> None:
    kind_norm = (kind or "BAD").upper()
    labels_json = json.dumps({l: 1 for l in selected}, ensure_ascii=False)
    with connect() as conn:
        try:
            conn.execute(
                "INSERT INTO reports(comment_id, created_at, kind, labels_json, reporter_hash, confidence, used_for_training, note) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    comment_id,
                    now_iso(),
                    kind_norm,
                    labels_json,
                    (reporter_hash or ""),
                    float(confidence),
                    int(used_for_training),
                    note,
                ),
            )
        except sqlite3.IntegrityError:
            # Likely duplicate report due to UNIQUE index; ignore silently.
            return

        conn.execute("UPDATE comments SET report_count = report_count + 1 WHERE id=?", (comment_id,))
        conn.commit()

    if int(used_for_training) == 1:
        refresh_training_count()


def _decide_status(scores: Dict[str, float], *, violent_threat: float) -> str:
    if float(violent_threat) >= float(VIOLENT_THREAT_DELETE_MIN):
        return STATUS_DELETED

    # Delete zone (per label)
    for label in LABELS:
        s = float(scores.get(label, 0.0))
        delete_min = float(DELETE_MIN_BY_LABEL.get(label, DELETE_MIN_DEFAULT))
        if s >= delete_min:
            return STATUS_DELETED

    # Pending zone
    for label in LABELS:
        s = float(scores.get(label, 0.0))
        pending_min = float(PENDING_MIN_BY_LABEL.get(label, PENDING_MIN))
        if s >= pending_min:
            return STATUS_PENDING

    return STATUS_APPROVED


def _maybe_train_from_consensus(
    *,
    comment_id: int,
    text_norm: str,
    kind: str,
    selected: List[str],
) -> bool:
    """Train from user reports only once when a new label-set hits consensus."""
    kind_norm = (kind or "BAD").upper()
    if kind_norm != "BAD":
        return False

    labels_json = json.dumps({l: 1 for l in selected}, ensure_ascii=False)
    with connect() as conn:
        already = conn.execute(
            "SELECT COUNT(DISTINCT reporter_hash) AS n "
            "FROM reports WHERE comment_id=? AND kind=? AND labels_json=? AND used_for_training=1 AND reporter_hash!=''",
            (comment_id, kind_norm, labels_json),
        ).fetchone()
        already_n = int(already["n"] if already else 0)
        if already_n >= CONSENSUS_MIN_REPORTERS:
            return False

        total = conn.execute(
            "SELECT COUNT(DISTINCT reporter_hash) AS n "
            "FROM reports WHERE comment_id=? AND kind=? AND labels_json=? AND reporter_hash!=''",
            (comment_id, kind_norm, labels_json),
        ).fetchone()
        total_n = int(total["n"] if total else 0)
        if total_n < CONSENSUS_MIN_REPORTERS:
            return False

        conn.execute(
            "UPDATE reports SET used_for_training=1 "
            "WHERE comment_id=? AND kind=? AND labels_json=? AND used_for_training=0 AND reporter_hash!=''",
            (comment_id, kind_norm, labels_json),
        )
        conn.commit()

        # Weight grows with #independent reporters (and is capped).
        rows = conn.execute(
            "SELECT reporter_hash, MAX(confidence) AS c "
            "FROM reports WHERE comment_id=? AND kind=? AND labels_json=? AND reporter_hash!='' "
            "GROUP BY reporter_hash",
            (comment_id, kind_norm, labels_json),
        ).fetchall()
        n_reporters = int(len(rows))
        sum_conf = float(sum(float(r["c"] or 0.0) for r in rows))
        weight = float(max(sum_conf, 1.0 + 0.5 * max(0, n_reporters - CONSENSUS_MIN_REPORTERS)))
        weight = float(max(0.1, min(2.0, weight)))

    refresh_training_count()

    online = load_online()
    with _model_lock:
        online.update_weighted(text_norm, selected, kind=kind_norm, weight=weight)
        with connect() as conn:
            conn.execute("UPDATE training_state SET model_blob=? WHERE id=1", (online.dumps(),))
            conn.commit()
    return True


def retrain_from_all_reports() -> int:
    """Rebuild the online learner from all stored reports.

    This is useful after seeding data or when you want a clean training pass.
    Returns the number of report rows used.
    """
    learner = OnlineLearner.fresh()
    with connect() as conn:
        rows = conn.execute(
            "SELECT r.kind, r.labels_json, r.confidence, c.text "
            "FROM reports r JOIN comments c ON c.id=r.comment_id "
            "WHERE r.used_for_training=1 "
            "ORDER BY r.id ASC"
        ).fetchall()

    used = 0
    for r in rows:
        kind = str(r["kind"] or "BAD")
        labels_json = r["labels_json"] or "{}"
        conf = float(r["confidence"] or 1.0)
        try:
            d = json.loads(labels_json)
            selected = [k for k, v in (d or {}).items() if v and k in LABELS]
        except Exception:  # noqa: BLE001
            selected = []
        t_norm = normalize_text(str(r["text"] or ""))
        learner.update_weighted(t_norm, selected, kind=kind, weight=conf)
        used += 1

    with connect() as conn:
        conn.execute("UPDATE training_state SET model_blob=? WHERE id=1", (learner.dumps(),))
        conn.commit()

    refresh_training_count()

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
    training_count = get_training_count()
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, text, status FROM comments ORDER BY id ASC"
        ).fetchall()

    updated = 0
    for r in rows:
        cid = int(r["id"])
        text = str(r["text"] or "")
        status = str(r["status"] or "")

        ch = two_channel(text)

        baseline = baseline_model.predict(ch.raw)
        with _model_lock:
            learned = learner.predict_proba(ch.norm)
        final = combine_scores(baseline, learned, training_count=training_count)

        # Hard rule overrides
        vth = float(violent_threat_score(ch.norm))
        nth = float(nonviolent_threat_score(ch.norm))

        rule_overrides = {
            "profanity": float(profanity_score(ch.norm)),
            "threat": float(max(vth, nth, float(threat_score(ch.norm)))),
            "spam": float(spam_score(ch.norm)),
        }
        for k, v in rule_overrides.items():
            final[k] = float(max(final.get(k, 0.0), v))

        top_label, top_score = max_label(final)

        new_status = status
        if status != STATUS_DELETED:
            new_status = _decide_status(final, violent_threat=vth)

        # Migrate old HIDDEN into the new pending status.
        if new_status == "HIDDEN":
            new_status = STATUS_PENDING

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
    refresh_training_count()


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, text, created_at, final_labels_json, final_max_label, final_max_score, report_count "
            "FROM comments WHERE status=? ORDER BY id DESC LIMIT 200",
            (STATUS_APPROVED,),
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

    ch = two_channel(t)
    training_count = get_training_count()

    baseline = baseline_model.predict(ch.raw)
    online = load_online()
    with _model_lock:
        learned = online.predict_proba(ch.norm)
    final = combine_scores(baseline, learned, training_count=training_count)

    # Hard, uni-project-safe rule overrides: obvious cases should work immediately.
    vth = float(violent_threat_score(ch.norm))
    nth = float(nonviolent_threat_score(ch.norm))
    rule_overrides = {
        "profanity": float(profanity_score(ch.norm)),
        "threat": float(max(vth, nth, float(threat_score(ch.norm)))),
        "spam": float(spam_score(ch.norm)),
    }
    for k, v in rule_overrides.items():
        final[k] = float(max(final.get(k, 0.0), v))

    top_label, top_score = max_label(final)

    status = _decide_status(final, violent_threat=vth)

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
        row = conn.execute(
            "SELECT id, text FROM comments WHERE id=? AND status=?",
            (comment_id, STATUS_APPROVED),
        ).fetchone()
        if not row:
            return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
        text = row["text"]

    ua = request.headers.get("user-agent") or ""
    rep_hash = _reporter_hash(ip, ua)

    insert_report(
        comment_id=comment_id,
        kind="BAD",
        selected=selected,
        note=note,
        reporter_hash=rep_hash,
        confidence=0.5,
        used_for_training=0,
    )

    # Weak labels: train only on consensus.
    trained = _maybe_train_from_consensus(
        comment_id=comment_id,
        text_norm=normalize_text(str(text or "")),
        kind="BAD",
        selected=selected,
    )
    return JSONResponse({"ok": True, "trained": bool(trained)})


@app.get("/moderation", response_class=HTMLResponse)
def moderation(request: Request) -> HTMLResponse:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, text, created_at, status, final_labels_json, final_max_label, final_max_score, report_count "
            "FROM comments ORDER BY id DESC LIMIT 300"
        ).fetchall()

    grouped: Dict[str, List[Dict]] = {
        STATUS_APPROVED: [],
        STATUS_PENDING: [],
        "HIDDEN": [],
        STATUS_DELETED: [],
    }
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

        # Display legacy HIDDEN as PENDING_REVIEW.
        if item["status"] == "HIDDEN":
            item["status"] = STATUS_PENDING

        all_items.append(item)
        if item["status"] in grouped:
            grouped[item["status"]].append(item)

    return templates.TemplateResponse(
        "moderation.html",
        {
            "request": request,
            "labels": LABELS,
            "all_comments": all_items,
            "approved_comments": grouped[STATUS_APPROVED],
            "pending_comments": grouped[STATUS_PENDING],
            "deleted_comments": grouped[STATUS_DELETED],
        },
    )


@app.post("/moderation/{comment_id}/approve")
def approve(comment_id: int) -> RedirectResponse:
    with connect() as conn:
        conn.execute("UPDATE comments SET status=? WHERE id=?", (STATUS_APPROVED, comment_id))
        conn.commit()
    return RedirectResponse(url="/moderation", status_code=303)


@app.post("/moderation/{comment_id}/good")
def mark_good(comment_id: int) -> RedirectResponse:
    # Moderator says: this comment is OK.
    with connect() as conn:
        row = conn.execute("SELECT id, text FROM comments WHERE id=?", (comment_id,)).fetchone()
        if not row:
            return RedirectResponse(url="/moderation", status_code=303)
        text = row["text"]

        conn.execute("UPDATE comments SET status=? WHERE id=?", (STATUS_APPROVED, comment_id))
        conn.commit()

    insert_report(
        comment_id=comment_id,
        kind="GOOD",
        selected=[],
        note="moderation:good",
        reporter_hash="moderator",
        confidence=1.0,
        used_for_training=1,
    )

    online = load_online()
    with _model_lock:
        online.update_weighted(normalize_text(str(text or "")), [], kind="GOOD", weight=1.0)
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

        conn.execute("UPDATE comments SET status=? WHERE id=?", (STATUS_DELETED, comment_id))
        conn.commit()

    insert_report(
        comment_id=comment_id,
        kind="BAD",
        selected=selected,
        note=(note or "moderation:bad"),
        reporter_hash="moderator",
        confidence=1.0,
        used_for_training=1,
    )

    online = load_online()
    with _model_lock:
        online.update_weighted(normalize_text(str(text or "")), selected, kind="BAD", weight=1.0)
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
        conn.execute("UPDATE comments SET status=? WHERE id=?", (STATUS_DELETED, comment_id))
        conn.commit()
    return RedirectResponse(url="/moderation", status_code=303)
