from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from moderation_agent.application import (
    QueryService,
    QueueService,
    ReviewService,
    ScoringService,
    TrainingService,
)
from moderation_agent.domain import MessageStatus, ModerationRules, ReviewKind
from moderation_agent.infrastructure import ModelStore, SettingsStore, init_db
from moderation_agent.ml import LABELS
from moderation_agent.ml.moderation_classifier import ModerationClassifier
from moderation_agent.runners import RetrainAgentRunner, ScoringAgentRunner


# Host-only: simple in-memory rate limit for reports per IP
REPORTS_PER_MINUTE = 10

# Host-only: server-side salt file to hash reporters without storing raw IP/UA.
SALT_ENV_VAR = "MODERATION_SALT"


app = FastAPI(title="Moderation App")
APP_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


_rate_lock = asyncio.Lock()
_rate_state: Dict[str, List[float]] = {}

_stop_event: Optional[asyncio.Event] = None
_worker_tasks: List[asyncio.Task] = []


# --- Shared agent wiring (brain lives outside web) ---

_classifier = ModerationClassifier()
_model_store = ModelStore()
_settings_store = SettingsStore()
_rules = ModerationRules(labels=tuple(LABELS))

_queue = QueueService()
_query = QueryService()
_scoring = ScoringService(classifier=_classifier, settings_store=_settings_store, rules=_rules)
_review = ReviewService(classifier=_classifier, settings_store=_settings_store, model_store=_model_store)
_training = TrainingService(classifier=_classifier, settings_store=_settings_store, model_store=_model_store)

_scoring_runner = ScoringAgentRunner(queue=_queue, scoring=_scoring)
_retrain_runner = RetrainAgentRunner(settings_store=_settings_store, training=_training)


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


async def rate_limit_ok(ip: str) -> bool:
    if not ip:
        return True
    cutoff = time.time() - 60
    async with _rate_lock:
        times = [t for t in _rate_state.get(ip, []) if t >= cutoff]
        if len(times) >= REPORTS_PER_MINUTE:
            _rate_state[ip] = times
            return False
        times.append(time.time())
        _rate_state[ip] = times
        return True


async def _scoring_worker() -> None:
    assert _stop_event is not None
    while not _stop_event.is_set():
        res = await _scoring_runner.tick_async()
        if res is None:
            await asyncio.sleep(0.2)
        else:
            await asyncio.sleep(0)


async def _retrain_worker() -> None:
    assert _stop_event is not None
    while not _stop_event.is_set():
        await _retrain_runner.tick_async()
        await asyncio.sleep(2.0)


@app.on_event("startup")
async def _startup() -> None:
    init_db()

    # Load active online model state
    blob = _model_store.load_active_model_blob()
    _classifier.loads_active_model(blob)

    global _stop_event
    _stop_event = asyncio.Event()

    _worker_tasks.append(asyncio.create_task(_scoring_worker()))
    _worker_tasks.append(asyncio.create_task(_retrain_worker()))


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _stop_event is not None:
        _stop_event.set()
    for t in list(_worker_tasks):
        t.cancel()


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    comments = _query.list_approved(limit=200)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "labels": LABELS,
            "comments": comments,
        },
    )


@app.post("/comments")
async def create_comment(text: str = Form(...)) -> RedirectResponse:
    t = (text or "").strip()
    if not t:
        return RedirectResponse(url="/", status_code=303)

    _queue.enqueue(t)
    # Host orchestration is OK: run one tick to reduce latency.
    await _scoring_runner.tick_async()

    return RedirectResponse(url="/", status_code=303)


@app.post("/report/{comment_id}")
async def report_comment(comment_id: int, request: Request) -> JSONResponse:
    ip = request.client.host if request.client else ""
    if not await rate_limit_ok(ip):
        return JSONResponse({"ok": False, "error": "rate_limited"}, status_code=429)

    payload = await request.json()
    labels = payload.get("labels") or []
    note = (payload.get("note") or "").strip()

    selected = [l for l in labels if l in LABELS]

    text = _query.get_approved_comment_text_for_reporting(message_id=comment_id)
    if text is None:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)

    ua = request.headers.get("user-agent") or ""
    rep_hash = _reporter_hash(ip, ua)

    _review.insert_report(
        message_id=comment_id,
        kind=ReviewKind.BAD,
        labels=selected,
        reporter_hash=rep_hash,
        confidence=0.5,
        used_for_training=False,
        note=note,
    )

    # Weak labels: train only on consensus (shared logic)
    trained = _review.maybe_train_from_consensus(
        message_id=comment_id,
        text=text,
        kind=ReviewKind.BAD,
        labels=selected,
    )

    return JSONResponse({"ok": True, "trained": bool(trained)})


@app.get("/moderation", response_class=HTMLResponse)
def moderation(request: Request) -> HTMLResponse:
    rows = _query.list_all(limit=300)

    grouped: Dict[str, List[Dict]] = {
        MessageStatus.APPROVED.value: [],
        MessageStatus.PENDING_REVIEW.value: [],
        MessageStatus.DELETED.value: [],
    }

    all_items: List[Dict] = []
    for item in rows:
        # Display legacy HIDDEN as PENDING_REVIEW.
        if item.get("status") == "HIDDEN":
            item["status"] = MessageStatus.PENDING_REVIEW.value
        all_items.append(item)
        if item["status"] in grouped:
            grouped[item["status"]].append(item)

    return templates.TemplateResponse(
        "moderation.html",
        {
            "request": request,
            "labels": LABELS,
            "all_comments": all_items,
            "approved_comments": grouped[MessageStatus.APPROVED.value],
            "pending_comments": grouped[MessageStatus.PENDING_REVIEW.value],
            "deleted_comments": grouped[MessageStatus.DELETED.value],
        },
    )


@app.post("/moderation/{comment_id}/approve")
def approve(comment_id: int) -> RedirectResponse:
    _queue.set_status(comment_id, MessageStatus.APPROVED)
    return RedirectResponse(url="/moderation", status_code=303)


@app.post("/moderation/{comment_id}/good")
def mark_good(comment_id: int) -> RedirectResponse:
    # Moderator says: this comment is OK.
    text = _query.get_comment_text(message_id=comment_id)
    if text is None:
        return RedirectResponse(url="/moderation", status_code=303)

    _queue.set_status(comment_id, MessageStatus.APPROVED)
    _review.moderator_mark_good(message_id=comment_id, text=text)

    return RedirectResponse(url="/moderation", status_code=303)


@app.post("/moderation/{comment_id}/bad")
async def mark_bad(comment_id: int, request: Request) -> JSONResponse:
    # Moderator says: this comment is bad, with multi-labels.
    payload = await request.json()
    labels = payload.get("labels") or []
    note = (payload.get("note") or "").strip()
    selected = [l for l in labels if l in LABELS]

    text = _query.get_comment_text(message_id=comment_id)
    if text is None:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)

    _queue.set_status(comment_id, MessageStatus.DELETED)
    _review.moderator_mark_bad(message_id=comment_id, text=text, labels=selected, note=note)

    return JSONResponse({"ok": True})


@app.post("/moderation/train")
def train_button(request: Request):
    version_id = _training.train_model(activate=True)
    rescored = _scoring.rescore_all()

    accept = (request.headers.get("accept") or "").lower()
    wants_html = "text/html" in accept
    if wants_html:
        return RedirectResponse(url="/moderation", status_code=303)

    return JSONResponse({"ok": True, "model_version_id": version_id, "comments_rescored": rescored})


@app.post("/moderation/{comment_id}/delete")
def delete(comment_id: int) -> RedirectResponse:
    _queue.set_status(comment_id, MessageStatus.DELETED)
    return RedirectResponse(url="/moderation", status_code=303)
