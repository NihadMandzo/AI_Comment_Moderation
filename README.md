# Comment Moderation (BCS) — FastAPI + SQLite + Online Learning

Small uni-project web app:
- Users submit comments
- Server runs multi-label moderation (`hate_speech`, `abuse`, `threat`, `profanity`, `spam`)
- Clean vanilla HTML/CSS/JS UI (no frontend frameworks)
- 3-zone decisioning: `APPROVED` / `PENDING_REVIEW` / `DELETED`
- Users can report approved comments with multi-select labels
- Online learner improves over time using **gold** (moderator) labels + **consensus** user reports
- Bosnian-friendly text normalization to catch obfuscations (e.g. `k u r a c`, `k.u.r.a.c`)

## Tech
- Python 3.11+
- FastAPI + Uvicorn
- Jinja2 templates
- SQLite (built-in)
- scikit-learn (HashingVectorizer char n-grams + one SGDClassifier per label, weighted `partial_fit`)
- Transformers (Hugging Face models)

## First run notes
- The first run will download Hugging Face models and can take a while.
- If model download fails (offline), the app still runs and falls back to safe defaults (all baseline scores = 0) while logging a warning.
- If `pip install -r requirements.txt` fails on `torch`, install CPU-only torch first, then re-run requirements:
	- Windows (PowerShell): `pip install --index-url https://download.pytorch.org/whl/cpu torch`

## Run (Windows PowerShell)
```powershell
cd moderation_app
py -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn moderation_web.main:app --reload
```

If port 8000 is blocked on Windows, use:
```powershell
python -m uvicorn --app-dir . moderation_web.main:app --reload --host 127.0.0.1 --port 8010
```

If you see `ModuleNotFoundError` with `--reload` on Windows, use the included runner (it sets `PYTHONPATH` for the reload subprocess):
```powershell
./run.ps1
```

## Run (Linux/Mac)
```bash
cd moderation_app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn moderation_web.main:app --reload
```

## Open
- Public: http://127.0.0.1:8000/ (or http://127.0.0.1:8010/ if you used the alternate port)
- Moderation: http://127.0.0.1:8000/moderation (or http://127.0.0.1:8010/moderation)

## Seed + Train (optional)
To quickly seed Bosnian good/bad examples and train the online model:
```powershell
cd moderation_app
python scripts\datasets\seed_bosnian_dataset.py
```

You can also train from the UI:
- Open the moderation page and click **Train model** (rebuilds the online learner from all stored reports, then rescoring existing comments so old false-positives get corrected).

## Statuses (3-zone)
- **APPROVED**: publicly visible (ALLOW zone)
- **PENDING_REVIEW**: hidden from public + queued for moderator review (abstain zone)
- **DELETED**: removed from public (DELETE zone)

## Reports, poisoning resistance, and training quality
- **Moderator actions** are treated as **gold labels** and are always used for training.
- **User reports** are treated as **weak labels** and are used for training only when there is consensus (≥2 independent reporters).
- Reports store `reporter_hash` (hash of IP + User-Agent + server salt, rotated monthly), `confidence`, and `used_for_training`.

Privacy note: the app does **not** store raw IP addresses in the DB for reporter identity.

## Labels
- `hate_speech`
- `abuse`
- `threat`
- `profanity`
- `spam`

## Database
SQLite DB is created automatically on first run.
DB file location:
- `moderation_app/data/moderation.db` (and SQLite `-wal`/`-shm` files live next to it)
Tables:
- `comments`
- `reports`
- `training_state` (single row, persists the online model)

## Ops knobs
- `MODERATION_SALT`: optional env var used to hash reporters. If unset, the server creates `moderation_web/_server_salt.txt` on first run.


## Notes
- For correctness and consistent online learning, run a **single** Uvicorn worker (do not scale workers) unless you also implement shared training_state coordination.
