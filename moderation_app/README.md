# Comment Moderation (BCS) — FastAPI + SQLite + Online Learning

Small uni-project web app:
- Users submit comments
- Server runs multi-label moderation (`hate_speech`, `abuse`, `threat`, `profanity`, `spam`)
- Clean vanilla HTML/CSS/JS UI (no frontend frameworks)
- Users can report approved comments with multi-select labels
- Online learner updates from reports and improves future predictions

## Tech
- Python 3.11+
- FastAPI + Uvicorn
- Jinja2 templates
- SQLite (built-in)
- scikit-learn (HashingVectorizer + one SGDClassifier per label)
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
uvicorn app.main:app --reload
```

If port 8000 is blocked on Windows, use:
```powershell
python -m uvicorn --app-dir . app.main:app --reload --host 127.0.0.1 --port 8010
```

If you see `ModuleNotFoundError: No module named 'app'` with `--reload` on Windows, use the included runner (it sets `PYTHONPATH` for the reload subprocess):
```powershell
./run.ps1
```

## Run (Linux/Mac)
```bash
cd moderation_app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Open
- Public: http://127.0.0.1:8000/ (or http://127.0.0.1:8010/ if you used the alternate port)
- Moderation: http://127.0.0.1:8000/moderation (or http://127.0.0.1:8010/moderation)

## Seed + Train (optional)
To quickly seed Bosnian good/bad examples and train the online model:
```powershell
cd moderation_app
python scripts\seed_bosnian_dataset.py
```

You can also train from the UI:
- Open the moderation page and click **Train model** (rebuilds the online learner from all stored reports, then rescoring existing comments so old false-positives get corrected).

## Deleting vs training
- **Bad (Delete + Train)** removes the comment from the public list (marks it DELETED) but keeps it in SQLite and uses it for training.
- Training always uses any comment (approved/hidden/deleted) that has reports.

## Labels
- `hate_speech`
- `abuse`
- `threat`
- `profanity`
- `spam`

## Database
SQLite DB is created automatically on first run.
Tables:
- `comments`
- `reports`
- `training_state` (single row, persists the online model)
