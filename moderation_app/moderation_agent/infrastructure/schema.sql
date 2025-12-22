PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS comments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL,
  baseline_labels_json TEXT NOT NULL,
  learned_labels_json TEXT NOT NULL,
  final_labels_json TEXT NOT NULL,
  final_max_label TEXT NOT NULL,
  final_max_score REAL NOT NULL,
  report_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS reports (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  comment_id INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'BAD',
  labels_json TEXT NOT NULL,
  reporter_hash TEXT,
  confidence REAL NOT NULL DEFAULT 0.3,
  used_for_training INTEGER NOT NULL DEFAULT 0,
  note TEXT,
  FOREIGN KEY(comment_id) REFERENCES comments(id)
);

CREATE TABLE IF NOT EXISTS training_state (
  id INTEGER PRIMARY KEY CHECK (id=1),
  model_blob BLOB NOT NULL
);

INSERT OR IGNORE INTO training_state(id, model_blob) VALUES (1, X'00');
