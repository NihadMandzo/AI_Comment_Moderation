from __future__ import annotations

from .db import connect


class ModelStore:
    """Persistence for the agent model state.

    For this project we keep the active model blob in SQLite (training_state.model_blob).
    """

    def load_active_model_blob(self) -> bytes:
        with connect() as conn:
            row = conn.execute("SELECT model_blob FROM training_state WHERE id=1").fetchone()
            if not row:
                return b"\x00"
            return bytes(row["model_blob"]) if row["model_blob"] is not None else b"\x00"

    def save_active_model_blob(self, blob: bytes) -> None:
        with connect() as conn:
            conn.execute("UPDATE training_state SET model_blob=? WHERE id=1", (blob,))
            conn.commit()

    def set_active_model_version(self, version_id: int | None) -> None:
        with connect() as conn:
            conn.execute(
                "UPDATE training_state SET active_model_version_id=? WHERE id=1",
                (int(version_id) if version_id is not None else None,),
            )
            conn.commit()
