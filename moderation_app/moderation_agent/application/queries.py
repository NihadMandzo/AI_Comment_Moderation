from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from moderation_agent.infrastructure import connect


class QueryService:
    def get_approved_comment_text_for_reporting(self, *, message_id: int) -> Optional[str]:
        with connect() as conn:
            row = conn.execute(
                "SELECT text FROM comments WHERE id=? AND status='APPROVED'",
                (int(message_id),),
            ).fetchone()
            if not row:
                return None
            return str(row["text"] or "")

    def get_comment_text(self, *, message_id: int) -> Optional[str]:
        with connect() as conn:
            row = conn.execute("SELECT text FROM comments WHERE id=?", (int(message_id),)).fetchone()
            if not row:
                return None
            return str(row["text"] or "")

    def list_approved(self, *, limit: int = 200) -> List[Dict[str, Any]]:
        with connect() as conn:
            rows = conn.execute(
                "SELECT id, text, created_at, final_labels_json, final_max_label, final_max_score, report_count "
                "FROM comments WHERE status='APPROVED' ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            final = json.loads(r["final_labels_json"]) if r["final_labels_json"] else {}
            out.append(
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
        return out

    def list_all(self, *, limit: int = 300) -> List[Dict[str, Any]]:
        with connect() as conn:
            rows = conn.execute(
                "SELECT id, text, created_at, status, final_labels_json, final_max_label, final_max_score, report_count "
                "FROM comments ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            final = json.loads(r["final_labels_json"]) if r["final_labels_json"] else {}
            out.append(
                {
                    "id": r["id"],
                    "text": r["text"],
                    "created_at": r["created_at"],
                    "status": r["status"],
                    "final": final,
                    "final_max_label": r["final_max_label"],
                    "final_max_score": r["final_max_score"],
                    "report_count": r["report_count"],
                }
            )
        return out
