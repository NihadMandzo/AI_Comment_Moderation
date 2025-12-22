from __future__ import annotations

from typing import Dict, Protocol


class ITextClassifier(Protocol):
    """Black-box ML component.

    The rest of the agent treats this as a scorer over labels.
    """

    def predict_scores(self, text_raw: str, text_norm: str) -> Dict[str, float]:
        ...

    def update_online(self, text_norm: str, selected_labels: list[str], *, kind: str, weight: float) -> None:
        ...

    def rebuild_from_training_rows(self, rows: list[tuple[str, list[str], float, str]]) -> int:
        """Rebuild model from (kind, labels, weight/conf, text_norm). Returns used rows."""
        ...

    def dumps_active_model(self) -> bytes:
        ...

    def loads_active_model(self, blob: bytes) -> None:
        ...
