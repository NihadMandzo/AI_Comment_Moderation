from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .enums import Decision, MessageStatus
from .entities import SystemSettings


@dataclass(frozen=True)
class ModerationRules:
    """Domain rules: convert scores + settings into a decision/status.

    This MUST NOT be duplicated in the web host.
    """

    labels: tuple[str, ...]

    def decide(
        self,
        scores: Dict[str, float],
        *,
        settings: SystemSettings,
        violent_threat_score: float,
    ) -> tuple[Decision, MessageStatus]:
        if float(violent_threat_score) >= float(settings.violent_threat_delete_min):
            return Decision.DELETE, MessageStatus.DELETED

        # Delete zone
        for label in self.labels:
            s = float(scores.get(label, 0.0))
            delete_min = float(settings.delete_min_by_label.get(label, settings.delete_min_default))
            if s >= delete_min:
                return Decision.DELETE, MessageStatus.DELETED

        # Pending zone
        for label in self.labels:
            s = float(scores.get(label, 0.0))
            pending_min = float(settings.pending_min_by_label.get(label, settings.pending_min_default))
            if s >= pending_min:
                return Decision.PENDING_REVIEW, MessageStatus.PENDING_REVIEW

        return Decision.ALLOW, MessageStatus.APPROVED
