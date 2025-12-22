from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from aiagents_core.core import AsyncSoftwareAgent

from moderation_agent.application import QueueService, ScoringService


@dataclass(frozen=True)
class ScoringTickResult:
    message_id: int
    decision: str
    new_status: str
    top_label: str
    top_score: float
    model_version_id: int | None


class ScoringAgentRunner(AsyncSoftwareAgent[object, object, ScoringTickResult, object]):
    """Sense→Think→Act tick that processes exactly one queued message."""

    def __init__(self, *, queue: QueueService, scoring: ScoringService) -> None:
        self._queue = queue
        self._scoring = scoring

    async def tick_async(self, *, ct=None) -> Optional[ScoringTickResult]:
        # SENSE
        item = self._queue.dequeue_next()
        if item is None:
            return None

        # THINK
        prediction = self._scoring.score_and_decide(item=item)

        # ACT
        self._scoring.persist_prediction(prediction=prediction)

        return ScoringTickResult(
            message_id=prediction.message_id,
            decision=prediction.decision.value,
            new_status=prediction.new_status.value,
            top_label=prediction.top_label,
            top_score=float(prediction.top_score),
            model_version_id=prediction.model_version_id,
        )
