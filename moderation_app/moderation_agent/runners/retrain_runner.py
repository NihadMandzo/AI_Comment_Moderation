from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from aiagents_core.core import AsyncSoftwareAgent

from moderation_agent.infrastructure import SettingsStore
from moderation_agent.application import TrainingService


@dataclass(frozen=True)
class RetrainTickResult:
    did_retrain: bool
    new_model_version_id: int | None
    new_gold_since_last_train: int
    retrain_gold_threshold: int


class RetrainAgentRunner(AsyncSoftwareAgent[object, object, RetrainTickResult, object]):
    """Sense→Think→Act→Learn tick that decides if retraining should happen."""

    def __init__(self, *, settings_store: SettingsStore, training: TrainingService) -> None:
        self._settings_store = settings_store
        self._training = training

    async def tick_async(self, *, ct=None) -> Optional[RetrainTickResult]:
        # SENSE
        settings = self._settings_store.load()

        # THINK
        should = bool(settings.enabled) and int(settings.new_gold_since_last_train) >= int(settings.retrain_gold_threshold)
        if not should:
            return RetrainTickResult(
                did_retrain=False,
                new_model_version_id=None,
                new_gold_since_last_train=int(settings.new_gold_since_last_train),
                retrain_gold_threshold=int(settings.retrain_gold_threshold),
            )

        # ACT
        version_id = self._training.train_model(activate=True)

        # LEARN (handled by TrainingService via counter reset)
        after = self._settings_store.load()

        return RetrainTickResult(
            did_retrain=True,
            new_model_version_id=version_id,
            new_gold_since_last_train=int(after.new_gold_since_last_train),
            retrain_gold_threshold=int(after.retrain_gold_threshold),
        )
