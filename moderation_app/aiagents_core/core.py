from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, Protocol, TypeVar

TPercept = TypeVar("TPercept")
TAction = TypeVar("TAction")
TResult = TypeVar("TResult")
TExperience = TypeVar("TExperience")


class IPerceptionSource(Protocol[TPercept]):
    @abstractmethod
    def sense(self) -> Optional[TPercept]:
        """Read one piece of 'world state' (or return None if no work)."""


class IPolicy(Protocol[TPercept, TAction]):
    @abstractmethod
    def decide(self, percept: TPercept) -> TAction:
        """Decide what to do, based on one percept."""


class IActuator(Protocol[TAction, TResult]):
    @abstractmethod
    def act(self, action: TAction) -> TResult:
        """Apply an action to the world and return a result."""


class ILearningComponent(Protocol[TExperience]):
    @abstractmethod
    def learn(self, experience: TExperience) -> None:
        """Update internal knowledge based on experience."""


@dataclass(frozen=True)
class TickNoWork:
    """Explicit 'no work' result (optional alternative to returning None)."""


class SoftwareAgent(Generic[TPercept, TAction, TResult, TExperience], ABC):
    """Base class for agents that operate in Sense→Think→Act→Learn ticks.

    A single tick/step should be small and (as much as possible) idempotent.
    Hosts (web/console/workers) must NOT implement business rules; they just call tick.
    """

    @abstractmethod
    def tick(self) -> Optional[TResult]:
        """Perform one agent tick. Return None when there's no work."""


class AsyncSoftwareAgent(Generic[TPercept, TAction, TResult, TExperience], ABC):
    """Async variant for I/O heavy agents."""

    @abstractmethod
    async def tick_async(self, *, ct=None) -> Optional[TResult]:
        """Perform one agent tick. Return None when there's no work."""
