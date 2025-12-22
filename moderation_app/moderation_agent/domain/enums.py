from __future__ import annotations

from enum import Enum


class MessageStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"

    APPROVED = "APPROVED"
    PENDING_REVIEW = "PENDING_REVIEW"
    DELETED = "DELETED"


class Decision(str, Enum):
    ALLOW = "ALLOW"
    PENDING_REVIEW = "PENDING_REVIEW"
    DELETE = "DELETE"


class ReviewKind(str, Enum):
    GOOD = "GOOD"
    BAD = "BAD"
