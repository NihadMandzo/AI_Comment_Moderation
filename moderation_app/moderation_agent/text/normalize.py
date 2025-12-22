from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_LEET_MAP = str.maketrans(
    {
        "@": "a",
        "4": "a",
        "$": "s",
        "5": "s",
        "0": "o",
        "1": "i",
        "3": "e",
        "7": "t",
        "€": "e",
    }
)

_DIACRITICS_MAP = str.maketrans(
    {
        "č": "c",
        "ć": "c",
        "ž": "z",
        "š": "s",
        "đ": "d",
        "Č": "c",
        "Ć": "c",
        "Ž": "z",
        "Š": "s",
        "Đ": "d",
    }
)


_WS_RE = re.compile(r"\s+", re.UNICODE)
_REPEAT_RE = re.compile(r"([a-z])\1{2,}", re.IGNORECASE)


def _strip_unicode_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _join_spelled_out_words(text: str) -> str:
    parts = re.split(r"[^a-z0-9]+", text)
    out: list[str] = []
    run: list[str] = []

    def flush_run() -> None:
        nonlocal run
        if not run:
            return
        if len(run) >= 3:
            out.append("".join(run))
        else:
            out.extend(run)
        run = []

    for p in parts:
        if not p:
            continue
        if len(p) == 1:
            run.append(p)
        else:
            flush_run()
            out.append(p)
    flush_run()

    return " ".join(out)


@dataclass(frozen=True)
class TwoChannelText:
    raw: str
    norm: str


def normalize_text(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    t = t.translate(_LEET_MAP)
    t = t.translate(_DIACRITICS_MAP)
    t = _strip_unicode_diacritics(t)
    t = t.lower()
    t = _join_spelled_out_words(t)
    t = _REPEAT_RE.sub(r"\1\1", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def two_channel(text: str) -> TwoChannelText:
    raw = (text or "").strip()
    return TwoChannelText(raw=raw, norm=normalize_text(raw))
