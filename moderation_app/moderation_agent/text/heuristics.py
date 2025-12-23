from __future__ import annotations

import re

from .normalize import normalize_text


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

_PROFANITY_RE = re.compile(
    r"\b("
    r"jeb\w*|jbt|jebote|"
    r"kur\w*|"
    r"p(i|í|i\u0301)?c\w*|pič\w*|pizd\w*|"
    r"sran\w*|govn\w*"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)

_ABUSE_RE = re.compile(
    r"\b("
    r"idiot\w*|kreten\w*|budal\w*|deb(il|ila|ilu)\w*|"
    r"glup\w*|retard\w*|imbecil\w*|moron\w*"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)

_VIOLENT_THREAT_RE = re.compile(
    r"\b("
    r"ub(i|í|i\u0301)?(j|cu|ću|ces|ćeš|cete|ćete)?\w*|"
    r"zakla\w*|zaklat\w*|"
    r"prebi\w*|razbi\w*|"
    r"spali\w*|"
    r"silova\w*|"
    r"bomba\w*|eksploz\w*|"
    r"puca\w*|upuca\w*|"
    r"noz\w*|pištolj\w*|pistolj\w*"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)

_NONVIOLENT_THREAT_RE = re.compile(
    r"\b("
    r"prijav\w*\s+te|prijavit\w*\s+te|"
    r"tuz\w*\s+te|tuž\w*\s+te|"
    r"pozva\w*\s+policij\w*|zva\w*\s+policij\w*|"
    r"banov\w*|banuj\w*|blokir\w*|"
    r"otpust\w*|"
    r"saznac\w*\s+ces|saznac\w*\s+ćeš"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)


def spam_score(text: str) -> float:
    raw = (text or "").strip()
    t = normalize_text(raw)
    if not raw and not t:
        return 0.0

    score = 0.0
    # IMPORTANT: detect URLs on raw text, not normalized text.
    # normalize_text() removes punctuation ('.', '/', ':') which breaks URL patterns.
    if _URL_RE.search(raw):
        score = max(score, 0.75)

    tokens = re.findall(r"\w+", t.lower(), flags=re.UNICODE)
    if tokens:
        unique_ratio = len(set(tokens)) / max(1, len(tokens))
        if len(tokens) >= 8 and unique_ratio < 0.35:
            score = max(score, 0.7)

    if re.search(r"(.)\1\1\1+", t):
        score = max(score, 0.5)

    non_word = len(re.findall(r"[^\w\s]", t, flags=re.UNICODE))
    if len(t) >= 20 and (non_word / len(t)) > 0.25:
        score = max(score, 0.5)

    return float(min(1.0, score))


def profanity_score(text: str) -> float:
    t = normalize_text(text)
    if not t:
        return 0.0

    matches = list(_PROFANITY_RE.finditer(t))
    if not matches:
        return 0.0
    if len(matches) == 1:
        return 0.85
    if len(matches) == 2:
        return 0.93
    return 0.98


def abuse_score(text: str) -> float:
    t = normalize_text(text)
    if not t:
        return 0.0
    if _ABUSE_RE.search(t):
        return 0.78
    return 0.0


def violent_threat_score(text: str) -> float:
    t = normalize_text(text)
    if not t:
        return 0.0
    if _VIOLENT_THREAT_RE.search(t):
        return 0.95
    return 0.0


def nonviolent_threat_score(text: str) -> float:
    t = normalize_text(text)
    if not t:
        return 0.0
    if _NONVIOLENT_THREAT_RE.search(t):
        return 0.65
    return 0.0


def threat_score(text: str) -> float:
    return float(max(violent_threat_score(text), nonviolent_threat_score(text)))
