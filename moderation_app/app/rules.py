from __future__ import annotations

import re

from .normalize import normalize_text


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Lightweight BCS profanity heuristic.
# Purpose: catch common swear-word stems in Bosnian/Croatian/Serbian text.
# This is intentionally incomplete and should be treated as a heuristic signal.
_PROFANITY_RE = re.compile(
    r"\b("
    r"jeb\w*|jbt|jebote|"
    r"kur\w*|"
    r"p(i|í|i\u0301)?c\w*|pič\w*|pizd\w*|"
    r"sran\w*|govn\w*|"
    r"mater\w*"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)

# Direct personal insults (abuse), kept separate from profanity.
_ABUSE_RE = re.compile(
    r"\b("
    r"idiot\w*|kreten\w*|budal\w*|deb(il|ila|ilu)\w*|"
    r"glup\w*|retard\w*|imbecil\w*|moron\w*"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)

# Threat rules are split into violent vs non-violent so actions can differ.
_VIOLENT_THREAT_RE = re.compile(
    r"\b("
    r"ub(i|í|i\u0301)?(j|cu|ću|ces|ćeš|cete|ćete)?\w*|"  # ubij, ubiću...
    r"zakla\w*|zaklat\w*|"  # zaklat/zaklaću
    r"prebi\w*|razbi\w*|"  # prebiću, razbiću
    r"spali\w*|"  # spalit ću
    r"silova\w*|"  # silovat
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
    """Heuristic spam score in [0, 1]."""
    t = normalize_text(text)
    if not t:
        return 0.0

    score = 0.0

    # Links are a strong signal
    if _URL_RE.search(t):
        score = max(score, 0.75)

    # Excessive repetition (same word many times)
    tokens = re.findall(r"\w+", t.lower(), flags=re.UNICODE)
    if tokens:
        unique_ratio = len(set(tokens)) / max(1, len(tokens))
        if len(tokens) >= 8 and unique_ratio < 0.35:
            score = max(score, 0.7)

    # Too many identical characters in a row (e.g. !!!!!, aaaaa)
    if re.search(r"(.)\1\1\1+", t):
        score = max(score, 0.5)

    # Too many non-word symbols
    non_word = len(re.findall(r"[^\w\s]", t, flags=re.UNICODE))
    if len(t) >= 20 and (non_word / len(t)) > 0.25:
        score = max(score, 0.5)

    return float(min(1.0, score))


def profanity_score(text: str) -> float:
    """Heuristic profanity score in [0, 1] for BCS languages."""
    t = normalize_text(text)
    if not t:
        return 0.0

    matches = list(_PROFANITY_RE.finditer(t))
    if not matches:
        return 0.0

    # One match is already strong; multiple matches saturate.
    if len(matches) == 1:
        return 0.85
    if len(matches) == 2:
        return 0.93
    return 0.98


def abuse_score(text: str) -> float:
    """Heuristic abuse (personal insult) score in [0, 1] for BCS languages."""
    t = normalize_text(text)
    if not t:
        return 0.0
    if _ABUSE_RE.search(t):
        return 0.78
    return 0.0


def violent_threat_score(text: str) -> float:
    """Heuristic violent threat score in [0, 1] for BCS languages."""
    t = normalize_text(text)
    if not t:
        return 0.0
    if _VIOLENT_THREAT_RE.search(t):
        return 0.95
    return 0.0


def nonviolent_threat_score(text: str) -> float:
    """Heuristic non-violent threat score in [0, 1] for BCS languages."""
    t = normalize_text(text)
    if not t:
        return 0.0
    if _NONVIOLENT_THREAT_RE.search(t):
        return 0.65
    return 0.0


def threat_score(text: str) -> float:
    """Backwards-compatible: max of violent + non-violent threats."""
    return float(max(violent_threat_score(text), nonviolent_threat_score(text)))
