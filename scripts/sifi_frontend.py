"""Sinhala grapheme-cluster and phonological feature extraction for SiFi-TTS.

This is the deterministic, lightweight frontend used before the trainable fusion
adapter. It deliberately does not guess pronunciations for English words.
"""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass


VOWELS = {
    "අ": ("a", "short"), "ආ": ("a", "long"),
    "ඇ": ("ae", "short"), "ඈ": ("ae", "long"),
    "ඉ": ("i", "short"), "ඊ": ("i", "long"),
    "උ": ("u", "short"), "ඌ": ("u", "long"),
    "ඍ": ("r", "short"), "ඎ": ("r", "long"),
    "එ": ("e", "short"), "ඒ": ("e", "long"), "ඓ": ("ai", "long"),
    "ඔ": ("o", "short"), "ඕ": ("o", "long"), "ඖ": ("au", "long"),
}

VOWEL_SIGNS = {
    "ා": ("a", "long"), "ැ": ("ae", "short"), "ෑ": ("ae", "long"),
    "ි": ("i", "short"), "ී": ("i", "long"),
    "ු": ("u", "short"), "ූ": ("u", "long"),
    "ෘ": ("r", "short"), "ෲ": ("r", "long"),
    "ෙ": ("e", "short"), "ේ": ("e", "long"), "ෛ": ("ai", "long"),
    "ො": ("o", "short"), "ෝ": ("o", "long"), "ෞ": ("au", "long"),
}

# base -> (place, manner, voiced, aspirated, prenasalized)
CONSONANTS = {
    "ක": ("velar", "stop", False, False, False),
    "ඛ": ("velar", "stop", False, True, False),
    "ග": ("velar", "stop", True, False, False),
    "ඝ": ("velar", "stop", True, True, False),
    "ඟ": ("velar", "stop", True, False, True),
    "ච": ("palatal", "affricate", False, False, False),
    "ඡ": ("palatal", "affricate", False, True, False),
    "ජ": ("palatal", "affricate", True, False, False),
    "ඣ": ("palatal", "affricate", True, True, False),
    "ඦ": ("palatal", "affricate", True, False, True),
    "ට": ("retroflex", "stop", False, False, False),
    "ඨ": ("retroflex", "stop", False, True, False),
    "ඩ": ("retroflex", "stop", True, False, False),
    "ඪ": ("retroflex", "stop", True, True, False),
    "ඬ": ("retroflex", "stop", True, False, True),
    "ත": ("dental", "stop", False, False, False),
    "ථ": ("dental", "stop", False, True, False),
    "ද": ("dental", "stop", True, False, False),
    "ධ": ("dental", "stop", True, True, False),
    "ඳ": ("dental", "stop", True, False, True),
    "ප": ("labial", "stop", False, False, False),
    "ඵ": ("labial", "stop", False, True, False),
    "බ": ("labial", "stop", True, False, False),
    "භ": ("labial", "stop", True, True, False),
    "ඹ": ("labial", "stop", True, False, True),
    "ඞ": ("velar", "nasal", True, False, False),
    "ඤ": ("palatal", "nasal", True, False, False),
    "ණ": ("retroflex", "nasal", True, False, False),
    "න": ("dental", "nasal", True, False, False),
    "ම": ("labial", "nasal", True, False, False),
    "ය": ("palatal", "approximant", True, False, False),
    "ර": ("alveolar", "trill", True, False, False),
    "ල": ("alveolar", "lateral", True, False, False),
    "ව": ("labial", "approximant", True, False, False),
    "ශ": ("palatal", "fricative", False, False, False),
    "ෂ": ("retroflex", "fricative", False, False, False),
    "ස": ("dental", "fricative", False, False, False),
    "හ": ("glottal", "fricative", False, False, False),
    "ළ": ("retroflex", "lateral", True, False, False),
    "ෆ": ("labial", "fricative", False, False, False),
}

VIRAMA = "්"
ZWJ = "\u200d"


@dataclass(frozen=True)
class GraphemeFeatures:
    grapheme: str
    kind: str
    base: str = ""
    place: str = ""
    manner: str = ""
    voiced: bool = False
    aspirated: bool = False
    prenasalized: bool = False
    vowel: str = ""
    vowel_length: str = ""
    inherent_vowel: bool = False
    conjunct: bool = False
    code_switch: bool = False


def grapheme_clusters(text: str) -> list[str]:
    """Segment enough of Unicode Sinhala for deterministic TTS tokenization."""
    text = unicodedata.normalize("NFC", text)
    clusters: list[str] = []
    current = ""
    join_next = False
    for char in text:
        combining = bool(unicodedata.combining(char)) or char in VOWEL_SIGNS or char in {VIRAMA, ZWJ, "ං", "ඃ"}
        if not current or combining or join_next:
            current += char
        else:
            clusters.append(current)
            current = char
        join_next = char in {VIRAMA, ZWJ}
    if current:
        clusters.append(current)
    return clusters


def extract_features(text: str) -> list[GraphemeFeatures]:
    output = []
    for cluster in grapheme_clusters(text):
        bases = [char for char in cluster if char in CONSONANTS]
        vowel_chars = [char for char in cluster if char in VOWEL_SIGNS]
        if bases:
            base = bases[0]
            place, manner, voiced, aspirated, prenasalized = CONSONANTS[base]
            killed = cluster.endswith(VIRAMA) or cluster.endswith(VIRAMA + ZWJ)
            vowel, length = VOWEL_SIGNS[vowel_chars[-1]] if vowel_chars else (("", "") if killed else ("a", "short"))
            output.append(GraphemeFeatures(
                grapheme=cluster, kind="consonant", base=base, place=place,
                manner=manner, voiced=voiced, aspirated=aspirated,
                prenasalized=prenasalized, vowel=vowel, vowel_length=length,
                inherent_vowel=not vowel_chars and not killed,
                conjunct=len(bases) > 1 or VIRAMA in cluster,
            ))
        elif cluster[0] in VOWELS:
            vowel, length = VOWELS[cluster[0]]
            output.append(GraphemeFeatures(
                grapheme=cluster, kind="vowel", base=cluster[0], vowel=vowel,
                vowel_length=length,
            ))
        else:
            code_switch = any("LATIN" in unicodedata.name(c, "") for c in cluster)
            kind = "space" if cluster.isspace() else "punctuation" if all(unicodedata.category(c).startswith("P") for c in cluster) else "other"
            output.append(GraphemeFeatures(
                grapheme=cluster, kind=kind, code_switch=code_switch,
            ))
    return output


def feature_dicts(text: str) -> list[dict]:
    return [asdict(item) for item in extract_features(text)]
