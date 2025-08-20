import random
import re
import unicodedata
from typing import Dict, List

# --- Basic transformations ---


# -----------------------------------------------------------------------------
# NFKD (Normalization Form Compatibility Decomposition):
# decomposes characters by compatibility. This means it replaces characters that
# have compatibility equivalents with their decomposed forms, even if it results
# in a loss of formatting information.
def drop_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


# -----------------------------------------------------------------------------
# NFKC (Normalization Form Compatibility Composition):
# This specific normalization form addresses "compatibility equivalences." It goes
# beyond canonical equivalences (which deal with combining characters and reordering)
# by also resolving compatibility differences. This means it can convert characters
# that are visually similar but have distinct semantic meanings into a common representation.
def nfkc_normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


# -----------------------------------------------------------------------------
# Randomize the case of a string given a certain percentage chance.
# EG Hello --> helLo
def case_randomize(s: str, p: float = 0.15) -> str:
    out = []
    for ch in s:
        if ch.isalpha() and random.random() < p:
            out.append(ch.upper() if ch.islower() else ch.lower())
        else:
            out.append(ch)
    return "".join(out)


# -----------------------------------------------------------------------------
# Upper case a string
def case_upper(s: str) -> str:
    return s.upper()


# -----------------------------------------------------------------------------
# Lower case a string
def case_lower(s: str) -> str:
    return s.lower()


# -----------------------------------------------------------------------------
# Capitalize a string
def case_capitalize(s: str) -> str:
    return s.capitalize()


# -----------------------------------------------------------------------------
# Titlize a string
def case_title(s: str) -> str:
    return s.title()


# -----------------------------------------------------------------------------
# Minimal QWERTY neighbors (extend as needed)
QWERTY_NEIGHBORS = {
    "a": "qwsz",
    "b": "vghn",
    "c": "xdfv",
    "d": "ersfcx",
    "e": "wsdr",
    "f": "rtgdvc",
    "g": "tyfhvb",
    "h": "yugjnb",
    "i": "ujko",
    "j": "uikhmn",
    "k": "ijolm,",
    "l": "kop;",
    "m": "njk",
    "n": "bhjm",
    "o": "iklp",
    "p": "ol;",
    "q": "was",
    "r": "edft",
    "s": "awedxz",
    "t": "rfgy",
    "u": "yhji",
    "v": "cfgb",
    "w": "qase",
    "x": "zsdc",
    "y": "tugh",
    "z": "asx",
}


# -----------------------------------------------------------------------------
# Simulates typos using a QWERY keyboard at a percentage chance.
# TODO: other keyboards?
def keyboard_subst(s: str, p: float = 0.08) -> str:
    out = []
    for ch in s:
        lower = ch.lower()
        if lower in QWERTY_NEIGHBORS and ch.isalpha() and random.random() < p:
            cand = random.choice(QWERTY_NEIGHBORS[lower])
            out.append(cand.upper() if ch.isupper() else cand)
        else:
            out.append(ch)
    return "".join(out)


# -----------------------------------------------------------------------------
# Possible OCR mistakes (extend as needed)
OCR_CONFUSIONS = [
    ("rn", "m"), ("cl", "d"), ("I", "l"), ("l", "1"), ("O", "0"), ("B", "8"),
    ("é", "e"), ("à", "a"), ("ü", "u"), ("ñ", "n"), ("c", "e"), ("t", "f"),
    (".", ",")
]


# -----------------------------------------------------------------------------
# Simulate OCR errors
def ocr_confuse(s: str, k: int = 1) -> str:
    for _ in range(max(0, k)):
        pat, rep = random.choice(OCR_CONFUSIONS)
        s = re.sub(pat, rep, s, count=1)
    return s


# -----------------------------------------------------------------------------
# randomly join or split hyphens/spaces
def hyphen_space_jitter(s: str) -> str:
    r = random.random()
    if r < 0.33:
        s = s.replace("-", " ")
    elif r < 0.66:
        s = s.replace(" ", "")
    else:
        # maybe add a hyphen between two tokens
        parts = s.split()
        if len(parts) >= 2:
            i = random.randint(0, len(parts) - 2)
            parts[i] = parts[i] + "-" + parts[i + 1]
            del parts[i + 1]
            s = " ".join(parts)
    return s


# -----------------------------------------------------------------------------
# Unicode homoglyph swaps
# TODO: add more homoglyphs
HOMOGLYPHS = {
    "A": "Α",  # Latin A -> Greek Alpha
    "B": "В",  # Latin B -> Cyrillic Ve
    "E": "Е",  # Latin E -> Cyrillic Ie
    "a": "а",  # Latin a -> Cyrillic a
    "e": "е",  # Latin e -> Cyrillic e
    "p": "р",  # Latin p -> Cyrillic er
    "H": "Н",  # Latin H -> Cyrillic en
}


# -----------------------------------------------------------------------------
# Randomly swap out homoglyphs at a certain rate
def homoglyph_swap(s: str, p: float = 0.05) -> str:
    out = []
    for ch in s:
        if ch in HOMOGLYPHS and random.random() < p:
            out.append(HOMOGLYPHS[ch])
        else:
            out.append(ch)
    return "".join(out)


# -----------------------------------------------------------------------------
# Muck about some with particles
def particle_omit_or_join(tokens: List[str], particles: List[str]) -> List[str]:
    i = 0
    out = []
    while i < len(tokens):
        t = tokens[i]
        if t and t.lower() in particles:
            r = random.random()
            if r < 0.5:
                # omit
                i += 1
                continue
            elif r < 0.9 and i + 1 < len(tokens):
                # join with next
                out.append((t + tokens[i + 1]).replace(" ", ""))
                i += 2
                continue
        out.append(t)
        i += 1
    return out


# -----------------------------------------------------------------------------
# Initials
def to_initials(tokens: List[str]) -> List[str]:
    if not tokens:
        return tokens
    out = tokens[:]
    # pick a token to initial
    idx = random.randrange(len(out))
    tok = out[idx]
    if tok and tok[0].isalpha():
        out[idx] = tok[0].upper()
    return out


# -----------------------------------------------------------------------------
# Abbreviations
# TODO: add more abbrieviations to the config file
def apply_abbreviation(tokens: List[str], abbr_map: Dict[str, List[str]]) -> List[str]:
    out = []
    for t in tokens:
        low = t.lower()
        if low in abbr_map and random.random() < 0.5:
            out.append(random.choice(abbr_map[low]))
        else:
            out.append(t)
    return out


# -----------------------------------------------------------------------------
# Nicknames
# TODO: add more nicknames to the the config file
def nickname_subst(tokens: List[str], nn_map: Dict[str, List[str]]) -> List[str]:
    out = []
    for t in tokens:
        low = t.lower()
        if low in nn_map and random.random() < 0.6:
            out.append(random.choice(nn_map[low]))
        else:
            out.append(t)
    return out


# -----------------------------------------------------------------------------
# Truncations
def truncate_name(s: str) -> str:
    if len(s) <= 3:
        return s
    mode = random.choice(["head", "tail", "mid"])
    n = max(1, int(len(s) * random.uniform(0.1, 0.3)))
    if mode == "head":
        return s[n:]
    elif mode == "tail":
        return s[:-n]
    else:
        i = random.randint(1, max(1, len(s) - n - 1))
        return s[:i] + s[i + n:]


# -----------------------------------------------------------------------------
# Token permutations:
# Swap names around.
def permute_tokens(tokens: List[str]) -> List[str]:
    if len(tokens) < 2:
        return tokens
    i, j = sorted(random.sample(range(len(tokens)), 2))
    out = tokens[:]
    out[i], out[j] = out[j], out[i]
    return out


# -----------------------------------------------------------------------------
# Placeholder insertion/removal
def placeholder_jitter(tokens: List[str], placeholders: List[str]) -> List[str]:
    r = random.random()
    out = tokens[:]
    if r < 0.5 and placeholders:
        # insert at front or back
        ph = random.choice(placeholders)
        if random.random() < 0.5:
            out = [ph] + out
        else:
            out = out + [ph]
    elif r < 0.7:
        # remove any placeholder present
        out = [t for t in out if t.upper() not in {p.upper() for p in placeholders}]
    return out


# -----------------------------------------------------------------------------
# Transliteration delegation (pluggable adapters)
def transliteration_cycle(s: str, lang: str) -> str:
    try:
        from . import translit
        out, _name = translit.transliterate(s, lang)
        return out
    except Exception:
        return s
