import random
import re
import unicodedata
from typing import List, Tuple

# -----------------------------------------------------------------------------
QWERTY_NEIGHBORS = {
    "a": "qsxz", "e": "wsdr", "i": "ujko", "o": "iklp", "n": "bhjm", "m": "njk",
    # TODO: finish filling out, include uppercase?  build per keyboard layout?
}
OCR_CONFUSIONS = [("rn", "m"), ("cl", "d"), ("O", "0"), ("l", "1"), ("B", "8"), ("é", "e")]
PARTICLES = [r"\bde\b", r"\bdel\b", r"\bda\b", r"\bvan\b", r"\bal\b", r"\bbin\b"]

TRANSFORMS = [
    ("drop_diacritics", lambda s: drop_diacritics(s)),
    ("keyboard_subst", lambda s: keyboard_subst(s)),
    ("ocr_confuse", lambda s: ocr_confuse(s)),
    ("hyphen_space", lambda s: hyphen_space_jitter(s)),
]


# -----------------------------------------------------------------------------
def drop_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


# -----------------------------------------------------------------------------
def keyboard_subst(s: str, p=0.08) -> str:
    out = []
    for ch in s:
        if ch.lower() in QWERTY_NEIGHBORS and random.random() < p:
            cand = random.choice(QWERTY_NEIGHBORS[ch.lower()])
            out.append(cand.upper() if ch.isupper() else cand)
        else:
            out.append(ch)
    return "".join(out)


# -----------------------------------------------------------------------------
def ocr_confuse(s: str, k=1) -> str:
    for _ in range(k):
        pat, rep = random.choice(OCR_CONFUSIONS)
        s = re.sub(pat, rep, s, count=1)
    return s


# -----------------------------------------------------------------------------
def particle_omit_or_join(tokens: List[str]) -> List[str]:
    # drop or join particles with neighbors
    i = 0
    out = []
    while i < len(tokens):
        t = tokens[i]
        if any(re.fullmatch(p, t.lower()) for p in PARTICLES):
            if random.random() < 0.5:
                i += 1
                continue  # omit
            if i + 1 < len(tokens) and random.random() < 0.5:
                out.append((t + tokens[i + 1]).replace(" ", ""))
                i += 2
                continue
        out.append(t)
        i += 1
    return out


# -----------------------------------------------------------------------------
def hyphen_space_jitter(s: str):
    s = s.replace("-", " ") if random.random() < 0.5 else s.replace(" ", "")
    return s


# -----------------------------------------------------------------------------
def augment_name(full_name: str, max_ops=2) -> Tuple[str, List[str]]:
    ops = random.sample(TRANSFORMS, k=random.randint(0, max_ops))
    s = full_name
    log = []
    # simple token-aware particle rule
    tok = full_name.split()
    if random.random() < 0.3:
        tok = particle_omit_or_join(tok)
        s = " ".join(tok)
        log.append("particle_rule")
    for name, fn in ops:
        s2 = fn(s)
        if s2 != s:
            s = s2
            log.append(name)
    s = s.strip()
    return s, log


# =============================================================================
if __name__ == "__main__":
    result = augment_name("Ronald von Lynn-Myer", max_ops=len(TRANSFORMS))
    print(result)
