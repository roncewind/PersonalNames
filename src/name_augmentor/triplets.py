import random
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import regex as re

from .augment import Augmentor
from .config import AugmentConfig, load_default_config


# -----------------------------------------------------------------------------
def _drop_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


# -----------------------------------------------------------------------------
def _normalize(s: str) -> str:
    s = _drop_diacritics(s).lower()
    s = re.sub(r"[^\p{L}A-Za-z]+", " ", s)  # letters only
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------------------------------------------------------------
def _consonant_skeleton(s: str) -> str:
    return re.sub(r"[aeiou]+", "", _normalize(s))


# -----------------------------------------------------------------------------
def _initials(tokens: List[str]) -> str:
    return "".join(t[0] for t in tokens if t)


# -----------------------------------------------------------------------------
def _lev(a: str, b: str) -> int:
    # Simple Levenshtein distance
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[m]


# -----------------------------------------------------------------------------
@dataclass
class Triplet:
    anchor: str
    pos: str
    neg: str
    anchor_lang: Optional[str]
    pos_lang: Optional[str]
    neg_lang: Optional[str]
    cluster_id: str
    neg_strategy: str
    ops_anchor: List[str]
    ops_pos: List[str]


# -----------------------------------------------------------------------------
class TripletGenerator:
    # -------------------------------------------------------------------------
    def __init__(self, config: Optional[AugmentConfig] = None, seed: Optional[int] = None):
        self.config = config or load_default_config()
        self.aug = Augmentor(self.config, seed=seed)
        if seed is not None:
            random.seed(seed)

    # -------------------------------------------------------------------------
    def _surname(self, name: str) -> str:
        toks = _normalize(name).split()
        return toks[-1] if toks else ""

    # -------------------------------------------------------------------------
    def _prep_indices(self, rows: Sequence[Tuple[str, str, Optional[str]]]) -> Dict[str, Dict[str, List[int]]]:
        # rows: list of (cluster_id, name, lang)
        by_surname: Dict[str, List[int]] = {}
        by_sk: Dict[str, List[int]] = {}
        by_inits: Dict[str, List[int]] = {}
        for idx, (cid, name, lang) in enumerate(rows):
            toks = _normalize(name).split()
            sname = self._surname(name)
            sk = _consonant_skeleton(name)
            ini = _initials(toks[:2]) if len(toks) >= 2 else _initials(toks)
            for d, key in ((by_surname, sname), (by_sk, sk), (by_inits, ini)):
                if key:
                    d.setdefault(key, []).append(idx)
        return {"surname": by_surname, "skeleton": by_sk, "initials": by_inits}

    # -------------------------------------------------------------------------
    def _choose_negative(self, rows, idx_anchor, indices, exclude_cluster) -> Tuple[int, str]:
        cid_a, name_a, lang_a = rows[idx_anchor]
        toks_a = _normalize(name_a).split()
        sname_a = self._surname(name_a)
        sk_a = _consonant_skeleton(name_a)
        ini_a = _initials(toks_a[:2]) if len(toks_a) >= 2 else _initials(toks_a)

        candidates = []
        for key, dct, label in [
            (sname_a, indices["surname"], "same_surname"),
            (sk_a, indices["skeleton"], "consonant_skeleton"),
            (ini_a, indices["initials"], "same_initials")
        ]:
            if key in dct:
                for idx in dct[key]:
                    if rows[idx][0] != exclude_cluster:  # different cluster
                        candidates.append((idx, label))

        random.shuffle(candidates)
        for idx, label in candidates[:50]:
            # prefer semi-hard: small edit distance but not too small
            dist = _lev(_normalize(name_a), _normalize(rows[idx][1]))
            if 1 <= dist <= 3:
                return idx, f"{label}_lev{dist}"
        # fallback: any different cluster
        pool = [i for i, (cid, _, _) in enumerate(rows) if cid != exclude_cluster]
        idx = random.choice(pool)
        return idx, "random_diff_cluster"

    # -------------------------------------------------------------------------
    def generate(self, rows: Sequence[Tuple[str, str, Optional[str]]], per_cluster: int = 2,
                 noise_rate: float = 0.5, max_ops: int = 2) -> List[Triplet]:
        """Generate triplets from (cluster_id, name, lang). For each cluster, sample anchors and build positives via augmentation; negatives via heuristics."""
        # Build cluster map
        clusters: Dict[str, List[int]] = {}
        for i, (cid, _name, _lang) in enumerate(rows):
            clusters.setdefault(cid, []).append(i)

        indices = self._prep_indices(rows)
        out: List[Triplet] = []

        for cid, idxs in clusters.items():
            if len(idxs) == 0:
                continue
            for _ in range(per_cluster):
                a_idx = random.choice(idxs)
                a_cid, a_name, a_lang = rows[a_idx]
                p_lang = a_lang

                # Positive: pick within cluster or augment anchor (label-preserving)
                if len(idxs) > 1 and random.random() < 0.6:
                    p_idx = random.choice([i for i in idxs if i != a_idx]) if len(idxs) > 1 else a_idx
                    p_cid, p_name, p_lang = rows[p_idx]
                    pos_aug, ops_pos = self.aug.augment(p_name, lang=p_lang, noise_rate=noise_rate, max_ops=max_ops)
                else:
                    pos_aug, ops_pos = self.aug.augment(a_name, lang=a_lang, noise_rate=max_ops > 0 and noise_rate or 0.0, max_ops=max_ops)

                # Anchor augmentation (light)
                anc_aug, ops_anc = self.aug.augment(a_name, lang=a_lang, noise_rate=noise_rate, max_ops=max_ops)

                # Negative
                n_idx, label = self._choose_negative(rows, a_idx, indices, exclude_cluster=cid)
                n_cid, n_name, n_lang = rows[n_idx]

                trip = Triplet(
                    anchor=anc_aug, pos=pos_aug, neg=n_name,
                    anchor_lang=a_lang,
                    pos_lang=p_lang if len(idxs) > 1 else a_lang, neg_lang=n_lang,
                    cluster_id=cid, neg_strategy=label,
                    ops_anchor=ops_anc, ops_pos=ops_pos
                )
                out.append(trip)
        return out
