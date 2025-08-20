import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import regex as re

from . import transforms as T
from .config import AugmentConfig, load_default_config

# TransformFn = Callable[[str], str]


# -----------------------------------------------------------------------------
@dataclass
class TransformSpec:
    name: str
    fn: Callable[..., str]
    kwargs: Dict[str, Any]


# -----------------------------------------------------------------------------
DEFAULT_REGISTRY: Dict[str, TransformSpec] = {
    "drop_diacritics": TransformSpec("drop_diacritics", T.drop_diacritics, {}),
    "nfkc_normalize": TransformSpec("nfkc_normalize", T.nfkc_normalize, {}),
    "case_randomize": TransformSpec("case_randomize", T.case_randomize, {"p": 0.15}),
    "case_upper": TransformSpec("case_upper", T.case_upper, {}),
    "case_lower": TransformSpec("case_lower", T.case_lower, {}),
    "case_capitalize": TransformSpec("case_capitalize", T.case_capitalize, {}),
    "case_title": TransformSpec("case_title", T.case_title, {}),
    "keyboard_subst": TransformSpec("keyboard_subst", T.keyboard_subst, {"p": 0.08}),
    "ocr_confuse": TransformSpec("ocr_confuse", T.ocr_confuse, {"k": 1}),
    "hyphen_space": TransformSpec("hyphen_space", T.hyphen_space_jitter, {}),
    "homoglyph_swap": TransformSpec("homoglyph_swap", T.homoglyph_swap, {"p": 0.05}),
    "truncate": TransformSpec("truncate", T.truncate_name, {}),
}


# -----------------------------------------------------------------------------
@dataclass
class LanguageProfile:
    particles: List[str]
    nicknames: Dict[str, List[str]]
    abbreviations: Dict[str, List[str]]
    placeholders: List[str]
    allow_transliteration: bool


# -----------------------------------------------------------------------------
class Augmentor:
    # -------------------------------------------------------------------------
    def __init__(self, config: Optional[AugmentConfig] = None, registry: Optional[Dict[str, TransformSpec]] = None, seed: Optional[int] = None):
        self.config = config or load_default_config()
        self.registry = registry or DEFAULT_REGISTRY
        if seed is not None:
            random.seed(seed)

    # -------------------------------------------------------------------------
    # language profiles are set up in the configuration file
    def _get_profile(self, lang: Optional[str]) -> LanguageProfile:
        langs = self.config.languages
        base = langs.get("default", {})
        overlay = langs.get(lang, {}) if lang else {}
        merged = {**base, **overlay}
        return LanguageProfile(
            particles=[p.lower() for p in merged.get("particles", [])],
            nicknames={k.lower(): v for k, v in merged.get("nicknames", {}).items()},
            abbreviations={k.lower(): v for k, v in merged.get("abbreviations", {}).items()},
            placeholders=merged.get("placeholders", []),
            allow_transliteration=bool(merged.get("allow_transliteration", False)),
        )

    # -------------------------------------------------------------------------
    def apply_token_rules(self, s: str, prof: LanguageProfile) -> Tuple[str, List[str]]:
        ops = []
        tokens = s.split()
        # particle rules
        if prof.particles and random.random() < 0.35:
            tokens = T.particle_omit_or_join(tokens, prof.particles)
            ops.append("particle_rule")
        # nickname
        if prof.nicknames and random.random() < 0.4:
            tokens = T.nickname_subst(tokens, prof.nicknames)
            ops.append("nickname")
        # abbreviation
        if prof.abbreviations and random.random() < 0.25:
            tokens = T.apply_abbreviation(tokens, prof.abbreviations)
            ops.append("abbreviation")
        # initials
        if random.random() < 0.25 and len(tokens) >= 2:
            tokens = T.to_initials(tokens)
            ops.append("initials")
        # placeholders
        if prof.placeholders and random.random() < 0.15:
            tokens = T.placeholder_jitter(tokens, prof.placeholders)
            ops.append("placeholder_jitter")
        return " ".join(tokens), ops

    # -------------------------------------------------------------------------
    def maybe_transliterate(self, s: str, lang: Optional[str], prof: LanguageProfile, p: float = 0.25) -> Tuple[str, List[str]]:
        ops = []
        if prof.allow_transliteration and lang and random.random() < p:
            s2 = T.transliteration_cycle(s, lang)
            if s2 != s:
                s = s2
                ops.append(f"transliteration({lang})")
        return s, ops

    # -------------------------------------------------------------------------
    def augment(self, full_name: str, lang: Optional[str] = None, max_ops: int = 2, noise_rate: float = 0.35) -> Tuple[str, List[str]]:
        """Augments a name by way of various name mangling methods.
        Args:
            full_name (str): Name to mangle.
            lang (str): Language of the name, used to load language specific manglations.
            max_ops (int): Maximum number of _character_ operations to perform on the name not total operations.
            noise_rate (float): percentage chance of manglation happening. Random# > noise_rate -> manglation.
        Returns:
            tuple: A tuple containing:
                - str: augmented name
                - list[str]: list of the operations applied to the given name.
        """
        s = full_name.strip()
        log: List[str] = []

        # TODO: how should we control the probability of name manglation happening?
        if random.random() > noise_rate:
            return s, log

        prof = self._get_profile(lang)

        # token-aware rules first
        s, ops = self.apply_token_rules(s, prof)
        log.extend(ops)

        # transliteration
        s, ops = self.maybe_transliterate(s, lang, prof)
        log.extend(ops)

        # TODO: round trip transliteration  ar -> en -> ar sort of thing

        # pick character-level transforms
        names = list(self.registry.keys())
        k = random.randint(0, max_ops)
        k = k if k <= len(names) else len(names)

        if k > 0:
            for spec_name in random.sample(names, k=k):
                spec = self.registry[spec_name]
                before = s
                s = spec.fn(s, **spec.kwargs)
                if s != before:
                    log.append(spec.name)

        # final cleanup, remove extra whitespace
        s = re.sub(r"\s+", " ", s).strip()
        return s, log

    # -------------------------------------------------------------------------
    def batch_augment(self, rows: Sequence[Tuple[str, Optional[str]]], **kwargs) -> List[Tuple[str, str, List[str]]]:
        """rows: list of (name, lang). Return list of (name, augmented, ops)."""
        out = []
        for name, lang in rows:
            aug, ops = self.augment(name, lang=lang, **kwargs)
            out.append((name, aug, ops))
        return out
