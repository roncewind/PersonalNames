from typing import Callable, Dict, Optional, Tuple

# Registry maps lang -> {"name": str, "fn": Callable[[str], str]}
_REGISTRY: Dict[str, Dict[str, object]] = {}


# -----------------------------------------------------------------------------
def register_adapter(lang: str, name: str, fn: Callable[[str], str]) -> None:
    """Register/override a transliteration adapter for a BCP47-like language code (e.g., 'ru', 'ar', 'ja')."""
    _REGISTRY[lang] = {"name": name, "fn": fn}


# -----------------------------------------------------------------------------
def get_adapter(lang: str) -> Optional[Callable[[str], str]]:
    from typing import cast
    entry = _REGISTRY.get(lang)
    return cast(Optional[Callable[[str], str]], entry.get("fn") if entry else None)


# -----------------------------------------------------------------------------
def get_adapter_name(lang: str) -> str:
    ent = _REGISTRY.get(lang)
    name = ent.get("name") if ent else None
    return str(name) if isinstance(name, str) else "none"


# -----------------------------------------------------------------------------
def transliterate(text: str, lang: Optional[str]) -> Tuple[str, str]:
    """Transliterate text for the given lang code using a registered or auto-loaded adapter.
    Returns (output, adapter_name). If no adapter, returns (text, 'none')."""
    if not lang:
        return text, "none"
    fn = get_adapter(lang)
    if fn is None:
        # attempt dynamic adapters if available
        _maybe_autoload(lang)
        fn = get_adapter(lang)
    if fn is None:
        return text, "none"
    try:
        return fn(text), get_adapter_name(lang)
    except Exception:
        return text, "error"


# -----------------------------------------------------------------------------
# ---- Built-in lightweight adapters ----
# Russian Cyrillic -> Latin (simple map)
_RU = {
    "А": "A", "а": "a", "Б": "B", "б": "b", "В": "V", "в": "v", "Г": "G", "г": "g", "Д": "D", "д": "d",
    "Е": "E", "е": "e", "Ё": "Yo", "ё": "yo", "Ж": "Zh", "ж": "zh", "З": "Z", "з": "z", "И": "I", "и": "i",
    "Й": "Y", "й": "y", "К": "K", "к": "k", "Л": "L", "л": "l", "М": "M", "м": "m", "Н": "N", "н": "n",
    "О": "O", "о": "o", "П": "P", "п": "p", "Р": "R", "р": "r", "С": "S", "с": "s", "Т": "T", "т": "t",
    "У": "U", "у": "u", "Ф": "F", "ф": "f", "Х": "Kh", "х": "kh", "Ц": "Ts", "ц": "ts", "Ч": "Ch", "ч": "ch",
    "Ш": "Sh", "ш": "sh", "Щ": "Shch", "щ": "shch", "Ы": "Y", "ы": "y", "Э": "E", "э": "e", "Ю": "Yu", "ю": "yu",
    "Я": "Ya", "я": "ya", "Ь": "", "ь": "", "Ъ": "", "ъ": ""
}


# -----------------------------------------------------------------------------
def _ru_cyr_to_lat(s: str) -> str:
    return "".join(_RU.get(ch, ch) for ch in s)


# # -----------------------------------------------------------------------------
# # Arabic -> Buckwalter-ish
# _AR = {
#     "ا": "A", "أ": "A", "إ": "I", "آ": "A", "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "h", "خ": "kh",
#     "د": "d", "ذ": "dh", "ر": "r", "ز": "z", "س": "s", "ش": "sh", "ص": "s", "ض": "d", "ط": "t", "ظ": "z",
#     "ع": "`", "غ": "gh", "ف": "f", "ق": "q", "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h", "ة": "h",
#     "و": "w", "ؤ": "w", "ي": "y", "ئ": "y", "ى": "a", "ﻻ": "la", "لا": "la", "ء": "'", "َ": "a", "ِ": "i", "ُ": "u"
# }


# # -----------------------------------------------------------------------------
# def _ar_to_buck(s: str) -> str:
#     return "".join(_AR.get(ch, ch) for ch in s)


# -----------------------------------------------------------------------------
# Greek -> Latin (rough)
_EL = {
    "Α": "A", "Β": "V", "Γ": "G", "Δ": "D", "Ε": "E", "Ζ": "Z", "Η": "I", "Θ": "Th", "Ι": "I", "Κ": "K", "Λ": "L", "Μ": "M",
    "Ν": "N", "Ξ": "X", "Ο": "O", "Π": "P", "Ρ": "R", "Σ": "S", "Τ": "T", "Υ": "Y", "Φ": "F", "Χ": "Ch", "Ψ": "Ps", "Ω": "O",
    "ά": "a", "έ": "e", "ί": "i", "ό": "o", "ύ": "y", "ή": "i", "ώ": "o", "ϊ": "i", "ϋ": "y", "ΐ": "i", "ΰ": "y",
    "α": "a", "β": "v", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "i", "θ": "th", "ι": "i", "κ": "k", "λ": "l", "μ": "m",
    "ν": "n", "ξ": "x", "ο": "o", "π": "p", "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "y", "φ": "f", "χ": "ch", "ψ": "ps", "ω": "o"
}


# -----------------------------------------------------------------------------
def _el_to_lat(s: str) -> str:
    return "".join(_EL.get(ch, ch) for ch in s)


# -----------------------------------------------------------------------------
# Register built-ins
register_adapter("ru", "builtin_ru_cyr2lat", _ru_cyr_to_lat)
# register_adapter("ar", "builtin_ar_buckwalter", _ar_to_buck)
register_adapter("el", "builtin_el_greeklish", _el_to_lat)


# -----------------------------------------------------------------------------
def _maybe_autoload(lang: str) -> None:
    # ar via arabic-buckwalter-transliteration
    if lang == "ar":
        try:
            from arabic_buckwalter_transliteration.transliteration import (
                arabic_to_buckwalter,
            )

            def _ar(s: str) -> str:
                return arabic_to_buckwalter(s)
            register_adapter("ar", "arabic_buckwalter", _ar)
        except Exception:
            pass

    # ja via pykakasi
    if lang == "ja":
        try:
            import pykakasi  # type: ignore
            kks = pykakasi.kakasi()

            def _ja(s: str) -> str:
                res = kks.convert(s)
                # possible result indices: ‘orig’, ‘kana’, ‘hira’, ‘hepburn’, ‘kunrei’, ‘passport’
                return " ".join([(r.get("hepburn") or r.get("kunrei") or r.get("passport") or r.get("orig") or "") for r in res])
            register_adapter("ja", "pykakasi_hepburn", _ja)
        except Exception:
            pass

    # zh via pypinyin
    if lang == "zh":
        try:
            # from pypinyin import pinyin  # type: ignore

            # def _zh(s: str) -> str:
            #     return ' '.join([item[0] for item in pinyin(s)])
            # register_adapter("zh", "pypinyin", _zh)

            from pypinyin import lazy_pinyin  # type: ignore

            def _zh(s: str) -> str:
                return " ".join(lazy_pinyin(s))
            register_adapter("zh", "pypinyin", _zh)
        except Exception:
            pass
