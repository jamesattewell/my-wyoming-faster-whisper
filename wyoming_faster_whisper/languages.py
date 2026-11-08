"""The languages Qwen3-ASR supports, and normalizing a client's code onto them.

``SUPPORTED_LANGUAGES`` is what build_info reports to Home Assistant, so it is
written in the tag form HA's matcher (homeassistant.util.language) compares
against: bare ISO-639-1 codes, plus a region qualifier only where the region
selects genuinely different model behaviour.

The region qualifier matters for exactly one case today. HA's matcher treats
"yue" and "zh" as unrelated languages, so a Cantonese-capable backend that only
advertises "yue" is invisible to HA's zh-HK pipeline -- home-assistant/intents
ships zh-CN/zh-HK/zh-TW and no bare "zh" or "yue" at all. Advertising "zh-HK"
instead scores an exact region match, so a zh-HK pipeline reaches Cantonese while
zh-CN/zh-TW still fall through to Mandarin via "zh".
"""

import re
from typing import Optional

# Home Assistant's own tag parser splits on either separator, so we do too.
_SEPARATORS = re.compile(r"[-_]")


def normalize_language_tag(code: str) -> str:
    """Lower-case a language tag and settle on "-" as the separator.

    "zh_HK" -> "zh-hk". The table below is keyed by full tag, so a client using
    the underscore form would otherwise miss the entry and fall back to the base
    language -- which for Hong Kong silently means Mandarin instead of Cantonese.
    """
    return _SEPARATORS.sub("-", code.lower(), count=1)


def base_language(code: str) -> str:
    """Strip the region from a language tag and lower-case it.

    "en-US" -> "en". Home Assistant sends back the tag we advertised, but the
    --language option is typed by hand and other Wyoming clients send whatever
    they like.
    """
    return _SEPARATORS.split(code.lower(), maxsplit=1)[0]


def canonical_language_code(code: str) -> str:
    """Upper-case the region of a language tag, as Home Assistant writes it.

    "zh-hk" -> "zh-HK". Bare codes are returned unchanged. Script subtags
    (e.g. "sr-Latn") would be mangled, but none are reported here.
    """
    language, separator, region = code.partition("-")
    return f"{language}{separator}{region.upper()}"


# Language names Qwen3-ASR was trained to accept in the forced-language suffix.
_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
    # Hong Kong audio is typically Cantonese. Without this entry the locale
    # fallback below strips the region and forces Mandarin, silently losing
    # Cantonese on a model that supports it.
    "zh-hk": "Cantonese",
}


def qwen3_asr_language(language: Optional[str]) -> Optional[str]:
    """Normalize a language code to a Qwen3-ASR language name, or None.

    None means "let the model detect the language", which is also the right
    outcome for a code it does not know.
    """
    if not language:
        return None

    code = normalize_language_tag(language)
    name = _LANGUAGE_NAMES.get(code)
    if name is None:
        # Accept locale-style codes like "en-US" or "zh-CN".
        name = _LANGUAGE_NAMES.get(base_language(code))

    return name


# Advertised form of the keys above (regions upper-cased, as HA writes them).
SUPPORTED_LANGUAGES = frozenset(
    canonical_language_code(code) for code in _LANGUAGE_NAMES
)
