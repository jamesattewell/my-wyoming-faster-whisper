"""Tests for language-code normalization and the advertised language list."""

import pytest

from wyoming_faster_whisper.languages import (
    SUPPORTED_LANGUAGES,
    base_language,
    canonical_language_code,
    qwen3_asr_language,
)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("en", "English"),
        ("EN", "English"),
        ("en-US", "English"),  # locale-style
        ("en_GB", "English"),
        ("zh", "Chinese"),
        ("zh-CN", "Chinese"),
        ("zh-TW", "Chinese"),
        # Hong Kong is Cantonese, not Mandarin. Without the explicit entry the
        # locale fallback would strip "-HK" and force Chinese.
        ("zh-HK", "Cantonese"),
        ("zh_HK", "Cantonese"),
        ("ZH_hk", "Cantonese"),
        ("yue", "Cantonese"),
        ("fil", "Filipino"),
        # Unsupported / empty -> None, which means auto-detect.
        ("xx", None),
        ("ga", None),
        ("auto", None),
        ("", None),
        (None, None),
    ],
)
def test_qwen3_asr_language(code, expected) -> None:
    assert qwen3_asr_language(code) == expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("en", "en"),
        ("en-US", "en"),
        ("EN_gb", "en"),
        ("zh-HK", "zh"),
        ("fil", "fil"),
    ],
)
def test_base_language(code, expected) -> None:
    assert base_language(code) == expected


def test_advertised_codes_all_resolve() -> None:
    """Every code we report must map to a trained language name."""
    for code in SUPPORTED_LANGUAGES:
        assert qwen3_asr_language(code) is not None, code


def test_advertised_codes_are_canonically_cased() -> None:
    """Home Assistant writes regions upper-cased (zh-HK), so we must too."""
    for code in SUPPORTED_LANGUAGES:
        assert canonical_language_code(code) == code, code


def test_cantonese_is_advertised_as_zh_hk() -> None:
    """Home Assistant's matcher treats "yue" and "zh" as unrelated languages.

    home-assistant/intents ships zh-CN/zh-HK/zh-TW and no bare "zh" or "yue", so
    a backend that only says "yue" can never be reached by a Cantonese pipeline.
    """
    assert "zh-HK" in SUPPORTED_LANGUAGES


def test_thirty_one_languages_are_advertised() -> None:
    assert len(SUPPORTED_LANGUAGES) == 31
