"""End-to-end tests: transcribe real recordings with the real model.

Downloads the default model (~785 MB) into local/ on first run.
"""

import asyncio
import re
import sys
import unicodedata
import wave
from asyncio.subprocess import PIPE

import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioStart, AudioStop, wav_to_chunks
from wyoming.event import async_read_event, async_write_event
from wyoming.info import Describe, Info

from wyoming_faster_whisper.const import DEFAULT_MODEL
from wyoming_faster_whisper.qwen3_asr_handler import Qwen3AsrTranscriber

from . import _DIR, _LOCAL_DIR, _SAMPLES_PER_CHUNK, _START_TIMEOUT, _TRANSCRIBE_TIMEOUT


async def _transcribe(wav_name: str, *extra_args: str, language: str = "en") -> str:
    """Start a server, send it one recording, and return the transcript."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "wyoming_faster_whisper",
        "--uri",
        "stdio://",
        "--data-dir",
        str(_LOCAL_DIR),
        "--language",
        language,
        *extra_args,
        stdin=PIPE,
        stdout=PIPE,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    # Check info
    await async_write_event(Describe().event(), proc.stdin)
    while True:
        event = await asyncio.wait_for(
            async_read_event(proc.stdout), timeout=_START_TIMEOUT
        )
        assert event is not None

        if not Info.is_type(event.type):
            continue

        info = Info.from_event(event)
        assert len(info.asr) == 1, "Expected one asr service"
        break

    await async_write_event(Transcribe().event(), proc.stdin)

    with wave.open(str(_DIR / wav_name), "rb") as example_wav:
        await async_write_event(
            AudioStart(
                rate=example_wav.getframerate(),
                width=example_wav.getsampwidth(),
                channels=example_wav.getnchannels(),
            ).event(),
            proc.stdin,
        )
        for chunk in wav_to_chunks(example_wav, _SAMPLES_PER_CHUNK):
            await async_write_event(chunk.event(), proc.stdin)

        await async_write_event(AudioStop().event(), proc.stdin)

    while True:
        event = await asyncio.wait_for(
            async_read_event(proc.stdout), timeout=_TRANSCRIBE_TIMEOUT
        )
        assert event is not None

        if Transcript.is_type(event.type):
            text = Transcript.from_event(event).text
            break

    # Need to close stdin for graceful termination
    proc.stdin.close()
    _, stderr = await proc.communicate()

    assert proc.returncode == 0, stderr.decode()
    return text


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z ]", "", text.lower().strip())


@pytest.mark.parametrize("extra_args", [[], ["--vad-clip"]])
@pytest.mark.asyncio
async def test_transcribe(extra_args: list) -> None:
    text = await _transcribe("turn_on_the_living_room_lamp.wav", *extra_args)
    assert _normalize(text) == "turn on the living room lamp"


@pytest.mark.asyncio
async def test_a_biasing_prompt_fixes_a_name() -> None:
    # Unbiased, the model hears "Ecobee" as "Igboy": it has no reason to think
    # the word exists.
    text = await _transcribe(
        "whats-the-temperature-of-the-ecobee.wav",
        "--initial-prompt",
        "Ecobee, Office Lamp.",
    )
    assert _normalize(text) == "whats the temperature of the ecobee"


# --- languages -------------------------------------------------------------


def _compact(text: str) -> str:
    """Drop punctuation and spaces and lower-case, for any script."""
    text = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    return text.replace(" ", "").lower()


@pytest.fixture(name="transcriber", scope="module")
def transcriber_fixture() -> Qwen3AsrTranscriber:
    # Loaded once and called directly: these recordings are already 16 kHz mono,
    # and test_transcribe covers the server around the model.
    return Qwen3AsrTranscriber(DEFAULT_MODEL, _LOCAL_DIR)


# "Turn on the living room light", as recorded in each language.
_LIVING_ROOM_LIGHT = {
    "en": "Turn on the living room light.",
    "de": "Schalte das Wohnzimmerlicht ein.",
    "es": "Enciende la luz del salón.",
    "ja": "リビングの電気をつけて。",
    # The speaker says 켜줘 ("turn it on"); the model hears the ending as 켜죠.
    "ko": "거실 불 켜죠.",
    "zh": "打开客厅的灯。",
    # Cantonese, transcribed in simplified Han with the colloquial 嘅.
    "yue": "开客厅嘅灯。",
}


@pytest.mark.parametrize(
    ("language", "recording"),
    [
        *((code, code) for code in _LIVING_ROOM_LIGHT),
        # Locale codes as Home Assistant sends them. zh-HK must stay Cantonese
        # rather than having its region stripped down to Mandarin.
        ("zh-CN", "zh"),
        ("zh-HK", "yue"),
        ("en-US", "en"),
    ],
)
def test_each_language_is_transcribed(transcriber, language, recording) -> None:
    text = transcriber.transcribe(_DIR / f"living_room_light_{recording}.wav", language)
    assert _compact(text) == _compact(_LIVING_ROOM_LIGHT[recording])


@pytest.mark.parametrize("recording", list(_LIVING_ROOM_LIGHT))
def test_each_language_is_detected(transcriber, recording) -> None:
    text = transcriber.transcribe(_DIR / f"living_room_light_{recording}.wav", None)
    assert _compact(text) == _compact(_LIVING_ROOM_LIGHT[recording])


@pytest.mark.parametrize("language", ["fr", "auto"])
@pytest.mark.asyncio
async def test_french_through_the_server(language) -> None:
    # 44.1 kHz, so this one goes through the server, which resamples it.
    text = await _transcribe("whats_your_name_french.wav", language=language)
    assert _compact(text) == _compact("Comment t'appelles-tu ?")
