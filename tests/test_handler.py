"""Tests for what the event handler actually hands the transcriber.

Drives real Wyoming events through ``handle_event`` with a fake transcriber, so
the whole biasing path is covered in-process: AudioStart starts the refresh,
AudioStop waits for it, and the resulting prompt reaches ``transcribe()``. No
model and no network.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytest

pytest.importorskip("aiohttp")

# pylint: disable=wrong-import-position
from wyoming.asr import Transcribe, Transcript  # noqa: E402
from wyoming.audio import AudioChunk, AudioStart, AudioStop  # noqa: E402
from wyoming.info import Info  # noqa: E402

from wyoming_faster_whisper.const import Transcriber  # noqa: E402
from wyoming_faster_whisper.handler import TranscriptionHandler  # noqa: E402
from wyoming_faster_whisper.hass_api import HomeAssistantError  # noqa: E402
from wyoming_faster_whisper.name_cache import HassNameCache  # noqa: E402
from wyoming_faster_whisper.vocabulary import RecognitionContext  # noqa: E402

RATE = 16000


class FakeTranscriber(Transcriber):
    """Captures every call. Counts one token per word, times a multiplier."""

    def __init__(self, tokens_per_word: int = 1) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._tokens_per_word = tokens_per_word

    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        initial_prompt: Optional[str] = None,
    ) -> str:
        self.calls.append(
            {
                "language": language,
                "initial_prompt": initial_prompt,
                "wav_path": str(wav_path),
            }
        )
        return "fake transcript"

    def count_prompt_tokens(self, text: str) -> int:
        return len(text.split()) * self._tokens_per_word

    @property
    def prompt(self) -> Optional[str]:
        """The prompt from the last transcribe() call."""
        assert self.calls, "transcribe() was never called"
        return self.calls[-1]["initial_prompt"]


class FakeHass:
    """Stands in for HomeAssistant."""

    def __init__(self, context: Optional[RecognitionContext] = None, delay=0.0) -> None:
        self.context = context if context is not None else RecognitionContext()
        self.delay = delay
        self.fail = False
        self.calls = 0

    async def get_context(self) -> RecognitionContext:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)

        if self.fail:
            raise HomeAssistantError("unreachable")

        return self.context


class Handler(TranscriptionHandler):
    """Captures written events instead of needing a socket."""

    def __init__(self, *args, **kwargs) -> None:
        self.written: List[Any] = []
        super().__init__(*args, **kwargs)

    async def write_event(self, event) -> None:
        self.written.append(event)

    @property
    def transcript(self) -> str:
        for event in self.written:
            if Transcript.is_type(event.type):
                return Transcript.from_event(event).text

        raise AssertionError("no transcript was written")


def _names(*entities) -> RecognitionContext:
    return RecognitionContext(priority_entities=list(entities))


def _handler(transcriber, names=None, **kwargs) -> Handler:
    kwargs.setdefault("language", "en")
    return Handler(Info(), transcriber, names, None, None, **kwargs)


def _cache(hass, **kwargs) -> HassNameCache:
    kwargs.setdefault("count_tokens", lambda text: len(text.split()))
    kwargs.setdefault("max_tokens", 1000)
    return HassNameCache(hass, **kwargs)


async def _utterance(handler: Handler, chunks: int = 3, pace: float = 0.0) -> None:
    """One full request, the way Home Assistant sends it."""
    await handler.handle_event(Transcribe(language="en").event())
    await handler.handle_event(AudioStart(rate=RATE, width=2, channels=1).event())
    for _ in range(chunks):
        await handler.handle_event(
            AudioChunk(audio=b"\x00\x00" * 160, rate=RATE, width=2, channels=1).event()
        )
        # Let background work run, as it does between real chunks.
        await asyncio.sleep(pace)

    await handler.handle_event(AudioStop().event())


# --- the prompt that reaches the transcriber ------------------------------


async def test_without_a_token_the_configured_prompt_is_used():
    transcriber = FakeTranscriber()
    handler = _handler(transcriber, names=None, initial_prompt="Vocabulary:")

    await _utterance(handler)

    assert transcriber.prompt == "Vocabulary:"


async def test_without_a_token_or_a_prompt_nothing_is_sent():
    transcriber = FakeTranscriber()
    await _utterance(_handler(transcriber))

    assert transcriber.prompt is None


async def test_home_assistant_names_reach_the_transcriber():
    transcriber = FakeTranscriber()
    handler = _handler(transcriber, names=_cache(FakeHass(_names("Ecobee", "Nanit"))))

    await _utterance(handler)

    assert transcriber.prompt == "Ecobee, Nanit."


async def test_names_are_appended_to_the_configured_prompt():
    transcriber = FakeTranscriber()
    cache = _cache(FakeHass(_names("Ecobee")), prefix="Vocabulary:")
    handler = _handler(transcriber, names=cache, initial_prompt="Vocabulary:")

    await _utterance(handler)

    assert transcriber.prompt == "Vocabulary: Ecobee."


async def test_the_transcript_still_comes_back():
    transcriber = FakeTranscriber()
    handler = _handler(transcriber, names=_cache(FakeHass(_names("Ecobee"))))

    await _utterance(handler)

    assert handler.transcript == "fake transcript"


# --- the AudioStart window ------------------------------------------------


async def test_audio_start_starts_the_fetch_and_audio_stop_uses_it():
    # Nothing is cached up front, so the names in the prompt can only have come
    # from a fetch that began at AudioStart and landed by AudioStop.
    hass = FakeHass(_names("Ecobee"))
    transcriber = FakeTranscriber()
    handler = _handler(transcriber, names=_cache(hass))

    await _utterance(handler)

    assert hass.calls == 1
    assert transcriber.prompt == "Ecobee."


async def test_an_utterance_fetches_once_even_over_many_paced_chunks():
    # The regression: a finished fetch is immediately due again, so without a
    # per-utterance latch every chunk would start another one. Pacing matters --
    # the fetch has to be able to finish mid-utterance to re-arm.
    hass = FakeHass(_names("Ecobee"))
    handler = _handler(FakeTranscriber(), names=_cache(hass))

    await _utterance(handler, chunks=30, pace=0.001)

    assert hass.calls == 1


async def test_a_second_utterance_fetches_again():
    hass = FakeHass(_names("Ecobee"))
    handler = _handler(FakeTranscriber(), names=_cache(hass))

    await _utterance(handler)
    await _utterance(handler)

    assert hass.calls == 2


async def test_a_rename_between_utterances_is_picked_up():
    hass = FakeHass(_names("Ecobee"))
    transcriber = FakeTranscriber()
    handler = _handler(transcriber, names=_cache(hass))

    await _utterance(handler)
    assert transcriber.prompt == "Ecobee."

    hass.context = _names("Ecobee", "Nanit")
    await _utterance(handler)
    assert transcriber.prompt == "Ecobee, Nanit."


async def test_chunks_without_audio_start_still_refresh():
    # Safety net for clients that skip AudioStart.
    hass = FakeHass(_names("Ecobee"))
    transcriber = FakeTranscriber()
    handler = _handler(transcriber, names=_cache(hass))

    await handler.handle_event(Transcribe(language="en").event())
    for _ in range(3):
        await handler.handle_event(
            AudioChunk(audio=b"\x00\x00" * 160, rate=RATE, width=2, channels=1).event()
        )
        await asyncio.sleep(0)

    await handler.handle_event(AudioStop().event())

    assert hass.calls == 1
    assert transcriber.prompt == "Ecobee."


# --- token budget --------------------------------------------------------


async def test_the_budget_uses_the_models_own_tokenizer():
    # 10 tokens per word: "Ecobee, Nanit." is 2 words -> 20, over a 15 budget,
    # so only the first name fits.
    transcriber = FakeTranscriber(tokens_per_word=10)
    cache = _cache(
        FakeHass(_names("Ecobee", "Nanit")),
        count_tokens=transcriber.count_prompt_tokens,
        max_tokens=15,
    )
    handler = _handler(transcriber, names=cache)

    await _utterance(handler)

    assert transcriber.prompt == "Ecobee."


# --- degradation ---------------------------------------------------------


async def test_an_unreachable_home_assistant_still_transcribes():
    hass = FakeHass(_names("Ecobee"))
    hass.fail = True
    transcriber = FakeTranscriber()
    handler = _handler(
        transcriber,
        names=_cache(hass, prefix="Vocabulary:"),
        initial_prompt="Vocabulary:",
    )

    await _utterance(handler)

    assert handler.transcript == "fake transcript"
    assert transcriber.prompt == "Vocabulary:"


async def test_a_slow_fetch_does_not_stop_the_transcript():
    hass = FakeHass(_names("Ecobee"), delay=5.0)
    transcriber = FakeTranscriber()
    handler = _handler(transcriber, names=_cache(hass, wait_timeout=0.01))

    await _utterance(handler)

    assert handler.transcript == "fake transcript"
    assert transcriber.prompt is None, "no names had arrived yet"


async def test_audio_stop_with_no_audio_returns_an_empty_transcript():
    transcriber = FakeTranscriber()
    handler = _handler(transcriber, names=_cache(FakeHass(_names("Ecobee"))))

    await handler.handle_event(Transcribe(language="en").event())
    await handler.handle_event(AudioStop().event())

    assert handler.transcript == ""
    assert not transcriber.calls


class ImmediateEndpoint:
    def reset(self) -> None:
        pass

    def process(self, _audio: bytes) -> bool:
        return False


async def test_vad_endpointing_sends_one_transcript_before_audio_stop():
    transcriber = FakeTranscriber()
    handler = _handler(transcriber, vad_endpointing=0.7)
    handler._endpoint_detector = ImmediateEndpoint()

    await handler.handle_event(AudioStart(rate=RATE, width=2, channels=1).event())
    await handler.handle_event(
        AudioChunk(audio=b"\x00\x00" * 160, rate=RATE, width=2, channels=1).event()
    )

    assert handler.transcript == "fake transcript"
    assert len(transcriber.calls) == 1

    # Chunks sent while the client is still working toward AudioStop are ignored,
    # and AudioStop does not generate a duplicate transcript.
    await handler.handle_event(
        AudioChunk(audio=b"\x00\x00" * 160, rate=RATE, width=2, channels=1).event()
    )
    await handler.handle_event(AudioStop().event())

    transcripts = [event for event in handler.written if Transcript.is_type(event.type)]
    assert len(transcripts) == 1
    assert len(transcriber.calls) == 1


# --- --vad-clip ------------------------------------------------------------


async def test_clipping_is_skipped_when_disabled():
    transcriber = FakeTranscriber()
    handler = _handler(transcriber, vad_clip=False)

    clipped: List[str] = []

    def fake_clip(src, dst, threshold, pad_ms) -> bool:
        clipped.append(str(src))
        return True

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("wyoming_faster_whisper.handler.clip_wav_to_speech", fake_clip)
        await _utterance(handler)

    assert not clipped
    assert transcriber.calls[-1]["wav_path"] == handler._wav_path


async def test_clipped_audio_is_transcribed_when_speech_is_found():
    transcriber = FakeTranscriber()
    handler = _handler(
        transcriber, vad_clip=True, vad_clip_threshold=0.3, vad_clip_pad_ms=100
    )

    calls: List[Any] = []

    def fake_clip(src, dst, threshold, pad_ms) -> bool:
        calls.append((str(src), threshold, pad_ms))
        return True

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("wyoming_faster_whisper.handler.clip_wav_to_speech", fake_clip)
        await _utterance(handler)

    assert calls == [(handler._wav_path, 0.3, 100)]
    assert transcriber.calls[-1]["wav_path"] == handler._clipped_wav_path


async def test_silence_only_falls_back_to_the_original_audio():
    transcriber = FakeTranscriber()
    handler = _handler(transcriber, vad_clip=True)

    def fake_clip(src, dst, threshold, pad_ms) -> bool:
        return False  # no speech found

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("wyoming_faster_whisper.handler.clip_wav_to_speech", fake_clip)
        await _utterance(handler)

    assert transcriber.calls[-1]["wav_path"] == handler._wav_path


# --- the language that reaches the transcriber -----------------------------


def _language_after(requested, preferred="en"):
    """Report what the handler resolves a client's Transcribe language to."""
    handler = _handler(FakeTranscriber(), language=preferred)
    asyncio.run(handler.handle_event(Transcribe(language=requested).event()))
    return handler._language  # noqa: SLF001


def test_auto_language_is_treated_as_no_language() -> None:
    """A client can ask for detection by name; the model spells that as None."""
    assert _language_after("auto", preferred=None) is None


def test_auto_language_still_falls_back_to_the_preferred_language() -> None:
    assert _language_after("auto", preferred="de") == "de"


def test_explicit_language_is_kept() -> None:
    assert _language_after("fr", preferred="de") == "fr"


def test_missing_language_falls_back_to_the_preferred_language() -> None:
    assert _language_after(None, preferred="de") == "de"


async def test_the_default_language_applies_without_a_transcribe_event():
    """--language must not depend on the client sending Transcribe."""
    transcriber = FakeTranscriber()
    handler = _handler(transcriber, language="de")

    for _ in range(2):  # and again after the first utterance resets
        await handler.handle_event(AudioStart(rate=RATE, width=2, channels=1).event())
        await handler.handle_event(
            AudioChunk(audio=b"\x00\x00" * 160, rate=RATE, width=2, channels=1).event()
        )
        await handler.handle_event(AudioStop().event())

    assert [call["language"] for call in transcriber.calls] == ["de", "de"]
