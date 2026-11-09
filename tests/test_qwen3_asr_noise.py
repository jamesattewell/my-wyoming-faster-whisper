"""Qwen3-ASR must not echo the Home Assistant biasing prompt on noise.

Qwen3-ASR has no notion of "no speech". Speech-like noise (a TV, people
talking in the next room) gets past Silero VAD and reaches the decoder, and given
nothing it can read in the audio, the model copies the only text in its context
-- the entity/area names from Home Assistant -- back out as the transcript, which
would then be sent to the intent pipeline as if the user had said it.

These run the real model to check that the echo guard catches that, does not
fire on real commands (or undo the biasing), and leaves the cached prompt alone.

Every test runs once per model size. Each model is skipped unless it is in
local/: the 0.6B default is put there by running the server once with
--stt-library qwen3-asr; the 1.7B has no published merged export, so it is
built locally into local/qwen3-asr-1.7b-onnx-int4-merged by
convert_qwen3_asr.py (see "Other Model Sizes" in the README).
"""

import wave
from pathlib import Path
from typing import List

import numpy as np
import pytest

from . import _DIR, _LOCAL_DIR

qwen3_asr_handler = pytest.importorskip(
    "wyoming_faster_whisper.qwen3_asr_handler",
    reason="qwen3-asr extra not installed",
)

from wyoming_faster_whisper.vad import clip_to_speech  # noqa: E402

_MODELS = {
    "0.6b": _LOCAL_DIR / "rhasspy" / "qwen3-asr-0.6b-onnx-int4-merged",
    "1.7b": _LOCAL_DIR / "qwen3-asr-1.7b-onnx-int4-merged",
}
_RATE = 16000

# What NameCache builds from a typical home (see test_hass_api), plus a
# multi-token name that one of the fixtures says in full.
_HASS_PROMPT = (
    "Office, Study, Upstairs, Top Floor, Living Room Light, Office Lamp, "
    "Desk Lamp, Kitchen, Downstairs, Ecobee, Thermostat."
)
_HASS_NAMES = [name.strip(" .") for name in _HASS_PROMPT.split(",")]

# Speech used only as raw material for the noise; no words survive.
_SOURCES = [
    "turn-off-the-lights-please.wav",
    "turn-on-the-office-lamp.wav",
    "whats-the-temperature-of-the-ecobee.wav",
    "nevermind.wav",
    "living_room_light_de.wav",
    "living_room_light_ja.wav",
]


def _load(name: str) -> np.ndarray:
    with wave.open(str(_DIR / name), "rb") as wav_file:
        pcm = wav_file.readframes(wav_file.getnframes())
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _shuffled_speech(n: int) -> np.ndarray:
    """Speech chopped into 40 ms pieces and shuffled.

    Has the spectrum and energy of speech, so VAD fires on it, but every
    phoneme is cut apart and reordered.
    """
    audio = np.concatenate([_load(name) for name in _SOURCES])
    seg = _RATE * 40 // 1000
    k = len(audio) // seg
    pieces = audio[: k * seg].reshape(k, seg)
    pieces = pieces[np.random.default_rng(1234).permutation(k)]
    return np.resize(pieces.reshape(-1), n)


def _babble(n: int) -> np.ndarray:
    """Several talkers played backwards over each other: crowd/TV murmur."""
    return sum(np.resize(_load(name)[::-1], n) for name in _SOURCES)


def _write_wav(path: Path, audio: np.ndarray, gain: float = 0.5) -> bytes:
    audio = gain * audio / np.abs(audio).max()
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    wav_file: wave.Wave_write = wave.open(str(path), "wb")
    with wav_file:
        wav_file.setframerate(_RATE)
        wav_file.setsampwidth(2)
        wav_file.setnchannels(1)
        wav_file.writeframes(pcm)
    return pcm


@pytest.fixture(name="model", scope="module", params=sorted(_MODELS))
def model_fixture(request, tmp_path_factory):
    model_dir = _MODELS[request.param]
    if not model_dir.is_dir():
        pytest.skip(f"Qwen3-ASR {request.param} not found in {model_dir}")

    return qwen3_asr_handler.Qwen3AsrTranscriber(
        str(model_dir), str(tmp_path_factory.mktemp("qwen3-cache"))
    )


def _noise_wav(tmp_path, make_noise, gain: float) -> Path:
    wav_path = tmp_path / "noise.wav"
    pcm = _write_wav(wav_path, make_noise(3 * _RATE), gain)

    # Precondition: this noise must get past VAD. If it did not, VAD would be
    # the thing filtering it and the decoder would not be under test.
    assert clip_to_speech(pcm), "VAD rejected the noise; test is not meaningful"
    return wav_path


def _echoed_names(text: str) -> List[str]:
    return [name for name in _HASS_NAMES if name.lower() in text.lower()]


# --- noise -----------------------------------------------------------------


@pytest.mark.parametrize("language", ["en", "de"])
@pytest.mark.parametrize("gain", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("make_noise", [_shuffled_speech, _babble])
def test_noise_does_not_echo_the_hass_prompt(
    model, tmp_path, make_noise, gain, language
) -> None:
    wav_path = _noise_wav(tmp_path, make_noise, gain)
    text = model.transcribe(wav_path, language=language, initial_prompt=_HASS_PROMPT)
    echoed = _echoed_names(text)
    assert len(echoed) < 3, f"Transcript echoes the biasing prompt: {text!r}"


def test_an_echo_is_cut_short(model, tmp_path, monkeypatch) -> None:
    """The echo is abandoned early, not read out to the end of the list."""
    steps = []
    run_decoder = model._run_decoder

    def counting(input_embeds, *args):
        if input_embeds.shape[1] == 1:
            steps.append(1)
        return run_decoder(input_embeds, *args)

    monkeypatch.setattr(model, "_run_decoder", counting)
    wav_path = _noise_wav(tmp_path, _shuffled_speech, 0.5)
    assert model.transcribe(wav_path, "en", initial_prompt=_HASS_PROMPT) == ""

    prompt_tokens = model.count_prompt_tokens(_HASS_PROMPT)
    assert len(steps) < prompt_tokens // 2, (len(steps), prompt_tokens)


# --- real speech -----------------------------------------------------------


@pytest.mark.parametrize(
    ("wav_name", "language", "expected"),
    [
        ("turn-on-the-office-lamp.wav", "en", "Turn on the office lamp."),
        # Biasing still works: without the prompt this is "...of the Igboy?"
        (
            "whats-the-temperature-of-the-ecobee.wav",
            "en",
            "What's the temperature of the Ecobee?",
        ),
        # Language not forced: the model writes its own preamble first.
        (
            "whats-the-temperature-of-the-ecobee.wav",
            None,
            "What's the temperature of the Ecobee?",
        ),
        # A multi-token name from the list, said in full.
        ("living_room_light_en.wav", "en", "Turn on the living room light."),
        ("living_room_light_de.wav", "de", "Schalte das Wohnzimmerlicht ein."),
        ("turn-off-the-lights-please.wav", "en", "Turn off the lights, please."),
        ("nevermind.wav", "en", "Never mind."),
        ("set-volume-to-30-percent.wav", "en", "Set volume to thirty percent."),
    ],
)
def test_real_commands_are_unaffected(model, wav_name, language, expected) -> None:
    text = model.transcribe(_DIR / wav_name, language, initial_prompt=_HASS_PROMPT)
    assert text == expected


# Every 16 kHz speech fixture. (The 44.1 kHz ones are resampled by the server
# before they reach the transcriber, which only takes 16 kHz.)
_SPEECH = sorted(
    path.name
    for path in _DIR.glob("*.wav")
    if wave.open(str(path), "rb").getframerate() == _RATE
)


@pytest.mark.parametrize("wav_name", _SPEECH)
def test_the_guard_never_changes_a_real_command(model, wav_name) -> None:
    """Whatever the model hears, the guard must pass real speech through as is."""
    wav_path = _DIR / wav_name
    guarded = model.transcribe(wav_path, None, initial_prompt=_HASS_PROMPT)

    # The same decode with the guard switched off.
    with wave.open(str(wav_path), "rb") as wav_file:
        pcm = wav_file.readframes(wav_file.getnframes())
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    mel = qwen3_asr_handler._log_mel_spectrogram(audio, model._mel_filters)
    features = model._encoder.run(None, {model._encoder_inputs[0]: mel})[0]
    context_ids, language_ids = model._context_ids(_HASS_PROMPT, None)
    tokens = model._generate(
        features,
        model._build_prompt_ids(features.shape[1], context_ids, language_ids),
        len(qwen3_asr_handler._system_ids(context_ids)),
    )
    assert tokens is not None
    if qwen3_asr_handler._ASR_TEXT in tokens:
        tokens = tokens[tokens.index(qwen3_asr_handler._ASR_TEXT) + 1 :]
    unguarded = model._tokenizer.decode(tokens, skip_special_tokens=True).strip()

    assert guarded == unguarded


# --- the prefix cache ------------------------------------------------------


def test_noise_does_not_cost_the_next_command_its_cached_prompt(
    model, tmp_path, monkeypatch
) -> None:
    command = _DIR / "turn-on-the-office-lamp.wav"
    expected = "Turn on the office lamp."
    assert model.transcribe(command, "en", initial_prompt=_HASS_PROMPT) == expected
    cached = model._prefix_kv
    assert cached is not None

    wav_path = _noise_wav(tmp_path, _shuffled_speech, 0.5)
    assert model.transcribe(wav_path, "en", initial_prompt=_HASS_PROMPT) == ""
    assert model._prefix_kv is cached

    # The next command must not prefill the prompt again: every decoder pass
    # should continue from cached KV rather than start from an empty one.
    from_scratch = []
    run_decoder = model._run_decoder

    def watching(input_embeds, positions, past_keys, past_values):
        if past_keys.shape[3] == 0:
            from_scratch.append(input_embeds.shape[1])
        return run_decoder(input_embeds, positions, past_keys, past_values)

    monkeypatch.setattr(model, "_run_decoder", watching)
    assert model.transcribe(command, "en", initial_prompt=_HASS_PROMPT) == expected
    assert from_scratch == []
