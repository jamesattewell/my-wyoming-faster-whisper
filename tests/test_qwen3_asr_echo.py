"""Unit tests for the Qwen3-ASR echo guard, against a scripted fake decoder.

The guard has two jobs -- tell an echo of the biasing prompt (the names read
back in order, however the model cases and spaces them) from a command that
mentions a name or two, and stop decoding early without disturbing the cached KV
of the prompt -- and neither needs real weights. The fake decoder emits a scripted
token sequence, so each test states exactly what the model "says". See
test_qwen3_asr_noise for the same behaviour on the real model.
"""

import re
import wave
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pytest

qwen3_asr_handler = pytest.importorskip(
    "wyoming_faster_whisper.qwen3_asr_handler",
    reason="qwen3-asr extra not installed",
)

from .test_qwen3_asr import _fake_model_dir, _FakeSession  # noqa: E402

_COMMA = 11
_ASR_TEXT = qwen3_asr_handler._ASR_TEXT
_EOS = 151645
_N = qwen3_asr_handler._ECHO_MIN_NAMES
_normalize = qwen3_asr_handler._normalize
_echo_patterns = qwen3_asr_handler._echo_patterns
_VOCAB = 151710
_N_AUDIO = 4

_PROMPT = "Office, Study, Upstairs, Top Floor, Office Lamp, Desk Lamp, Kitchen."


# --- fakes -----------------------------------------------------------------


class _FakeTokenizer:
    """Words and punctuation, one id each."""

    def __init__(self) -> None:
        self._ids: Dict[str, int] = {",": _COMMA}
        self._text: Dict[int, str] = {_COMMA: ","}

    def _id(self, piece: str) -> int:
        if piece not in self._ids:
            new_id = 1000 + len(self._ids)
            self._ids[piece] = new_id
            self._text[new_id] = piece
        return self._ids[piece]

    def encode(self, text: str, add_special_tokens: bool = False):
        pieces = re.findall(r"[,.?\uff0c]|\s?[^\s,.?\uff0c]+", text)
        return SimpleNamespace(ids=[self._id(piece) for piece in pieces])

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return "".join(self._text.get(i, "") for i in ids if i < 151643)


class _ScriptedDecoder:
    """Stands in for _run_decoder: says ``tokens``, one per step, then EOS."""

    def __init__(self, tokens: List[int]) -> None:
        self.tokens = tokens
        self.calls: List[SimpleNamespace] = []
        self._prompt_len = 0

    def __call__(self, input_embeds, positions, past_keys, past_values):
        positions = np.asarray(positions)
        q_len = input_embeds.shape[1]
        past = past_keys.shape[3]
        self.calls.append(SimpleNamespace(q_len=q_len, past=past))

        # A multi-token call is a prefill; where it ends is where output starts.
        if q_len > 1:
            self._prompt_len = past + q_len

        step = int(positions[-1]) - (self._prompt_len - 1)
        logits = np.zeros((1, q_len, _VOCAB), dtype=np.float32)
        logits[0, -1, self.tokens[step] if step < len(self.tokens) else _EOS] = 1.0

        kv = np.zeros((4, 1, 2, past + q_len, 8), dtype=np.float32)
        return logits, kv, kv

    @property
    def steps(self) -> int:
        """Single-token decoder passes, i.e. tokens generated after the first."""
        return sum(1 for call in self.calls if call.q_len == 1)

    @property
    def prefills_from_scratch(self) -> int:
        return sum(1 for call in self.calls if call.past == 0)


@pytest.fixture(name="model")
def model_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(qwen3_asr_handler.ort, "InferenceSession", _FakeSession)
    monkeypatch.setattr(
        qwen3_asr_handler,
        "Tokenizer",
        SimpleNamespace(from_file=lambda path: _FakeTokenizer()),
    )
    monkeypatch.setattr(
        qwen3_asr_handler.np,
        "fromfile",
        lambda *args, **kwargs: np.zeros(16, dtype=np.float16),
    )
    model = qwen3_asr_handler.Qwen3AsrTranscriber(
        str(_fake_model_dir(tmp_path)), str(tmp_path)
    )
    model._embed_tokens = np.zeros((_VOCAB, 8), dtype=np.float16)
    model._encoder = SimpleNamespace(
        run=lambda outputs, inputs: [np.zeros((1, _N_AUDIO, 8), dtype=np.float32)]
    )
    return model


@pytest.fixture(name="wav_path")
def wav_path_fixture(tmp_path):
    path = tmp_path / "audio.wav"
    wav_file: wave.Wave_write = wave.open(str(path), "wb")
    with wav_file:
        wav_file.setframerate(16000)
        wav_file.setsampwidth(2)
        wav_file.setnchannels(1)
        wav_file.writeframes(bytes(2 * 16000))
    return path


def _ids(model, text: str) -> List[int]:
    return model._tokenizer.encode(text).ids


def _say(model, text: str) -> _ScriptedDecoder:
    """Make the fake model say ``text`` for the next utterance."""
    decoder = _ScriptedDecoder(_ids(model, text) + [_EOS])
    model._run_decoder = decoder
    return decoder


# --- _normalize / _echo_patterns -----------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Office, Study, Upstairs.", "office,study,upstairs"),
        # How the 1.7B model echoes: its own casing and word splits.
        (
            "The office, study, up stairs, top floor",
            "theoffice,study,upstairs,topfloor",
        ),
        ("Living Room, Light", "livingroom,light"),
        ("办公室，书房、厨房", "办公室,书房,厨房"),  # CJK separators
        ("What's the 2nd one?", "whatsthe2ndone"),
    ],
)
def test_normalize(text, expected) -> None:
    assert _normalize(text) == expected


def test_patterns_are_runs_of_consecutive_names() -> None:
    patterns = _echo_patterns("Office, Study, Upstairs, Top Floor.")
    assert patterns == {"office,study,upstairs", "study,upstairs,topfloor"}


def test_a_prompt_prefix_stays_part_of_the_first_name() -> None:
    patterns = _echo_patterns("Vocabulary: Office, Study, Upstairs, Top Floor.")
    assert "vocabularyoffice,study,upstairs" in patterns
    assert "study,upstairs,topfloor" in patterns  # so an echo is still caught


def test_a_short_prompt_is_matched_as_a_whole() -> None:
    assert _echo_patterns("Kitchen, Office.") == {"kitchen,office"}


@pytest.mark.parametrize("prompt", [None, "", "Ecobee.", " , ."])
def test_a_prompt_that_is_not_a_list_has_no_patterns(prompt) -> None:
    assert _echo_patterns(prompt) == frozenset()


# --- transcribe: not an echo -----------------------------------------------


@pytest.mark.parametrize(
    "said",
    [
        "Turn on the Office Lamp.",
        "Office Lamp.",  # just a name from the list
        "Office, Study.",  # two names in prompt order: too short to be an echo
        "Turn on Upstairs and Top Floor.",
        "Kitchen, Study, Office.",  # three names, but not in the prompt's order
    ],
)
def test_speech_is_returned_unchanged(model, wav_path, said) -> None:
    _say(model, said)
    assert model.transcribe(wav_path, "en", initial_prompt=_PROMPT) == said


def test_without_a_prompt_nothing_is_an_echo(model, wav_path) -> None:
    _say(model, _PROMPT)
    assert model.transcribe(wav_path, "en") == _PROMPT


def test_a_different_prompt_is_not_an_echo(model, wav_path) -> None:
    # The same words, but not what this utterance was biased with.
    _say(model, _PROMPT)
    assert model.transcribe(wav_path, "en", initial_prompt="Ecobee.") == _PROMPT


# --- transcribe: echo ------------------------------------------------------


def test_an_echo_of_the_prompt_gives_an_empty_transcript(model, wav_path) -> None:
    _say(model, _PROMPT)
    assert model.transcribe(wav_path, "en", initial_prompt=_PROMPT) == ""


def test_an_echo_of_the_middle_of_the_prompt_is_caught(model, wav_path) -> None:
    _say(model, "Upstairs, Top Floor, Office Lamp, Desk Lamp.")
    assert model.transcribe(wav_path, "en", initial_prompt=_PROMPT) == ""


def test_an_echo_after_real_words_is_caught(model, wav_path) -> None:
    # Speech that trails off into the list is still not a usable command.
    _say(model, "Turn on the" + _PROMPT)
    assert model.transcribe(wav_path, "en", initial_prompt=_PROMPT) == ""


@pytest.mark.parametrize(
    "echo",
    [
        # How the 1.7B model echoes the list.
        "Office, study, upstairs, top floor, office lamp, desk lamp, kitchen.",
        "The office, study, up stairs, top floor, office lamp.",
        "OFFICE, STUDY, UPSTAIRS.",
        "Office\uff0cStudy\uff0cUpstairs.",  # fullwidth commas
    ],
)
def test_an_echo_in_the_models_own_spelling_is_caught(model, wav_path, echo) -> None:
    _say(model, echo)
    assert model.transcribe(wav_path, "en", initial_prompt=_PROMPT) == ""


def test_an_echo_of_a_short_prompt_is_caught(model, wav_path) -> None:
    _say(model, "Kitchen, office.")
    assert model.transcribe(wav_path, "en", initial_prompt="Kitchen, Office.") == ""


def test_a_name_from_a_short_prompt_is_not_an_echo(model, wav_path) -> None:
    _say(model, "Turn on the kitchen.")
    assert (
        model.transcribe(wav_path, "en", initial_prompt="Kitchen, Office.")
        == "Turn on the kitchen."
    )


def test_an_echo_after_the_language_preamble_is_caught(model, wav_path) -> None:
    # Language not forced: the model writes "language English<asr_text>" first.
    decoder = _ScriptedDecoder(
        _ids(model, "language English") + [_ASR_TEXT] + _ids(model, _PROMPT) + [_EOS]
    )
    model._run_decoder = decoder
    assert model.transcribe(wav_path, None, initial_prompt=_PROMPT) == ""


def test_an_echo_is_stopped_once_the_names_line_up(model, wav_path) -> None:
    echo = _ids(model, _PROMPT)
    # The first prefix of the output that reads back _N names in a row.
    patterns = _echo_patterns(_PROMPT)
    caught_at = next(
        k
        for k in range(1, len(echo) + 1)
        if any(p in _normalize(model._tokenizer.decode(echo[:k])) for p in patterns)
    )
    assert caught_at < len(echo) // 2

    decoder = _say(model, _PROMPT)
    model.transcribe(wav_path, "en", initial_prompt=_PROMPT)
    # That many tokens are generated (the first by the prefill, the rest by
    # steps); the rest of the list is never read out.
    assert decoder.steps == caught_at - 1


# --- the prefix cache ------------------------------------------------------


def test_an_echo_leaves_the_prompts_cached_kv_in_place(model, wav_path) -> None:
    context, _ = model._context_ids(_PROMPT, "en")
    system_key = tuple(qwen3_asr_handler._system_ids(context))

    _say(model, _PROMPT)
    model.transcribe(wav_path, "en", initial_prompt=_PROMPT)

    assert model._prefix_key == system_key
    assert model._prefix_kv is not None


def test_the_utterance_after_an_echo_reuses_the_cached_prompt(model, wav_path) -> None:
    first = _say(model, _PROMPT)
    model.transcribe(wav_path, "en", initial_prompt=_PROMPT)
    assert first.prefills_from_scratch == 1  # the system turn, once
    cached = model._prefix_kv

    second = _say(model, "Turn on the Kitchen.")
    assert (
        model.transcribe(wav_path, "en", initial_prompt=_PROMPT)
        == "Turn on the Kitchen."
    )
    assert second.prefills_from_scratch == 0  # nothing re-read
    assert model._prefix_kv is cached
