"""Tests for the Qwen3-ASR prompt/front-end helpers.

These cover the pure logic only: the chat template the decoder is prompted
with, and the mel front-end. tests/test_e2e.py runs the real model.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wyoming_faster_whisper import qwen3_asr_handler

_build_prompt_ids = qwen3_asr_handler.Qwen3AsrTranscriber._build_prompt_ids
_causal_mask = qwen3_asr_handler._causal_mask
_log_mel_spectrogram = qwen3_asr_handler._log_mel_spectrogram
_mel_filterbank = qwen3_asr_handler._mel_filterbank
_system_ids = qwen3_asr_handler._system_ids

_AUDIO_PAD = qwen3_asr_handler._AUDIO_PAD
_BLOCKED = qwen3_asr_handler._MASK_BLOCKED


# --- prompt construction ---------------------------------------------------


def test_prompt_has_one_audio_pad_per_encoder_frame() -> None:
    ids = _build_prompt_ids(None, 7, [], [])
    assert ids.count(_AUDIO_PAD) == 7


def test_prompt_matches_chat_template_layout() -> None:
    ids = _build_prompt_ids(None, 1, [], [])
    # <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
    assert ids[:9] == [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669]
    # ...<|audio_end|><|im_end|>\n<|im_start|>assistant\n
    assert ids[10:] == [151670, 151645, 198, 151644, 77091, 198]


def test_context_is_placed_in_the_system_turn() -> None:
    context = [1111, 2222]
    ids = _build_prompt_ids(None, 1, context, [])
    # Context sits after "<|im_start|>system\n" and before "<|im_end|>".
    assert ids[:3] == [151644, 8948, 198]
    assert ids[3:5] == context
    assert ids[5] == 151645


def test_forced_language_suffix_goes_after_the_assistant_turn() -> None:
    language_ids = [3333, qwen3_asr_handler._ASR_TEXT]
    ids = _build_prompt_ids(None, 1, [], language_ids)
    assert ids[-2:] == language_ids
    assert ids[-3] == 198  # newline closing "<|im_start|>assistant\n"


# --- merged decoder: the cached span ---------------------------------------


def test_system_turn_is_the_prompts_prefix() -> None:
    """The cached span must be a literal prefix of the prompt, or reusing its KV
    would silently attend to the wrong positions."""
    context = [1111, 2222]
    ids = _build_prompt_ids(None, 3, context, [])
    system = _system_ids(context)
    assert ids[: len(system)] == system


def test_system_turn_holds_the_context_and_no_audio() -> None:
    system = _system_ids([1111, 2222])
    assert system == [151644, 8948, 198, 1111, 2222, 151645, 198]
    # Nothing about the audio may be inside the cached span.
    assert _AUDIO_PAD not in system


def test_system_turn_length_tracks_the_context() -> None:
    assert len(_system_ids([])) == 5
    assert len(_system_ids([1, 2, 3])) == 8


# --- merged decoder: attention mask ----------------------------------------


def test_mask_is_causal_when_there_is_no_cache() -> None:
    mask = _causal_mask(3, 0)
    assert mask.shape == (1, 1, 3, 3)
    # Row i may attend to columns <= i.
    assert (mask[0, 0] == 0).tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]


def test_mask_allows_every_cached_position() -> None:
    mask = _causal_mask(2, 4)
    assert mask.shape == (1, 1, 2, 6)
    # The four cached positions are open to both new tokens...
    assert (mask[0, 0, :, :4] == 0).all()
    # ...and the new tokens stay causal among themselves.
    assert mask[0, 0, 0, 4] == 0
    assert mask[0, 0, 0, 5] == _BLOCKED
    assert mask[0, 0, 1, 4] == 0
    assert mask[0, 0, 1, 5] == 0


def test_single_token_step_attends_to_everything() -> None:
    mask = _causal_mask(1, 7)
    assert mask.shape == (1, 1, 1, 8)
    assert (mask == 0).all()


def test_mask_blocks_with_a_large_negative_and_is_float32() -> None:
    mask = _causal_mask(2, 0)
    assert mask.dtype == np.float32
    assert mask[0, 0, 0, 1] == _BLOCKED
    assert _BLOCKED < -1e30


# --- model file selection --------------------------------------------------


class _FakeSession:
    """Enough of an ORT session to get through the constructor."""

    def __init__(self, path, **kwargs) -> None:
        self.path = str(path)
        self.providers = list(kwargs.get("providers") or ["CPUExecutionProvider"])

    def get_providers(self):
        # The real session reports what it actually ended up using, which is how
        # the handler detects a CUDA provider that failed to load.
        return self.providers

    def get_inputs(self):
        return [SimpleNamespace(name="mel")]

    def get_outputs(self):
        # logits, then present_keys/present_values shaped
        # [num_layers, batch, kv_heads, seq, head_dim] with the batch and
        # sequence dims symbolic, as the real export has them.
        kv = SimpleNamespace(name="present_keys", shape=[4, "batch", 2, "seq", 8])
        return [SimpleNamespace(name="logits", shape=["batch", "q", 99]), kv, kv]


def _write_files(directory, names) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"")


def _fake_model_dir(tmp_path):
    """A model directory with the right filenames and no real weights."""
    path = tmp_path / "model"
    _write_files(path, qwen3_asr_handler._MODEL_FILES)
    (path / "config.json").write_text(json.dumps({"decoder": {"hidden_size": 8}}))
    return path


@pytest.fixture(name="stub_runtime")
def stub_runtime_fixture(monkeypatch):
    monkeypatch.setattr(qwen3_asr_handler.ort, "InferenceSession", _FakeSession)
    monkeypatch.setattr(
        qwen3_asr_handler, "Tokenizer", SimpleNamespace(from_file=lambda path: object())
    )
    monkeypatch.setattr(
        qwen3_asr_handler.np,
        "fromfile",
        lambda *args, **kwargs: np.zeros(16, dtype=np.float16),
    )


def test_empty_kv_is_shaped_from_the_graph(tmp_path, stub_runtime) -> None:
    # Taken from present_keys so the cache does not depend on config.json.
    model = qwen3_asr_handler.Qwen3AsrTranscriber(
        str(_fake_model_dir(tmp_path)), str(tmp_path)
    )
    assert model._empty_kv.shape == (4, 1, 2, 0, 8)
    assert model._empty_kv.dtype == np.float32


def test_prefix_cache_starts_empty(tmp_path, stub_runtime) -> None:
    model = qwen3_asr_handler.Qwen3AsrTranscriber(
        str(_fake_model_dir(tmp_path)), str(tmp_path)
    )
    assert model._prefix_key is None
    assert model._prefix_kv is None


def test_only_the_int4_merged_files_are_downloaded() -> None:
    assert set(qwen3_asr_handler._MODEL_FILES) == {
        "config.json",
        "tokenizer.json",
        "embed_tokens.bin",
        "encoder.int4.onnx",
        "encoder.int4.onnx.data",
        "decoder_merged.int4.onnx",
        "decoder_merged.int4.onnx.data",
    }


# --- finding and downloading the model -------------------------------------

_REPO = "rhasspy/qwen3-asr-0.6b-onnx-int4-merged"


@pytest.fixture(name="hub")
def hub_fixture(monkeypatch):
    """Record snapshot_download calls; "download" by writing the model files."""
    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append({"repo_id": repo_id, **kwargs})
        local_dir = Path(kwargs["local_dir"])
        _write_files(local_dir, qwen3_asr_handler._MODEL_FILES)
        return str(local_dir)

    monkeypatch.setattr(qwen3_asr_handler, "snapshot_download", fake_snapshot_download)
    return calls


def test_a_directory_is_used_as_is(tmp_path, hub) -> None:
    model = tmp_path / "my-model"
    _write_files(model, qwen3_asr_handler._MODEL_FILES)

    assert qwen3_asr_handler._model_dir(str(model), tmp_path / "data", False) == model
    assert not hub


def test_a_split_export_directory_is_rejected(tmp_path, hub) -> None:
    """The older decoder_init/decoder_step export is no longer supported."""
    model = tmp_path / "split-model"
    _write_files(
        model,
        [
            "config.json",
            "tokenizer.json",
            "embed_tokens.bin",
            "encoder.int4.onnx",
            "encoder.int4.onnx.data",
            "decoder_init.int4.onnx",
            "decoder_step.int4.onnx",
            "decoder_weights.int4.data",
        ],
    )

    with pytest.raises(FileNotFoundError, match="decoder_merged.int4.onnx"):
        qwen3_asr_handler._model_dir(str(model), tmp_path / "data", False)

    assert not hub


def test_a_repo_is_downloaded_into_a_plain_folder(tmp_path, hub) -> None:
    """Not the Hugging Face cache: its symlinks put each .onnx file and its
    .data file in different directories, which onnxruntime refuses to load."""
    model_dir = qwen3_asr_handler._model_dir(_REPO, tmp_path, False)

    assert (
        model_dir == tmp_path.resolve() / "rhasspy" / "qwen3-asr-0.6b-onnx-int4-merged"
    )
    assert len(hub) == 1
    assert hub[0]["local_dir"] == str(model_dir)
    assert hub[0]["allow_patterns"] == list(qwen3_asr_handler._MODEL_FILES)
    assert "cache_dir" not in hub[0]


def test_a_downloaded_model_is_loaded_without_the_hub(tmp_path, hub) -> None:
    qwen3_asr_handler._model_dir(_REPO, tmp_path, False)
    qwen3_asr_handler._model_dir(_REPO, tmp_path, False)

    assert len(hub) == 1, "the second start must not contact the hub"


def test_a_missing_file_is_downloaded(tmp_path, hub) -> None:
    # An interrupted download: everything but the decoder's weights.
    local_dir = tmp_path / "rhasspy" / "qwen3-asr-0.6b-onnx-int4-merged"
    _write_files(
        local_dir,
        [
            name
            for name in qwen3_asr_handler._MODEL_FILES
            if name != "decoder_merged.int4.onnx.data"
        ],
    )

    qwen3_asr_handler._model_dir(_REPO, tmp_path, False)

    assert len(hub) == 1


def test_a_repo_without_the_merged_decoder_is_rejected(tmp_path, monkeypatch) -> None:
    """The split repo "downloads" fine (non-matching patterns are skipped), so
    the missing decoder must be caught here rather than inside onnxruntime."""

    def fake_snapshot_download(repo_id, **kwargs):
        _write_files(
            Path(kwargs["local_dir"]),
            [
                "config.json",
                "tokenizer.json",
                "embed_tokens.bin",
                "encoder.int4.onnx",
                "encoder.int4.onnx.data",
            ],
        )
        return kwargs["local_dir"]

    monkeypatch.setattr(qwen3_asr_handler, "snapshot_download", fake_snapshot_download)

    with pytest.raises(FileNotFoundError, match="decoder_merged.int4.onnx"):
        qwen3_asr_handler._model_dir(
            "rhasspy/qwen3-asr-0.6b-onnx-int4", tmp_path, False
        )


@pytest.mark.parametrize("model", ["/no/such/model", "./no-such-model"])
def test_a_missing_directory_is_not_sent_to_the_hub(tmp_path, hub, model) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        qwen3_asr_handler._model_dir(model, tmp_path, False)

    assert not hub


def test_local_files_only_never_downloads(tmp_path, hub) -> None:
    with pytest.raises(FileNotFoundError):
        qwen3_asr_handler._model_dir(_REPO, tmp_path, True)

    assert not hub


def test_local_files_only_loads_a_downloaded_model(tmp_path, hub) -> None:
    qwen3_asr_handler._model_dir(_REPO, tmp_path, False)

    assert qwen3_asr_handler._model_dir(_REPO, tmp_path, True)
    assert len(hub) == 1


# --- mel front-end ---------------------------------------------------------


def test_mel_filterbank_shape_and_normalization() -> None:
    filters = _mel_filterbank()
    assert filters.shape == (128, (400 // 2) + 1)
    assert filters.dtype == np.float32
    assert (filters >= 0).all()
    # Every filter carries some weight (no empty bands).
    assert (filters.sum(axis=1) > 0).all()


def test_log_mel_frame_count_and_range() -> None:
    # 1 second of 16kHz audio -> 100 frames (10ms hop), minus the dropped frame.
    audio = np.zeros(16000, dtype=np.float32)
    mel = _log_mel_spectrogram(audio, _mel_filterbank())
    assert mel.shape == (1, 128, 100)
    assert mel.dtype == np.float32
    # The log scale is clamped to an 8-decade window then rescaled.
    assert mel.max() - mel.min() <= 2.0 + 1e-6
