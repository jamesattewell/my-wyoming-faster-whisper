"""Tests for --device translation.

Dependency-free: the rules can be checked without onnxruntime or a GPU.
"""

import pytest

from wyoming_faster_whisper.device import (
    device_index,
    is_gpu,
    onnx_providers,
    warn_if_no_onnx_gpu,
)

# --- is_gpu ----------------------------------------------------------------


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:3", "CUDA", " cuda "])
def test_cuda_is_gpu(device) -> None:
    assert is_gpu(device)


@pytest.mark.parametrize("device", ["cpu", "CPU", "auto", "mps", "xpu", ""])
def test_everything_else_is_not_gpu(device) -> None:
    # Only CUDA counts. "auto" deliberately reads as non-GPU so onnxruntime
    # gets a provider list that works rather than one that might not.
    assert not is_gpu(device)


# --- device index ----------------------------------------------------------


def test_index_is_none_when_unspecified() -> None:
    assert device_index("cuda") is None
    assert device_index("cpu") is None


def test_index_is_parsed() -> None:
    assert device_index("cuda:0") == 0
    assert device_index("cuda:2") == 2


def test_unparsable_index_is_ignored() -> None:
    assert device_index("cuda:first") is None


# --- onnxruntime providers -------------------------------------------


def test_onnx_providers_on_cpu() -> None:
    assert onnx_providers("cpu") == ["CPUExecutionProvider"]


def test_onnx_providers_keep_cpu_fallback() -> None:
    # A container without a usable driver must degrade to CPU rather than fail
    # to load the model at all.
    providers = onnx_providers("cuda")
    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_onnx_providers_carry_the_device_id() -> None:
    providers = onnx_providers("cuda:2")
    assert providers == [
        ("CUDAExecutionProvider", {"device_id": 2}),
        "CPUExecutionProvider",
    ]


# --- silent-CPU-fallback warning ------------------------------------------


def test_no_warning_when_cpu_was_asked_for(caplog) -> None:
    assert not warn_if_no_onnx_gpu("cpu", ["CPUExecutionProvider"])
    assert not caplog.records


def test_gpu_available_is_not_warned_about(caplog) -> None:
    assert warn_if_no_onnx_gpu(
        "cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    assert not caplog.records


def test_missing_cuda_provider_warns(caplog) -> None:
    # The whole point: onnxruntime loads the model on the CPU provider without
    # complaint, so the only symptom would be no speedup. This fires both when
    # the CPU package is installed and when a session falls back because the
    # CUDA provider library could not be loaded.
    assert not warn_if_no_onnx_gpu("cuda", ["CPUExecutionProvider"])
    assert len(caplog.records) == 1
    assert "onnxruntime-gpu" in caplog.records[0].getMessage()
