"""Translation of the --device option into onnxruntime execution providers.

Accepted values are "cpu", "cuda", and "cuda:N". Anything else is treated as CPU.
Kept free of an onnxruntime import so the rules are testable on their own.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

_LOGGER = logging.getLogger(__name__)

# Execution provider entries are either a name or (name, options).
OnnxProvider = Union[str, Tuple[str, Dict[str, Any]]]

_CPU_PROVIDER = "CPUExecutionProvider"
_CUDA_PROVIDER = "CUDAExecutionProvider"


def is_gpu(device: str) -> bool:
    """Report whether a device string names a CUDA GPU."""
    return device.strip().lower().startswith("cuda")


def device_index(device: str) -> Optional[int]:
    """Extract the GPU ordinal from "cuda:N", or None if unspecified."""
    _, _, index = device.strip().lower().partition(":")
    if not index:
        return None

    try:
        return int(index)
    except ValueError:
        _LOGGER.warning("Ignoring unparsable device index in '%s'", device)
        return None


def onnx_providers(device: str) -> List[OnnxProvider]:
    """Return the onnxruntime execution providers to try, in order.

    The CPU provider is always kept as the last entry so a container without a
    usable driver degrades to CPU inference instead of failing to load a model.
    """
    if not is_gpu(device):
        return [_CPU_PROVIDER]

    index = device_index(device)
    if index is None:
        return [_CUDA_PROVIDER, _CPU_PROVIDER]

    return [(_CUDA_PROVIDER, {"device_id": index}), _CPU_PROVIDER]


def warn_if_no_onnx_gpu(device: str, providers: Sequence[str]) -> bool:
    """Warn when a GPU was asked for but onnxruntime is not using CUDA.

    Returns whether the GPU is in use.

    Pass a live session's ``get_providers()`` rather than
    ``ort.get_available_providers()``: the latter lists the CUDA provider as
    available even when its shared library cannot be loaded (the usual cause
    being an onnxruntime-gpu built for a different CUDA major version than the
    one installed), in which case sessions silently fall back to the CPU. Either
    way the failure is quiet - the model still loads and transcribes, just not
    on the GPU - so it is worth a warning rather than leaving it to be noticed as
    "the GPU image is no faster".
    """
    if not is_gpu(device):
        return False

    if _CUDA_PROVIDER in providers:
        return True

    _LOGGER.warning(
        "Device '%s' was requested but onnxruntime is not using %s "
        "(in effect: %s), so inference will run on the CPU. Either "
        "onnxruntime-gpu is not installed, or its CUDA/cuDNN requirements are "
        "not met here - check the onnxruntime errors logged above.",
        device,
        _CUDA_PROVIDER,
        ", ".join(providers) or "none",
    )
    return False
