#!/usr/bin/env python3
"""Build a Qwen3-ASR model folder in the merged int4 layout this server loads.

Only the 0.6B model is published in that layout (rhasspy/qwen3-asr-0.6b-onnx-
int4-merged). This builds it for other sizes, e.g. 1.7B:

    python convert_qwen3_asr.py --size 1.7b --out local/qwen3-asr-1.7b-onnx-int4-merged
    wyoming-faster-whisper --model local/qwen3-asr-1.7b-onnx-int4-merged ...

Run by hand; nothing in the server calls it. It needs its own environment, as
it uses PyTorch (~1 GB) which the server does not:

    python3 -m venv .venv-convert
    .venv-convert/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    .venv-convert/bin/pip install "transformers>=5.13" onnx onnxruntime onnxscript \\
        safetensors huggingface_hub numpy

What goes into the folder:

* decoder_merged.int4.onnx (+ .data): exported from the PyTorch checkpoint
  Qwen/Qwen3-ASR-<size>-hf as one graph that takes a KV cache *and* any number
  of new tokens, which is what lets the server reuse the biasing prompt's KV.
  Quantized with the recipe of the published int4 decoders (RTN, block 64,
  asymmetric, accuracy level 4).
* encoder.int4.onnx (+ .data): andrewleech/qwen3-asr-<size>-onnx's audio
  encoder, unchanged. It is FP32 whatever its name says; only its weights are
  moved into the .data file the server expects. It is not quantized: int4 puts
  the audio features ~20% off the original model's, enough that 1.7B writes
  "incubi" for "Ecobee" with Ecobee in the prompt. (The published 0.6B model
  has an int4 encoder, so this does not reproduce it byte for byte.)
* embed_tokens.bin, tokenizer.json, config.json: from that repo, unchanged.

Resources for 1.7B: ~10 GB of downloads, ~2 minutes, and ~12-15 GB of RAM at
peak (0.6B: ~5 GB). Each phase runs in its own process so memory is returned
between them. Keep --work-dir off tmpfs (/tmp on many systems), whose files are
held in RAM.
"""

import argparse
import gc
import os
import resource
import shutil
import subprocess
import sys
import time

import numpy as np

_SIZES = {
    "0.6b": ("Qwen/Qwen3-ASR-0.6B-hf", "andrewleech/qwen3-asr-0.6b-onnx"),
    "1.7b": ("Qwen/Qwen3-ASR-1.7B-hf", "andrewleech/qwen3-asr-1.7b-onnx"),
}
# Taken unchanged from the ONNX repo.
_ONNX_FILES = ["encoder.int4.onnx", "embed_tokens.bin", "tokenizer.json", "config.json"]

_DECODER = "decoder_merged.int4.onnx"
_ENCODER = "encoder.int4.onnx"
_FP32 = os.path.join("fp32", "decoder_merged.onnx")
_REFERENCE = "reference.npz"
_INPUTS = ["input_embeds", "position_ids", "attention_mask", "past_keys", "past_values"]
_OUTPUTS = ["logits", "present_keys", "present_values"]

# A real prompt: the chat template's system and user turn openers and some text.
_PROMPT_IDS = [151644, 8948, 198, 151645, 198, 151644, 872, 198, 13, 2000, 3000, 4000]


def _peak(label: str) -> None:
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print(f"[{label}] peak RSS {rss_gb:.1f} GB", flush=True)


def _warn_if_tmpfs(path: str) -> None:
    """Files on tmpfs live in RAM, which is what the conversion needs."""
    path = os.path.realpath(path)
    best, fs_type = "", ""
    try:
        with open("/proc/mounts", "r", encoding="utf-8") as mounts:
            for line in mounts:
                _, mount_point, kind = line.split()[:3]
                if path.startswith(mount_point) and len(mount_point) > len(best):
                    best, fs_type = mount_point, kind
    except OSError:
        return

    if fs_type == "tmpfs":
        print(
            f"WARNING: {path} is on tmpfs, so ~10 GB of downloads and exports "
            "will be held in RAM. Use --work-dir on a disk.",
            file=sys.stderr,
            flush=True,
        )


# --- download --------------------------------------------------------------


def download(size: str, work_dir: str) -> None:
    from huggingface_hub import snapshot_download

    checkpoint, onnx_repo = _SIZES[size]
    for repo, patterns in ((checkpoint, None), (onnx_repo, _ONNX_FILES)):
        print(f"Downloading {repo}", flush=True)
        snapshot_download(
            repo, allow_patterns=patterns, local_dir=os.path.join(work_dir, repo)
        )


# --- export ----------------------------------------------------------------


def _modules():
    """The wrapper modules; defined here so torch is only imported by phases that need it."""
    import torch
    import torch.nn.functional as F
    from torch import nn
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

    def layer_forward(
        layer, hidden_states, cos, sin, mask, past_key, past_value, kv_groups
    ):
        """One Qwen3 decoder layer on plain tensors.

        The native layer keeps its KV in a Cache object, which tracing cannot
        follow, so the same computation is spelled out on stacked tensors. The
        result is checked against the native model in `check-native`.
        """
        attn = layer.self_attn
        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)
        shape = (*normed.shape[:-1], -1, attn.head_dim)

        query = attn.q_norm(attn.q_proj(normed).view(shape)).transpose(1, 2)
        key = attn.k_norm(attn.k_proj(normed).view(shape)).transpose(1, 2)
        value = attn.v_proj(normed).view(shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # The cache may be empty (a prefill); concatenating handles both.
        key = torch.cat([past_key, key], dim=2)
        value = torch.cat([past_value, value], dim=2)

        weights = (
            torch.matmul(query, repeat_kv(key, kv_groups).transpose(2, 3))
            * attn.scaling
        )
        weights = F.softmax(weights + mask, dim=-1, dtype=torch.float32).to(query.dtype)
        out = (
            torch.matmul(weights, repeat_kv(value, kv_groups))
            .transpose(1, 2)
            .contiguous()
        )
        hidden_states = residual + attn.o_proj(out.reshape(*normed.shape[:-1], -1))

        residual = hidden_states
        hidden_states = residual + layer.mlp(
            layer.post_attention_layernorm(hidden_states)
        )
        return hidden_states, key, value

    class DecoderMerged(nn.Module):
        """Prefill *and* step in one graph: any q_len, any past_len (including 0)."""

        def __init__(self, text_model, lm_head, text_config):
            super().__init__()
            self.layers = text_model.layers
            self.norm = text_model.norm
            self.rotary_emb = text_model.rotary_emb
            self.lm_head = lm_head
            self.kv_groups = (
                text_config.num_attention_heads // text_config.num_key_value_heads
            )

        def forward(
            self, input_embeds, position_ids, attention_mask, past_keys, past_values
        ):
            # Qwen3-ASR's text decoder has 3-stream MRoPE, but for audio-only
            # ASR all three streams carry the same positions, which is plain
            # RoPE -- what the -hf checkpoints' rotary module computes.
            cos, sin = self.rotary_emb(input_embeds, position_ids)
            hidden_states = input_embeds
            keys, values = [], []
            for i, layer in enumerate(self.layers):
                hidden_states, key, value = layer_forward(
                    layer,
                    hidden_states,
                    cos,
                    sin,
                    attention_mask,
                    past_keys[i],
                    past_values[i],
                    self.kv_groups,
                )
                keys.append(key)
                values.append(value)

            logits = self.lm_head(self.norm(hidden_states))
            return logits, torch.stack(keys), torch.stack(values)

    class PreTransposedLinear(nn.Module):
        """nn.Linear storing W^T, so it exports as MatMul(x, W^T).

        A plain Linear exports as MatMul(x, Transpose(W)), and constant folding
        then materializes W^T for every layer while W is still held: a second
        full copy of the model in RAM. Storing W^T up front avoids it.
        """

        def __init__(self, linear):
            super().__init__()
            self.weight_t = nn.Parameter(
                linear.weight.detach().t().contiguous(), requires_grad=False
            )
            bias = linear.bias
            self.bias = (
                None
                if bias is None
                else nn.Parameter(bias.detach().clone(), requires_grad=False)
            )

        def forward(self, x):
            y = torch.matmul(x, self.weight_t)
            return y if self.bias is None else y + self.bias

    return DecoderMerged, PreTransposedLinear


def _mask(q_len: int, past_len: int):
    """The additive mask the server builds (qwen3_asr_handler._causal_mask)."""
    import torch

    new = torch.triu(
        torch.full((q_len, q_len), torch.finfo(torch.float32).min), diagonal=1
    )
    return torch.cat([torch.zeros((q_len, past_len)), new], dim=-1)[None, None]


def _load(checkpoint_dir: str):
    import torch
    from transformers import Qwen3ASRForConditionalGeneration

    # bf16 is how the checkpoint is stored, so this is the smallest load.
    return Qwen3ASRForConditionalGeneration.from_pretrained(
        checkpoint_dir, dtype=torch.bfloat16, trust_remote_code=False
    ).eval()


def check_native(checkpoint_dir: str) -> None:
    """The wrapper against the native transformers model, in both cache modes."""
    import torch

    DecoderMerged, _ = _modules()
    model = _load(checkpoint_dir).float()
    text_model, config = model.model.language_model, model.config.text_config
    wrapper = DecoderMerged(text_model, model.lm_head, config).eval()

    n = len(_PROMPT_IDS)
    embeds = text_model.embed_tokens(torch.tensor([_PROMPT_IDS]))
    positions = torch.arange(n)[None]
    empty = torch.zeros(
        config.num_hidden_layers, 1, config.num_key_value_heads, 0, config.head_dim
    )
    with torch.no_grad():
        native = model.lm_head(
            text_model(
                inputs_embeds=embeds, position_ids=positions, use_cache=False
            ).last_hidden_state
        )
        one_shot, _, _ = wrapper(embeds, positions, _mask(n, 0), empty, empty)
        first, keys, values = wrapper(
            embeds[:, :7], positions[:, :7], _mask(7, 0), empty, empty
        )
        rest, _, _ = wrapper(
            embeds[:, 7:], positions[:, 7:], _mask(n - 7, 7), keys, values
        )

    scale = float(native.abs().max())
    for label, got in (
        ("one prefill", one_shot),
        ("prefill 7 + cached 5", torch.cat([first, rest], 1)),
    ):
        err = float((got - native).abs().max())
        agree = bool((got.argmax(-1) == native.argmax(-1)).all())
        print(
            f"  wrapper vs native, {label}: max|dlogits|={err:.1e} (scale {scale:.0f})",
            flush=True,
        )
        assert err < 1e-3 * scale and agree, label


def export(checkpoint_dir: str, work_dir: str) -> None:
    import torch
    from torch import nn

    DecoderMerged, PreTransposedLinear = _modules()

    t = time.monotonic()
    model = _load(checkpoint_dir)
    text_model, config = model.model.language_model, model.config.text_config
    embeds = text_model.embed_tokens.weight.detach()[_PROMPT_IDS].float()[None]

    # Keep only the decoder: no audio encoder, no embedding table (the server
    # looks embeddings up itself). lm_head is tied to the embedding table, so
    # it gets its own transposed copy before the model is dropped.
    lm_head = PreTransposedLinear(model.lm_head.to(torch.float32))
    wrapper = DecoderMerged(text_model, lm_head, config).eval()
    del model, text_model
    gc.collect()

    # Cast and transpose one Linear at a time, so at most one extra weight
    # exists at any moment (bf16 -> fp32 is exact).
    for module in list(wrapper.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, PreTransposedLinear(child.to(torch.float32)))
                del child
    wrapper.to(torch.float32)
    gc.collect()
    print(f"Loaded the decoder in {time.monotonic() - t:.0f}s", flush=True)

    # Reference outputs on real tokens and a real KV cache, for `check`.
    n = len(_PROMPT_IDS)
    empty = torch.zeros(
        config.num_hidden_layers, 1, config.num_key_value_heads, 0, config.head_dim
    )
    positions = torch.arange(n)[None]
    cases = {}
    reference = {}
    with torch.no_grad():
        cases["prefill"] = (embeds, positions, _mask(n, 0), empty, empty)
        _, keys, values = wrapper(
            embeds[:, :7], positions[:, :7], _mask(7, 0), empty, empty
        )
        cases["cached"] = (
            embeds[:, 7:],
            positions[:, 7:],
            _mask(n - 7, 7),
            keys,
            values,
        )
        _, keys, values = wrapper(
            embeds[:, :-1], positions[:, :-1], _mask(n - 1, 0), empty, empty
        )
        cases["step"] = (
            embeds[:, -1:],
            positions[:, -1:],
            _mask(1, n - 1),
            keys,
            values,
        )
        cases["step_empty_past"] = (
            embeds[:, :1],
            positions[:, :1],
            _mask(1, 0),
            empty,
            empty,
        )
        for case, inputs in cases.items():
            for name, value in zip(_INPUTS + _OUTPUTS, (*inputs, *wrapper(*inputs))):
                reference[f"{case}/{name}"] = value.numpy()
    np.savez(os.path.join(work_dir, _REFERENCE), **reference)

    fp32_path = os.path.join(work_dir, _FP32)
    os.makedirs(os.path.dirname(fp32_path), exist_ok=True)
    t = time.monotonic()
    with torch.no_grad():
        # Traced with q_len and past_len both > 1 and different, so neither is
        # frozen into the graph as a constant.
        torch.onnx.export(
            wrapper,
            cases["cached"],
            fp32_path,
            input_names=_INPUTS,
            output_names=_OUTPUTS,
            dynamic_axes={
                "input_embeds": {0: "batch", 1: "q_len"},
                "position_ids": {0: "batch", 1: "q_len"},
                "attention_mask": {2: "q_len", 3: "total_len"},
                "past_keys": {1: "batch", 3: "past_len"},
                "past_values": {1: "batch", 3: "past_len"},
                "logits": {0: "batch", 1: "q_len"},
                "present_keys": {1: "batch", 3: "total_len"},
                "present_values": {1: "batch", 3: "total_len"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"Exported the FP32 decoder in {time.monotonic() - t:.0f}s", flush=True)
    _peak("export")


# --- quantize --------------------------------------------------------------


def quantize(src: str, dst: str) -> None:
    """int4 with the published decoders' recipe (andrewleech/qwen3-asr-onnx quantize_nbits.py)."""
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        MatMulNBitsQuantizer,
        RTNWeightOnlyQuantConfig,
    )
    from onnxruntime.quantization.quant_utils import QuantFormat

    t = time.monotonic()
    quantizer = MatMulNBitsQuantizer(
        model=onnx.load(src),
        block_size=64,
        is_symmetric=False,
        accuracy_level=4,
        algo_config=RTNWeightOnlyQuantConfig(quant_format=QuantFormat.QOperator),
    )
    quantizer.process()

    model = quantizer.model.model
    if any(n.op_type == "MatMulNBits" for n in model.graph.node) and not any(
        o.domain == "com.microsoft" for o in model.opset_import
    ):
        opset = model.opset_import.add()
        opset.domain = "com.microsoft"
        opset.version = 1

    for name in (dst, dst + ".data"):
        if os.path.exists(name):
            os.remove(name)  # save_model_to_file appends to existing external data
    quantizer.model.save_model_to_file(dst, use_external_data_format=True)
    n_int4 = sum(1 for n in model.graph.node if n.op_type == "MatMulNBits")
    print(
        f"Quantized {os.path.basename(src)} in {time.monotonic() - t:.0f}s ({n_int4} int4 MatMuls)"
    )
    _peak("quantize")


def copy_encoder(src: str, dst: str) -> None:
    """Copy the encoder unchanged, with its weights in dst + ".data" as the server expects."""
    import onnx

    for name in (dst, dst + ".data"):
        if os.path.exists(name):
            os.remove(name)  # onnx appends to existing external data
    onnx.save_model(
        onnx.load(src),
        dst,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(dst) + ".data",
        size_threshold=1024,
    )
    print(f"Copied {os.path.basename(src)} (FP32)")
    _peak("encoder")


# --- check -----------------------------------------------------------------


def check(path: str, reference_path: str, strict: bool) -> None:
    """The ONNX graph against PyTorch, including shapes the trace never saw.

    With strict (the FP32 graph), outputs must match to float rounding. For the
    int4 graph this only shows the shapes work; whether int4 transcribes well
    is a question for real audio, not for these 12 tokens.
    """
    import onnxruntime as ort

    reference = np.load(reference_path)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for case in ("prefill", "cached", "step", "step_empty_past"):
        inputs = {name: reference[f"{case}/{name}"] for name in _INPUTS}
        got = dict(zip(_OUTPUTS, session.run(_OUTPUTS, inputs)))
        for name in _OUTPUTS:
            want = reference[f"{case}/{name}"]
            assert got[name].shape == want.shape, (
                case,
                name,
                got[name].shape,
                want.shape,
            )
        err = float(np.abs(got["logits"] - reference[f"{case}/logits"]).max())
        q_len, past = inputs["input_embeds"].shape[1], inputs["past_keys"].shape[3]
        print(
            f"  {os.path.basename(path)} {case} (q={q_len}, past={past}): max|dlogits|={err:.1e}"
        )
        if strict:
            assert err < 2e-3, case
    _peak("check")


# --- driver ----------------------------------------------------------------


def _phase(*argv: str) -> None:
    """Run one phase in a child process, so its memory goes back to the OS."""
    subprocess.run([sys.executable, os.path.abspath(__file__), *argv], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--size", choices=sorted(_SIZES), required=True)
    parser.add_argument("--out", required=True, help="model folder to write")
    parser.add_argument(
        "--work-dir",
        default=os.path.join("local", "convert"),
        help="downloads and intermediate files (default: local/convert)",
    )
    parser.add_argument(
        "--keep-fp32", action="store_true", help="keep the FP32 decoder"
    )
    parser.add_argument(
        "--check-fp32",
        action="store_true",
        help="also check the FP32 decoder against PyTorch (needs ~2x its size in RAM)",
    )
    parser.add_argument("--phase", help=argparse.SUPPRESS)  # internal
    args = parser.parse_args()

    checkpoint, onnx_repo = _SIZES[args.size]
    checkpoint_dir = os.path.join(args.work_dir, checkpoint)
    onnx_dir = os.path.join(args.work_dir, onnx_repo)
    work = os.path.join(args.work_dir, "out", args.size)
    fp32_path = os.path.join(work, _FP32)
    reference_path = os.path.join(work, _REFERENCE)

    if args.phase:
        {
            "check-native": lambda: check_native(checkpoint_dir),
            "export": lambda: export(checkpoint_dir, work),
            "check-fp32": lambda: check(fp32_path, reference_path, strict=True),
            "quantize": lambda: quantize(fp32_path, os.path.join(args.out, _DECODER)),
            "encoder": lambda: copy_encoder(
                os.path.join(onnx_dir, _ENCODER), os.path.join(args.out, _ENCODER)
            ),
            "check": lambda: check(
                os.path.join(args.out, _DECODER), reference_path, strict=False
            ),
        }[args.phase]()
        return

    _warn_if_tmpfs(args.work_dir)
    os.makedirs(work, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    common = ["--size", args.size, "--out", args.out, "--work-dir", args.work_dir]
    t = time.monotonic()

    download(args.size, args.work_dir)
    _phase(*common, "--phase", "check-native")
    _phase(*common, "--phase", "export")
    if args.check_fp32:
        _phase(*common, "--phase", "check-fp32")
    _phase(*common, "--phase", "quantize")
    if not args.keep_fp32:
        shutil.rmtree(os.path.dirname(fp32_path))
    _phase(*common, "--phase", "check")
    _phase(*common, "--phase", "encoder")
    for name in ("embed_tokens.bin", "tokenizer.json", "config.json"):
        shutil.copy2(os.path.join(onnx_dir, name), os.path.join(args.out, name))

    print(f"\nWrote {args.out} in {time.monotonic() - t:.0f}s:")
    for name in sorted(os.listdir(args.out)):
        size_mb = os.path.getsize(os.path.join(args.out, name)) / 1e6
        print(f"  {name:32s} {size_mb:9.1f} MB")


if __name__ == "__main__":
    main()
