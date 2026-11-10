# TODO: Qwen3-ASR-only refactor

## Where things stand

- Branch `feature/qwenasr-only`. All changes are **staged, not committed**.
- `.venv/` is set up (`script/setup --dev --hass`). The default model is
  already downloaded to `local/rhasspy/qwen3-asr-0.6b-onnx-int4-merged`.
  Both are gitignored.
- Last full run: 221 tests passed (including 22 end-to-end tests on the real
  model), and `script/lint` passes.
- Both Docker images are built and tested (section 1). Not tested yet: the
  GitHub Actions workflows, and CUDA on a real GPU.

## 1. ~~Build and smoke-test the Docker images~~ (done)

- **CPU image:** 570 MB. Started in 6 s from the mounted model with no download,
  and Docker's `HEALTHCHECK` reports it healthy. Through the container: the
  44.1 kHz lamp recording (so resampling works), French, `zh-HK` → Cantonese,
  and German auto-detected all transcribed correctly.
- **GPU image:** 8.41 GB, with `onnxruntime-gpu` 1.26.0 as the only onnxruntime.
  On the CPU it transcribes correctly, and biasing works ("Ecobee"). With
  `--device cuda` and no GPU it warns and falls back to the CPU. CUDA itself is
  untested: this machine has no NVIDIA GPU.
- Cleaned up afterwards. Docker is back to its starting state.

## 2. Delete temporary files outside the project

The permission check blocked these. This session created all of them:

```sh
rm -rf ~/.cache/huggingface /tmp/pytest-of-james /tmp/tmp_frcfvbl
rm -f ~/.cache/Microsoft/DeveloperTools/.onnxruntime/onnxruntime.db*
```

`~/.cache/huggingface` is about 650 KB of download scratch data. Later model
downloads may recreate a small amount of it.

`onnxruntime.db` is a small file onnxruntime 1.31 writes on every run. Keep
the rest of `~/.cache/Microsoft`: it predates this work and belongs to other
tools (Foundry Local, onnxruntime-genai).

## 3. Decisions needed

- [ ] **Nine recordings no test has ever used:** `nevermind.wav`, six
  `set-volume-to-*.wav`, `turn-off-the-lights-please.wav` and
  `turn-on-the-office-lamp.wav`. They came with the name-biasing work (commit
  `5d3ee58`). Delete them, or turn them into tests?
- [ ] **The `github_release` job in `publish.yml`** is still enabled. On a
  `v*.*.*` tag it creates a Release on this fork with the wheel and sdist. The
  Docker Hub and PyPI pushes are commented out. Keep it or disable it?

## 4. Then

- [ ] Commit the staged changes (and open a PR into `main` if wanted).
- [ ] Push and check that the `Test` workflow passes on GitHub. In particular,
  check that the model cache is saved on the first run and restored on the
  second.
- [ ] Before a release: give the CHANGELOG's `Unreleased` section a version.
  The changes are breaking, so 4.0.0 would fit. Update `version` in
  `pyproject.toml` to match.

## 5. Keep the decoder's KV cache on the GPU

- [ ] Use onnxruntime IO binding in the Qwen3-ASR handler so that, with
  `--device cuda`, the KV cache stays on the GPU between tokens.

### The problem

`_run_decoder` ([qwen3_asr_handler.py:392](wyoming_faster_whisper/qwen3_asr_handler.py#L392))
calls `InferenceSession.run()` with numpy arrays and gets numpy arrays back.
Numpy arrays live in system RAM. With the CUDA provider, every call:

1. copies every input to the GPU, including `past_keys` / `past_values` (the
   whole cache so far);
2. runs the graph;
3. copies every output back to RAM: `present_keys` / `present_values` (the
   cache again, one position longer) and the logits.

`_generate` feeds those numpy arrays straight back in for the next token. So
the whole cache crosses PCIe twice per generated token, and the copies grow as
the output gets longer. The cached system-turn KV (`_prefix_kv`) and
`_empty_kv` are numpy too, so they are uploaded again on every command.

On the CPU this costs almost nothing (no PCIe), which is why it never showed up
in the measurements in PERFORMANCE.md.

**Where it came from:** upstream `main`, not our changes:
- `c014311` (Michael Hansen) added the merged decoder and the prompt KV cache,
  written CPU-first;
- `98791e3` added `--device` and the GPU image without changing how the
  decoder is called.

Upstream's split-decoder path (`decoder_init` / `decoder_step`) has the same
pattern. Our echo-guard commit does not touch it.

### How much is copied

- **KV per position:** 28 layers × 8 KV heads × 128 head dim × 2 (keys and
  values) × 4 bytes (FP32) = **229,376 bytes**. It is the same for the 0.6B and
  the 1.7B: the two decoders differ in width, not in KV layout.
- **A typical command** (3 s of audio, measured median over the 17 test
  recordings, with `local/convert/profile_parts.py`):
  - 86 prompt positions: 37 for the cached system turn, 49 for the audio and
    assistant preamble;
  - about 10 decode steps.
- **Per decode step:** about 20.6 MB in plus 20.9 MB out, so **~42 MB per
  token**.
- **Prompt pass:**
  - 8.5 MB in (the cached 37 positions);
  - 19.7 MB out (86 positions);
  - 29.8 MB of logits out: all 49 positions × 151,936 × 4 bytes, of which only
    the last row is used.
- **A 256-token repetition loop on noise:** the copy grows to about 157 MB per
  token by the end.

### Expected gain (estimated, not measured)

These figures are for an RTX 5060 Ti 16 GB, worked out from this machine's CPU
measurements and the card's specs:
- The card uses PCIe 5.0 x8. Copies from ordinary (pageable) RAM are assumed to
  reach 6–15 GB/s, depending on the board and its PCIe generation.

The expected gains:
- **Per token:** removes about 3–7 ms.
- **Per command:** saves about 30–60 ms, out of an estimated ~0.2 s.
- **256-token noise loops:** from about 3.5–7.5 s down to roughly 2–3 s.
- **On the CPU:** no change.

### The fix

Use IO binding when the device is CUDA:

```python
binding = self._decoder.io_binding()
binding.bind_cpu_input("input_embeds", input_embeds)        # small
binding.bind_cpu_input("position_ids", position_ids)
binding.bind_cpu_input("attention_mask", mask)
binding.bind_ortvalue_input("past_keys", past_keys)         # OrtValue on the GPU
binding.bind_ortvalue_input("past_values", past_values)
binding.bind_output("logits", "cpu")                         # only this comes back
binding.bind_output("present_keys", "cuda", device_id)       # stays on the GPU
binding.bind_output("present_values", "cuda", device_id)
self._decoder.run_with_iobinding(binding)
logits, keys, values = binding.get_outputs()                 # OrtValues
logits = logits.numpy()
```

Details to handle:
- **Cached tensors:** keep `_prefix_kv` and `_empty_kv` as GPU `OrtValue`s
  (`ort.OrtValue.ortvalue_from_numpy(array, "cuda", device_id)`), created
  once.
- **Device id:** take it from the `--device` string with
  `device.device_index()`.
- **Cache length:** `_causal_mask` needs the cache length. It reads
  `past_keys.shape[3]` today; an `OrtValue` gives it with `.shape()`.
- **Concurrent requests:** they share `_prefix_kv`. That stays safe as long as
  nothing writes into it in place; the graph's `Concat` makes new present
  tensors. Check this.
- **The CPU path:** leave it as it is. Alternatively, use the same binding code
  with device `"cpu"` and one code path; if so, check the CPU is not slower.
- **Logits:** keep bringing them back to RAM; argmax stays in numpy. The 30 MB
  of prompt-pass logits happens once per command; see section 6 for removing
  it.

### Testing

- **Correctness:** IO binding also works on the CPU provider, so the new path
  can be checked here, forced on for CPU:
  - `tests/test_qwen3_asr_noise.py`;
  - the 113-input re-test (`local/convert/retest.py run merged-1.7b`), whose
    raw outputs must equal `local/convert/retest/merged-1.7b.json` exactly.
- **Speed:** this can only be measured on the GPU machine. Compare time per
  command and one noise loop with `--device cuda`, before and after.

## 6. Other GPU findings (from the 5060 Ti estimate)

None of these have been run on a GPU. The estimate (per 3 s command, cached
prompt):

| | This CPU (measured) | 5060 Ti (estimated) | Weights in VRAM | VRAM in use (estimated) |
|---|---|---|---|---|
| 1.7B int4 | 1.44 s | ~0.21 s (0.16–0.28) | 2.25 GB | ~4–5 GB |
| 1.7B int8 decoder | 2.06 s | ~0.21 s (0.16–0.29) | 3.13 GB | ~5–6 GB |
| 1.7B all-int8 | 2.09 s | ~0.21 s (0.17–0.29) | 2.22 GB | ~4–5 GB |
| 0.6B int4 | 0.76 s | ~0.18 s (0.13–0.25) | 0.46 GB | ~1.5–2.5 GB |

The estimate is built from these measurements and figures:
- **The CPU decode step,** fitted over the int4 models: 13 ms fixed plus
  37 ms per GB of weights, which is this laptop's 27 GB/s memory bandwidth. The
  fit predicts the int8 step to within 4%.
- **The GPU:** 448 GB/s of memory bandwidth.
- **The source:** onnxruntime 1.26's CUDA code
  (`contrib_ops/cuda/quantization/matmul_nbits.cc`, `matmul_4bits.cu`,
  `matmul_8bits.cu`).
- **The data:** CPU timings are in `local/convert/retest/profile.json`.

- [ ] **Check the card is supported.** `Dockerfile.gpu` pins
  onnxruntime-gpu < 1.27 on a CUDA 12.8.1 base. Whether that build runs on the
  5060 Ti (Blackwell, sm_120) is unverified. If it doesn't, the server warns
  that CUDA is not in use and falls back to the CPU.
- [ ] **Launch overhead.** One decode step runs **~2,870 kernels** after
  onnxruntime's graph optimization (2,991 for the 0.6B). About half are shape
  operations: Unsqueeze 507, Gather 392, Concat 283, Reshape 224, Shape 140.
  On a GPU this is estimated at 6–11 ms per token, more than reading the
  weights (~3 ms int4, ~6 ms int8), and it is why the 0.6B and 1.7B would run
  at almost the same speed. Possible fixes:
  - export with fewer shape operations: static shapes where possible,
    onnx-simplifier, or folding the shape subgraphs;
  - a `GroupQueryAttention`-based export.

  CUDA graphs need fixed shapes, so they are not a straightforward fix.
- [ ] **The prompt pass returns logits for every position:** 29.8 MB, of which
  only the last row is used. Taking only the last position before `lm_head` in
  the export removes the copy. It also saves about 30 of the prompt pass's
  ~169 GFLOP: `lm_head` is 311M parameters, applied to 49 positions.
- [ ] **FP32 activations on CUDA.** For passes over more than one token (the
  prompt pass and a quantized encoder), MatMulNBits expands each layer's
  weights to FP32 in GPU memory and then runs cuBLAS. For the 1.7B prompt pass
  that is ~14 GB of memory traffic, an estimated ~35 ms. The one-token decode
  kernels read the packed weights directly, for both 4-bit and 8-bit. An FP16
  export would use onnxruntime's faster `fpA_intB` kernels instead. It changes
  the numerics, though, so the exact-string comparison against Qwen's original
  weights (PERFORMANCE.md) would need re-running.

## Optional follow-ups

- Rename the package (`wyoming_faster_whisper`), command, `WYO_WHISPER_*`
  environment variables and zeroconf name (`faster-whisper`). They were kept so
  existing configs carry over.
- onnxruntime writes a small `Microsoft/DeveloperTools/.onnxruntime/` folder
  under `$XDG_CACHE_HOME`. The images set that to `/data`, so it appears in the
  data volume. It's harmless, and the old images did the same.
- `--data-dir` can be given several times, but only the first one is used
  (that predates this work).
