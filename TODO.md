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

## Optional follow-ups

- Rename the package (`wyoming_faster_whisper`), command, `WYO_WHISPER_*`
  environment variables and zeroconf name (`faster-whisper`). They were kept so
  existing configs carry over.
- onnxruntime writes a small `Microsoft/DeveloperTools/.onnxruntime/` folder
  under `$XDG_CACHE_HOME`. The images set that to `/data`, so it appears in the
  data volume. It's harmless, and the old images did the same.
- `--data-dir` can be given several times, but only the first one is used
  (that predates this work).
