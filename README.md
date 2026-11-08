# Wyoming Qwen3-ASR

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for the
[Qwen3-ASR](https://huggingface.co/Qwen) speech to text model, run through
onnxruntime.

This is a fork of
[wyoming-faster-whisper](https://github.com/OHF-Voice/wyoming-faster-whisper)
cut down to its Qwen3-ASR backend. The Python package
(`wyoming_faster_whisper`), the command (`wyoming-faster-whisper`) and the
`WYO_WHISPER_*` environment variables keep their upstream names, so existing
configurations carry over.

## Local Install

Clone the repository and set up Python virtual environment:

``` sh
git clone https://github.com/jamesattewell/my-wyoming-faster-whisper.git
cd my-wyoming-faster-whisper
script/setup
```

Run a server anyone can connect to:

```sh
script/run --language en --uri 'tcp://0.0.0.0:10300' --data-dir /data
```

**NOTE**: Models are downloaded to the first `--data-dir` directory (or
`--download-dir`, if given).

## Model

The default `--model` is `rhasspy/qwen3-asr-0.6b-onnx-int4-merged` (785 MB). It
can also be another Hugging Face repo with the same files, or a local directory
holding them:

`config.json`, `tokenizer.json`, `embed_tokens.bin`, `encoder.int4.onnx` (+
`.data`), `decoder_merged.int4.onnx` (+ `.data`)

Only this *merged* decoder export is supported, because it lets the biasing
prompt be computed once and reused (see [Prompt cost](#prompt-cost)). The older
*split* export (`decoder_init`/`decoder_step`, e.g.
`rhasspy/qwen3-asr-0.6b-onnx-int4`) holds the same weights but is refused at
startup.

A repo is downloaded into a plain folder named after it under the download
directory — `/data/rhasspy/qwen3-asr-0.6b-onnx-int4-merged` with
`--data-dir /data`. It deliberately does not use the Hugging Face cache layout:
there the files are symlinks into a content-addressed store, which puts each
`.onnx` graph and its `.data` weights in different directories, and onnxruntime
refuses to load weights that resolve outside the graph's own directory.

Earlier versions used that cache layout, so after upgrading the model is
downloaded once more. The old copy is in `models--rhasspy--qwen3-asr-*` folders
(and, with huggingface_hub 2.x, a top-level `blobs/` folder) in the download
directory, and can be deleted once nothing else uses that directory as a Hugging
Face cache.

Decoding is greedy, and `--cpu-threads` (default 4) sets onnxruntime's thread
count.

### Other Model Sizes

Only the 0.6B model is published in the merged layout. `convert_qwen3_asr.py`,
at the root of this repo, builds the folder for another size from Qwen's
PyTorch checkpoint. It is run by hand, in an environment of its own because it
needs PyTorch (which the server does not):

```sh
python3 -m venv .venv-convert
.venv-convert/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-convert/bin/pip install "transformers>=5.13" onnx onnxruntime onnxscript \
    safetensors huggingface_hub numpy

.venv-convert/bin/python convert_qwen3_asr.py --size 1.7b \
    --out local/qwen3-asr-1.7b-onnx-int4-merged
```

Then pass that folder as `--model`. For 1.7B it downloads ~10 GB into
`local/convert` (`--work-dir`), takes a couple of minutes, and needs ~12–15 GB
of RAM at its peak. Keep the work directory off tmpfs (`/tmp` on many systems),
whose files are held in RAM. Run with `--size 0.6b`, it reproduces the published
model: the decoder and encoder weights come out byte-identical.

On the test recordings 1.7B transcribes the same as 0.6B, except that it does
not pick up "Ecobee" from the prompt ("incubi"). It is about twice as slow: on
an x86 desktop with 4 threads, ~1.5 s against ~0.7 s for a 3 s command.

## Server-side VAD Endpointing

By default, the server waits for the client to send `audio-stop` before it
transcribes. Set `--vad-endpointing` to a number of seconds to end the command
after that much Silero-detected silence and send the transcript immediately:

```sh
script/run --uri 'tcp://0.0.0.0:10300' --data-dir /data \
    --vad-endpointing 0.7
```

When enabled, the Wyoming service advertises that it does not require external
VAD, so Home Assistant leaves endpointing to this server. Speech must be detected
before the silence timer starts; brief pauses reset when speech resumes, and a
command is allowed to run for at least one second. Audio received after the
endpoint is ignored until the client's eventual `audio-stop`.

## Clipping Silence

`--vad-clip` trims leading and trailing silence from the recording before it is
transcribed. Qwen3-ASR's cost grows with the length of the audio, so this
shortens every request, and it leaves the model less silence to hallucinate on.
`--vad-clip-threshold` (default 0.5) is the speech probability that counts as
speech, and `--vad-clip-pad-ms` (default 400) is how much context is kept on
each side. A recording with no speech detected is transcribed unclipped.

It can be combined with `--vad-endpointing`.

## Reported Languages

The Wyoming service advertises Qwen3-ASR's 31 languages:

`ar`, `cs`, `da`, `de`, `el`, `en`, `es`, `fa`, `fi`, `fil`, `fr`, `hi`, `hu`,
`id`, `it`, `ja`, `ko`, `mk`, `ms`, `nl`, `pl`, `pt`, `ro`, `ru`, `sv`, `th`,
`tr`, `vi`, `yue`, `zh`, `zh-HK`

`--language` is checked against this list at startup, and a region-qualified code
is accepted everywhere — `en-US` is treated as `en`, and `zh-HK`/`zh_HK` stays
Cantonese rather than falling back to Mandarin. A language the model does not
know, or `auto`, lets it detect the language itself.

This matters because Home Assistant picks the pipeline's STT language out of this
list. Codes are reported in the form Home Assistant matches against, which is why
Cantonese is advertised as **`zh-HK`** rather than only `yue`. Home Assistant
treats `yue` and `zh` as unrelated languages, and `home-assistant/intents` ships
`zh-CN`/`zh-HK`/`zh-TW` with no bare `zh` or `yue` — so a server that says only
`yue` can never be reached by a Cantonese pipeline. `zh-CN` and `zh-TW` still
fall through to Mandarin.

## Biasing Toward Your Home Assistant Names

The model has never heard of your thermostat. "What's the temperature of the
Ecobee?" comes back as "What's the temperature of the Igboy?" — the acoustics were
fine, the model just has no reason to think that word exists.

Given a long-lived access token, the server reads the names in your home over the
Home Assistant websocket API and feeds them to the model as context, which fixes
exactly that class of error:

```sh
script/run --uri 'tcp://0.0.0.0:10300' --data-dir /data \
    --hass-token "$TOKEN" --hass-api 'http://homeassistant.local:8123/api'
```

Requires the `hass` extra (`script/setup --hass`, or):

```sh
pip install '.[hass]'
```

It collects the names of **conversation-exposed** entities and their aliases, plus
your area and floor names — the names a speaker can actually say. It also reads
which area each exposed entity is in, so the areas you can actually command rank
above the ones you can't. Nothing else is read, and no service is ever called.

The fetch is free in latency terms: it starts when the audio starts, while the
speaker is still talking, and the names are ready by the time the audio stops.
Home Assistant being slow or unreachable only costs freshness — the previous names
are used, or none at all, and the transcript still comes back.

| Option | Default | Purpose |
| --- | --- | --- |
| `--hass-token` | | Long-lived access token. Enables everything above. Use `WYO_WHISPER_HASS_TOKEN_FILE` to keep it off the command line (see [Environment Variables](#environment-variables)). |
| `--hass-api` | `http://homeassistant.local:8123/api` | Where to find Home Assistant. |
| `--hass-refresh-seconds` | `0` | Minimum seconds between refreshes. `0` refreshes every utterance, so a rename takes effect immediately. |
| `--hass-prompt-max-tokens` | `200` | Token budget for names, counted with the model's own tokenizer. |
| `--hass-prompt-timeout` | `1.0` | How long to wait on an unfinished refresh before transcribing with the names already on hand. |

A large home has more names than the budget holds, so they are added in priority
order and cut off when it runs out:

1. **Areas and floors that hold an exposed entity** — said in nearly every command
   ("turn on the *office* lamp"), and known to be real targets because something
   in them can actually be commanded.
2. **Entity names in the domains people say out loud** — `light`, `switch`, `fan`,
   `media_player`, `climate`, `scene`, `todo`. These are the proper nouns that get
   misheard.
3. **The remaining areas and floors** — sayable, but with nothing exposed in them
   there is no command they can complete.
4. **The remaining entity names** — in a big home mostly sensors, which are
   hundreds in number and usually asked about by area ("the temperature in the
   office") rather than by their own name.

Aliases are not a tier of their own. An alias is what you say *instead of* the
name the integration gave the thing, so each one sits directly behind the name it
belongs to and shares its tier — a light's "standing light" is kept or dropped
along with the light, never left out while lower-priority names get in.

Edit `PRIORITY_DOMAINS` in `wyoming_faster_whisper/hass_api.py` to change what
lands in tier 2. Run with `--debug` to see how many names were dropped and the
exact prompt used. `--initial-prompt` still works and is kept at the front of the
prompt, ahead of anything discovered from Home Assistant.

### Prompt Cost

The prompt is not free: the model has to read it before it starts decoding, at
roughly 2.8ms per token, so a 50-name list can double the time for a short
command.

The merged decoder avoids this. The prompt sits ahead of the audio in the chat
template, so its state depends only on the prompt and is computed once, then
reused for every later utterance. The older split export, which is no longer
supported, re-read it every time.

Measured on a Pi 5 (4 threads, 3.2s command, 50 names):

| | split (unsupported) | merged |
| --- | --- | --- |
| latency | 3.42s | 2.20s |
| peak RSS | 2.25 GB | 1.55 GB |
| on disk | 1407 MB | 785 MB |

The latency win is for short commands. Long-form audio gains little (~1.04x on a
30s clip), because the cached prompt is a small share of that work — though the
memory saving grows with length.

Accuracy is unchanged: on LibriSpeech test-other (n=200) the two produce
byte-identical transcripts with no prompt (5.35% WER for both), and 5.33% vs
5.43% with a 50-name prompt.

## Docker Image

Build the image and run it:

``` sh
docker build -t wyoming-qwen3-asr .
docker run -it -p 10300:10300 -v /path/to/local/data:/data wyoming-qwen3-asr \
    --language en
```

**NOTE**: Models are downloaded to `/data`, so make sure this points to a Docker volume.

### Running Without Internet Access

A model that is already downloaded is loaded from disk without contacting
Hugging Face, so a container that has been started once keeps working on a
network with no route out. Only a model with files *missing* is fetched.

That matters on a network which drops outbound traffic rather than refusing it —
a Docker bridge marked `internal`, for example. An update check on every start
would sit there until the TCP connection timed out, and with `restart: always`
the container would boot-loop.

To rule out downloads entirely, pass `--local-files-only` (or
`WYO_WHISPER_LOCAL_FILES_ONLY=true`) once the model is in `/data`. A model that
is not there is then an error instead of a download:

```yaml
services:
  whisper:
    image: wyoming-qwen3-asr
    volumes:
      - ./data:/data
    environment:
      WYO_WHISPER_LANGUAGE: en
      WYO_WHISPER_LOCAL_FILES_ONLY: "true"
    networks:
      - no-internet

networks:
  no-internet:
    internal: true
```

Both images point `HF_HOME` and `XDG_CACHE_HOME` at `/data` so the caches kept
outside the model directory land on the volume too. The largest is the Xet chunk
cache used while downloading, which can run to several GB. Outside Docker, set
them yourself if you don't want them in `~/.cache`.

### Health Check

Both images carry a `HEALTHCHECK`, so `docker ps` reports `healthy` or
`unhealthy` and a Compose stack can wait on it with `depends_on:` /
`condition: service_healthy`. It sends the server a `Describe` and requires an
`Info` with an ASR program back, rather than only opening a socket: the port is
bound by the OS, so a connect-only check stays green even when the event loop is
wedged, while a round trip proves the accept loop and the event handler are both
still running.

The first 5 minutes don't count against it, because the server only starts
listening once the model is loaded — which means downloading it on first run —
and it takes 3 consecutive failures to turn the container unhealthy, because
transcription runs on the event loop and a check can time out behind a long
request.

To run it by hand:

``` sh
docker exec whisper \
    /usr/src/.venv/bin/python3 -m wyoming_faster_whisper.health_check
```

It prints nothing and exits 0 when healthy, or prints `unhealthy: <reason>` and
exits 1. That reason is what `docker inspect` shows under `.State.Health`.

The URI it checks is `tcp://127.0.0.1:10300`, or `WYO_WHISPER_URI` when the
container sets one (a listen-everywhere host like `0.0.0.0` is rewritten to
loopback). A `--uri` passed to `docker run` instead of the variable is invisible
to the check, so give it the same one:

```yaml
    healthcheck:
      test: ["CMD", "/usr/src/.venv/bin/python3", "-m",
             "wyoming_faster_whisper.health_check",
             "--uri", "tcp://127.0.0.1:10400"]
```

### GPU Image

`Dockerfile.gpu` runs the model on an NVIDIA GPU through onnxruntime's CUDA
execution provider. Home Assistant OS has no GPU passthrough, so this is for
people running Docker directly:

``` sh
docker build -f Dockerfile.gpu -t wyoming-qwen3-asr:gpu .
```

Running it needs the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host, and `--gpus`:

``` sh
docker run -it --gpus all -p 10300:10300 -v /path/to/local/data:/data \
    wyoming-qwen3-asr:gpu --language en
```

`--device cuda` is the default in this image; pass `--device cuda:1` to pick a
GPU or `--device cpu` to fall back. The model's weights are int4, which is where
most of its speed comes from, so the GPU gain is smaller than for an fp32 model.

If `--device cuda` produces no speedup, check the log: the server warns when
onnxruntime is not actually using the CUDA provider.

### GPU Without Docker

The package depends on the CPU `onnxruntime`. `onnxruntime-gpu` installs the same
`onnxruntime` module, and whichever is installed second wins, so swap it in after
installing:

``` sh
pip install '.[hass]'
pip uninstall -y onnxruntime
pip install 'onnxruntime-gpu>=1.19,<1.27'
```

The `<1.27` bound keeps it on CUDA 12: onnxruntime-gpu 1.27.0 moved to CUDA 13.
Use a newer version if that is what your system has. `Dockerfile.gpu` does
exactly this and is the working reference.

## Environment Variables

Every command-line option can also be set from the environment, which is what
Docker Compose gives you to configure a container without rewriting its
`command:`. The variable is the option's name, uppercased, with dashes as
underscores and a `WYO_WHISPER_` prefix:

| Option | Variable |
| --- | --- |
| `--model` | `WYO_WHISPER_MODEL` |
| `--language` | `WYO_WHISPER_LANGUAGE` |
| `--vad-endpointing` | `WYO_WHISPER_VAD_ENDPOINTING` |
| `--hass-token` | `WYO_WHISPER_HASS_TOKEN` |
| ...and so on for every option in `--help` | |

**Precedence**: a command-line argument always wins over the environment, so a
variable left over in a container can never silently override an explicit
argument. A variable that starts with `WYO_WHISPER_` but matches no option is
reported at startup rather than ignored, so a typo doesn't leave the server
running with a default you thought you had changed.

A few options don't take a plain string on the command line, so they read one
specially:

- **Flags** (`--debug`, `--vad-clip`, `--local-files-only`, ...) take
  `1`/`true`/`yes`/`on` or `0`/`false`/`no`/`off`/empty. Unlike a plain "is it
  set?" test, `WYO_WHISPER_DEBUG=false` really does mean off.
- **`--data-dir`**, which can be repeated, splits on `:` —
  `WYO_WHISPER_DATA_DIR=/data:/media/models`. Passing any `--data-dir` on the
  command line replaces the list rather than adding to it.
- **`--zeroconf`**, whose value is optional, takes the name to announce, or
  empty for the default name.

Any other variable left empty means the same as not setting it at all, which is
what a compose file or a `.env` produces for a value whose author left it blank.
An empty `WYO_WHISPER_URI` gets you argparse's "the following arguments are
required", not a server that starts up and fails on an empty string.

### Secrets

Each variable also has a `_FILE` form naming a file to read the value out of:
`WYO_WHISPER_HASS_TOKEN_FILE=/run/secrets/hass_token`. That is the convention
Docker Compose and Swarm secrets use — a secret is mounted as a file rather than
handed over as a variable — and it keeps a long-lived Home Assistant token out
of both the process's command line and its environment. The file is preferred
over the plain variable when both are set, and its trailing newline is stripped.

```yaml
services:
  whisper:
    image: wyoming-qwen3-asr
    ports:
      - "10300:10300"
    volumes:
      - ./data:/data
    environment:
      WYO_WHISPER_LANGUAGE: en
      WYO_WHISPER_HASS_API: http://homeassistant.local:8123/api
      WYO_WHISPER_HASS_TOKEN_FILE: /run/secrets/hass_token
    secrets:
      - hass_token

secrets:
  hass_token:
    file: ./secrets/hass_token.txt
```

In the Docker image, `--uri`, `--data-dir`, and `--device` have defaults baked
into the entrypoint; those are dropped when the matching variable is set, so
`WYO_WHISPER_URI` and friends work there too.
