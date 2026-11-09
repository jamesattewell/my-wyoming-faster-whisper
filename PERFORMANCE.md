# Qwen3-ASR Performance

This report compares three Qwen3-ASR models on the same inputs: our merged
1.7B, the published merged 0.6B, and Qwen's original 1.7B weights as the
reference. It covers test results, exact transcripts, speed and memory, and
behaviour on speech-like noise that gets past VAD.

It also checks the merged 1.7B against the unmerged 1.7B it was converted from,
in [Merged vs unmerged 1.7B](#merged-vs-unmerged-17b). Their outputs are
identical on all 113 inputs.

Measured 2026-11-09 on an Intel Core i7-9750H (x86, 6 cores / 12 threads),
CPU only, Linux, with 4 threads for every model (the server's default
`--cpu-threads`).

## Models

| Name in this report | What it is | Audio encoder | Decoder | On disk | Peak RAM |
|---|---|---|---|---|---|
| **original 1.7B** | `Qwen/Qwen3-ASR-1.7B-hf`, Qwen's full weights, run with transformers 5.22 (PyTorch) | full precision | full precision | 3.8 GB | 12.2 GB |
| **merged 1.7B** | built by `convert_qwen3_asr.py --size 1.7b` | FP32 | int4 (197 int4 layers) | 2.9 GB | 3.8 GB |
| **merged 0.6B** | `rhasspy/qwen3-asr-0.6b-onnx-int4-merged`, as downloaded | int4 | int4 | 0.8 GB | 1.4 GB |
| **unmerged 1.7B** | `andrewleech/qwen3-asr-1.7b-onnx`, as downloaded, run by upstream `main` | FP32 | int4 (197 int4 layers), split into `decoder_init` + `decoder_step` | 4.1 GB | 4.9 GB |

Notes on the models:
- The original weights are stored in bf16 and were run in FP32. The server
  does not run PyTorch; this model is the reference the other two are measured
  against.
- The merged 1.7B has the same precision layout as Qwen's non-merged 1.7B ONNX
  export (`andrewleech/qwen3-asr-1.7b-onnx`):
  - **Encoder:** the export's own encoder, unchanged. Its graph (664 nodes) and
    all 419 weight tensors were checked equal, and all are FP32.
  - **Decoder:** int4 in both, with the same 197 int4 layers. "Merged" means
    one decoder graph instead of the export's two, so the server can cache the
    prompt.
- File names: the source repo names its FP32 encoder `encoder.int4.onnx`, and
  the server loads that name, so the name stays. The "int4" in the folder name
  refers to the decoder.

## What was tested

There are 113 inputs in all. Every model got identical 16 kHz audio. The two
44.1 kHz recordings were converted once with the server's own converter (the
wyoming `AudioChunkConverter`, as `handler.py` uses). The prompt is the Home
Assistant one from the tests:

> Office, Study, Upstairs, Top Floor, Living Room Light, Office Lamp, Desk Lamp,
> Kitchen, Downstairs, Ecobee, Thermostat.

| Group | Inputs | What they are |
|---|---|---|
| Commands | 8 | the commands in `test_real_commands_are_unaffected`, with the language and exact text each test expects |
| Speech | 17 | every 16 kHz test recording, with the prompt and the language auto-detected (`test_the_guard_never_changes_a_real_command`) |
| Noise | 12 | the noise clips of `test_noise_does_not_echo_the_hass_prompt`: 2 noise types × 3 volumes × English/German, with the prompt |
| Wide | 76 | all 19 test recordings × prompt on/off × language auto-detected/forced to the recording's own language |

The first three groups are taken from `tests/test_qwen3_asr_noise.py` itself,
so they are exactly the tests' inputs.

The two merged models were run through the server's own code
(`Qwen3AsrTranscriber.transcribe`). The original weights were run through
Qwen's own pipeline (`processor.apply_transcription_request` and
`model.generate`), with greedy decoding and the same 256-token limit. Its
output was cleaned up the way the server does it: the text after the language
tag, special tokens removed, whitespace stripped.

Each model's output is given two ways:
- **Returned:** what the server would hand to Home Assistant, after the echo
  guard. The original model has no guard, so its column applies the guard's
  test to its finished output.
- **Raw:** what the model wrote before the guard.

On all 113 inputs, both merged models build exactly the same prompt tokens as
Qwen's own processor. This was checked on every input.

## Results at a glance

| | merged 1.7B | merged 0.6B |
|---|---|---|
| Test suite | 331 passed (whole suite, both models) | |
| Its own noise and command tests | 39/39 | 39/39 |
| Commands identical to the original 1.7B (and to the tests) | 8/8 | 8/8 |
| Speech identical to the original 1.7B | 16/17 | 16/17 |
| Wide set identical to the original 1.7B | 71/76 | 71/76 |
| Next-token agreement with the original 1.7B | 949/958 (99.1%) | 950/958 (99.2%) |
| Noise clips returning nothing (original 1.7B: 5/12) | 5/12 | 12/12 |
| Median time per command | 1.44 s | 0.76 s |

- **Next-token agreement:** the original 1.7B's output tokens are fed through
  each model, and this counts how often the model's own next-token choice
  matches the original's. It is summed over the commands, speech and wide
  groups.
- **Merged 1.7B vs the original:** the only differences are numbers written as
  digits ("16" for "sixteen"). Every word, name and language matches.
- **0.6B vs the original 1.7B:** the differences are listed under
  [Accuracy](#accuracy).

## Test suite

`pytest tests`: **331 passed** in 3 min 0 s, with nothing skipped and no
expected failures.

`tests/test_qwen3_asr_noise.py` runs every test once per model:

| Test | Inputs | merged 1.7B | merged 0.6B |
|---|---|---|---|
| Noise does not echo the prompt | 12 noise clips | 12/12 | 12/12 |
| An echo is cut short | 1 noise clip | 1/1 | 1/1 |
| Real commands are unaffected (exact text) | 8 commands | 8/8 | 8/8 |
| The guard never changes a real command | 17 recordings | 17/17 | 17/17 |
| Noise does not cost the next command its cached prompt | noise, then a command | 1/1 | 1/1 |

The noise test checks that fewer than three prompt names come back. It does not
check that the output is empty, so a made-up sentence passes it (see
[Noise](#noise)).

## Accuracy

### Commands: identical on all three

All three models return exactly the same text, case and punctuation included,
and it is what each test expects.

| Recording | Language | Text (all three models) |
|---|---|---|
| turn-on-the-office-lamp | en | `Turn on the office lamp.` |
| whats-the-temperature-of-the-ecobee | en | `What's the temperature of the Ecobee?` |
| whats-the-temperature-of-the-ecobee | auto | `What's the temperature of the Ecobee?` |
| living_room_light_en | en | `Turn on the living room light.` |
| living_room_light_de | de | `Schalte das Wohnzimmerlicht ein.` |
| turn-off-the-lights-please | en | `Turn off the lights, please.` |
| nevermind | en | `Never mind.` |
| set-volume-to-30-percent | en | `Set volume to thirty percent.` |

### Speech: 15 of 17 identical on all three

| Recording | original 1.7B | merged 1.7B | merged 0.6B |
|---|---|---|---|
| set-volume-to-16-percent | `Set volume to sixteen percent.` | `Set volume to 16 percent.` | `Set volume to sixteen percent.` |
| living_room_light_es | `Enciende la luz del salón.` | `Enciende la luz del salón.` | `utterance` |

The other 15 are identical on all three models:
- `living_room_light` in German, English, Japanese, Korean, Cantonese and
  Chinese;
- `nevermind`;
- `set-volume-to` 13, 15, 30, 50 and 60 percent;
- `turn-off-the-lights-please`;
- `turn-on-the-office-lamp`;
- `whats-the-temperature-of-the-ecobee`.

### Wide set: 66 of 76 identical on all three

Every difference is listed below. The returned and raw outputs are the same
here: the echo guard did not fire on any recording.

| Recording | Prompt | Language | original 1.7B | merged 1.7B | merged 0.6B |
|---|---|---|---|---|---|
| set-volume-to-16-percent | yes | auto | `Set volume to sixteen percent.` | `Set volume to 16 percent.` | `Set volume to sixteen percent.` |
| set-volume-to-16-percent | yes | en | `Set volume to sixteen percent.` | `Set volume to 16 percent.` | `Set volume to sixteen percent.` |
| set-volume-to-16-percent | no | auto | `Set volume to sixteen percent.` | `Set volume to 16 percent.` | `Set volume to sixteen percent.` |
| set-volume-to-16-percent | no | en | `Set volume to sixteen percent.` | `Set volume to 16 percent.` | `Set volume to sixteen percent.` |
| set-volume-to-60-percent | no | en | `Set volume to sixty percent.` | `Set volume to 60 percent.` | `Set volume to sixty percent.` |
| living_room_light_es | yes | auto | `Enciende la luz del salón.` | `Enciende la luz del salón.` | `utterance` |
| living_room_light_ko | no | auto | `거실 불 켜줘.` | `거실 불 켜줘.` | `거실 불 켜죠.` |
| living_room_light_ko | no | ko | `거실 불 켜줘.` | `거실 불 켜줘.` | `거실 불 켜죠.` |
| whats-the-temperature-of-the-ecobee | no | auto | `What's the temperature of the incubi?` | `What's the temperature of the incubi?` | `What's the temperature of the Igboy?` |
| whats-the-temperature-of-the-ecobee | no | en | `What's the temperature of the incubi?` | `What's the temperature of the incubi?` | `What's the temperature of the Igboy?` |

**Merged 1.7B against the original 1.7B: 71/76.** All 5 differences are numbers
written as digits instead of words. Its encoder is the original's, and its
prompt and mel spectrogram match the original's, so the int4 decoder is the
likely cause. Both forms carry the same meaning; whether Home Assistant's
intent matching treats "16 percent" and "sixteen percent" the same was not
tested here.

**Merged 0.6B against the original 1.7B: 71/76.** This is a smaller model, so
it is not expected to match exactly. Its differences:
- **`utterance`** instead of the Spanish command, with the prompt and the
  language auto-detected. This is a wrong transcript.
- **Korean:** `켜죠` instead of `켜줘` in the last syllable, without the prompt.
- **Ecobee without the prompt:** "Igboy" against 1.7B's "incubi". Both are
  wrong without the prompt, and all three models get "Ecobee" right with it.

Next-token agreement with the original 1.7B, by group:

| Group | merged 1.7B | merged 0.6B |
|---|---|---|
| Commands | 68/68 (100%) | 68/68 (100%) |
| Speech | 182/184 (98.9%) | 182/184 (98.9%) |
| Wide | 699/706 (99.0%) | 700/706 (99.2%) |

## Noise

These are 3 s clips of speech-like noise that Silero VAD accepts as speech:
- shuffled speech: 40 ms pieces in random order;
- babble: six talkers played backwards over each other.

Each was run at three volumes, in English and German, with the prompt. Noise has
no correct transcript, so the question is what reaches Home Assistant.

What the server returns (echo guard on):

| Noise | Volume | Language | original 1.7B | merged 1.7B | merged 0.6B |
|---|---|---|---|---|---|
| shuffled | 0.1 | en | *(empty)* | *(empty)* | *(empty)* |
| shuffled | 0.1 | de | `Klimaschutz, Klimaschutz, Klimaschutz.` | `Ich glaube, ich glaube, …` (×50, to the 256-token limit) | *(empty)* |
| shuffled | 0.5 | en | *(empty)* | *(empty)* | *(empty)* |
| shuffled | 0.5 | de | `Klimaschutz, Klimaschutz, Klimaschutz.` | *(empty)* | *(empty)* |
| shuffled | 0.9 | en | *(empty)* | *(empty)* | *(empty)* |
| shuffled | 0.9 | de | `Klimaschalter, Lichtschalter, Thermostat.` | *(empty)* | *(empty)* |
| babble | 0.1 | en | *(empty)* | *(empty)* | *(empty)* |
| babble | 0.1 | de | *(empty)* | *(empty)* | *(empty)* |
| babble | 0.5 | en | `And it's on the TV.` | `And it's 30 degrees on the TV.` | *(empty)* |
| babble | 0.5 | de | `Ich bin in der Meeting.` | `Der Kühlschrank ist auf der linken Seite.` | *(empty)* |
| babble | 0.9 | en | `And it's thirty-one degrees.` | `And it's 30 degrees on the meeting.` | *(empty)* |
| babble | 0.9 | de | `Ich bin hier im Meeting.` | `Es ist 30 Grad.` | *(empty)* |

What the models wrote before the guard:

| Model | Read the prompt back | Caught by the guard | Wrote something else |
|---|---|---|---|
| original 1.7B | 5/12 | 5/5 | 7/12 |
| merged 1.7B | 7/12 | 7/7 | 5/12 |
| merged 0.6B | 12/12 | 12/12 | 0/12 |

What this shows:
- **Every model reads the prompt back on noise**, Qwen's original weights
  included. The echo is how the model itself behaves, not a fault of the ONNX
  builds. The guard caught all 24 echoes.
- **The echoes vary in form:**
  - the merged 0.6B splits "Living Room, Light";
  - the merged 1.7B lowercases names and writes "down stairs";
  - the original 1.7B starts its echo at "Upstairs".

  The guard matches without regard to case, spacing or punctuation.
- **1.7B, original and merged alike, turns babble into sentences** that are not
  echoes, so the guard cannot catch them. Examples are "Es ist 30 Grad." and
  "And it's 30 degrees on the TV.". Home Assistant would receive these as
  commands. The 0.6B returned nothing on all 12 clips.
- **The merged and original 1.7B write different things on the same noise.**
  Noise has no correct answer, so small numerical differences send decoding
  down different paths. Both produce the same kind of output.

## Merged vs unmerged 1.7B

The merged 1.7B is built from Qwen's PyTorch weights by
`convert_qwen3_asr.py`, and takes its encoder from the unmerged ONNX export
(`andrewleech/qwen3-asr-1.7b-onnx`). This section checks whether the two behave
the same.

**How it was run.** The unmerged model was downloaded as published, and run by
the Qwen3-ASR handler on upstream `main` (commit 5f40739). That handler still
supports the split layout, and was run from a separate checkout. It got the same
113 inputs and the same 4 threads as the other models. `main` has no echo
guard, so its output is compared with the merged model's raw output.

**Outputs: identical on all 113 inputs.**

| Group | Identical to the merged 1.7B (raw) |
|---|---|
| Commands | 8/8 |
| Speech | 17/17 |
| Noise | 12/12 |
| Wide | 76/76 |

Identical noise output is the strongest sign. On noise, the smallest numerical
difference usually sends the output somewhere else: the merged and original
1.7B agree on only 2 of the 12 clips. The unmerged model also writes
"16 percent" where Qwen's original weights write "sixteen". That difference
comes from the int4 model both share, not from the merge.

**Weights: identical.**

| Part | Merged 1.7B vs unmerged 1.7B |
|---|---|
| Audio encoder | same graph (664 nodes) and same 419 FP32 tensors; it is the unmerged model's own file |
| Decoder | all 197 int4 layers identical: weights, scales and zero points. `decoder_init` and `decoder_step` share the same 197 |
| Embeddings, tokenizer, config | the unmerged model's own files, copied unchanged |

The int4 decoder weights match without being copied. The converter exports the
decoder from Qwen's PyTorch weights and quantizes it with the recipe the
unmerged export uses (round-to-nearest, block 64, asymmetric), and that produces
the same bytes.

**What does differ:**

| | merged 1.7B (this branch) | unmerged 1.7B (upstream `main`) |
|---|---|---|
| Decoder graphs | one, which takes cached keys and values *and* any number of new tokens | two: `decoder_init` processes the whole prompt, and `decoder_step` takes one token at a time |
| Prompt handling | the Home Assistant prompt is computed once and reused for every command | the prompt is processed again for every command |
| Echo guard | yes: stops and returns nothing once 3 prompt names are read back | no: returns whatever the model writes |
| Median time per command | 1.44 s | 1.87 s |
| Noise clip, median / slowest | 1.25 s / 16.23 s | 2.80 s / 15.70 s |
| Noise clips returning nothing | 5/12 | 0/12 (7 echoes of the prompt, 5 sentences) |
| Peak RAM | 3.8 GB | 4.9 GB |
| On disk | 2.9 GB | 4.1 GB |

- **Speed:** the merged model is 23% faster per command because it does not
  re-process the prompt, and the gap grows with the prompt's length. It is
  faster on noise because the guard stops echoes early.
- **Memory and disk:** `decoder_init` carries its own 1.2 GB FP32 copy of the
  embedding table, which it needs because it takes token ids. The tensor is
  named `lm_head.weight`, a 151936 × 2048 table; the embeddings and output
  layer share their weights in this model. The merged decoder takes
  embeddings, which the server reads from `embed_tokens.bin`.
- **The repetition loop:** both models take about 16 s on the "Ich glaube"
  clip. The model and its weights are the same, and neither handler stops
  repetition.

## Speed and memory

Timings exclude model loading:
- **Merged models:** the server's `transcribe`, best of 3 runs, with the prompt
  already cached.
- **Original 1.7B:** one `model.generate` call per input, which does not
  include building the inputs.
- **All three:** 4 threads. The 17 speech recordings are 1.9–3.5 s long.

| | original 1.7B | merged 1.7B | merged 0.6B |
|---|---|---|---|
| Median time per command | 3.29 s | 1.44 s | 0.76 s |
| Mean / slowest | 3.37 s / 4.08 s | 1.41 s / 1.61 s | 0.74 s / 0.84 s |
| Median real-time factor | 1.26 | 0.51 | 0.25 |
| Noise clip, median | 5.63 s | 1.25 s | 0.64 s |
| Noise clip, slowest | 8.89 s | **16.23 s** | 0.66 s |
| Model load | 5.1 s | 4.3 s | 2.7 s |
| Peak RAM | 12.2 GB | 3.8 GB | 1.4 GB |
| On disk | 3.8 GB | 2.9 GB | 0.8 GB |

- **The merged 1.7B is about 1.9× slower than the 0.6B** (1.44 s against
  0.76 s per command). It is 2.3× faster than the original weights, and needs
  under a third of their memory.
- **The 16.23 s noise case** is the merged 1.7B repeating "Ich glaube, " until
  it reaches the 256-token limit. The guard does not stop it because the text
  contains no prompt names.
- **The original model's noise times are high** because it has no guard and
  reads the whole prompt back. The merged models stop as soon as the guard
  fires.
- **The unmerged 1.7B** takes 1.87 s per command, with 4.9 GB of peak RAM. Its
  timings are in [Merged vs unmerged 1.7B](#merged-vs-unmerged-17b).

## Open issues

1. **1.7B turns noise into sentences that reach Home Assistant.** It returned
   text on 5 of 12 noise clips, and 4 of those were sentences such as "Es ist
   30 Grad.". Qwen's original weights do the same, so the conversion is not the
   cause. The echo guard cannot catch these; the 0.6B returned nothing on all
   12 clips.
2. **The repetition loop on noise costs 16 s** on the merged 1.7B, because
   decoding runs to the 256-token limit.
3. **The merged 1.7B writes some numbers as digits** ("16 percent" where the
   original writes "sixteen percent"). This is probably the int4 decoder. It
   was not tested whether this changes how Home Assistant handles the command.
4. **The merged 0.6B returned "utterance"** for the Spanish command with the
   language auto-detected.

## Reproducing

- **Test suite:** `script/test`. The noise and command tests run once for each
  model folder found in `local/`: `local/rhasspy/qwen3-asr-0.6b-onnx-int4-merged`
  (downloaded on first server run) and `local/qwen3-asr-1.7b-onnx-int4-merged`.
- **The merged 1.7B:** `convert_qwen3_asr.py --size 1.7b --out
  local/qwen3-asr-1.7b-onnx-int4-merged`. See "Other Model Sizes" in the
  README.
- **The comparison scripts are not committed:** they are in the gitignored
  `local/convert/` folder.
  - `retest.py` covers the three main models. Its steps are `prep`,
    `run <model>` and `report`. The original-model run needs the converter's
    environment (PyTorch, transformers ≥ 5.13).
  - `run_split.py` runs the unmerged 1.7B with the handler from a checkout of
    upstream `main` in `local/convert/main-wt`.
