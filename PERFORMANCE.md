# Qwen3-ASR Performance

This report compares five Qwen3-ASR models on the same inputs:
- our merged 1.7B, with an int4 decoder;
- the same model with an int8 decoder;
- the same model with both the encoder and the decoder in int8;
- the published merged 0.6B;
- Qwen's original 1.7B weights, as the reference.

It covers test results, exact transcripts, speed and memory, and behaviour on
speech-like noise that gets past VAD.

Both int8 builds return exactly what the original weights return on all 101
recordings. The int4 1.7B differs on 6, by writing numbers as digits. See
[int8 builds](#int8-builds).

It also checks the merged 1.7B against the unmerged 1.7B it was converted from,
in [Merged vs unmerged 1.7B](#merged-vs-unmerged-17b). Their outputs are
identical on all 113 inputs.

Measured 2026-11-09 on an Intel Core i7-9750H (x86, 6 cores / 12 threads),
CPU only, Linux, with 4 threads for every model (the server's default
`--cpu-threads`). Sizes are in GB of 10⁹ bytes: disk is the model folder,
RAM is the process's peak resident memory.

## Models

| Name in this report | What it is | Audio encoder | Decoder | On disk | Peak RAM |
|---|---|---|---|---|---|
| **original 1.7B** | `Qwen/Qwen3-ASR-1.7B-hf`, Qwen's full weights, run with transformers 5.22 (PyTorch) | full precision | full precision | 4.09 GB | 12.8 GB |
| **merged 1.7B** | built by `convert_qwen3_asr.py --size 1.7b` | FP32 | int4 (197 int4 layers) | 2.89 GB | 4.0 GB |
| **int8 1.7B** | the merged 1.7B with its decoder quantized to int8 instead of int4 (`local/convert/quant_int8.py`) | FP32 (the same file) | int8 (the same 197 layers) | 3.76 GB | 4.9 GB |
| **all-int8 1.7B** | the int8 1.7B with its encoder also quantized to int8 (`local/convert/quant_int8_all.py`) | int8 (147 layers) | int8 (the int8 1.7B's file) | 2.85 GB | 3.7 GB |
| **merged 0.6B** | `rhasspy/qwen3-asr-0.6b-onnx-int4-merged`, as downloaded | int4 | int4 | 0.79 GB | 1.5 GB |
| **unmerged 1.7B** | `andrewleech/qwen3-asr-1.7b-onnx`, as downloaded, run by upstream `main` | FP32 | int4 (197 int4 layers), split into `decoder_init` + `decoder_step` | 4.53 GB | 5.2 GB |

Notes on the models:
- The original weights are stored in bf16 and were run in FP32. The server
  does not run PyTorch; this model is the reference the others are measured
  against.
- The merged 1.7B has the same precision layout as Qwen's non-merged 1.7B ONNX
  export (`andrewleech/qwen3-asr-1.7b-onnx`):
  - **Encoder:** the export's own encoder, unchanged. Its graph (664 nodes) and
    all 419 weight tensors were checked equal, and all are FP32.
  - **Decoder:** int4 in both, with the same 197 int4 layers. "Merged" means
    one decoder graph instead of the export's two, so the server can cache the
    prompt.
- The int8 1.7B differs from the merged 1.7B only in the decoder's weights. It
  was quantized from the same FP32 decoder export, with the same recipe
  (round-to-nearest, block 64, asymmetric, `accuracy_level` 4), at 8 bits
  instead of 4. Its encoder, embeddings, tokenizer and config are the merged
  1.7B's own files.
- The all-int8 1.7B uses the int8 1.7B's decoder file, and its encoder is the
  same FP32 encoder quantized to int8 with the same recipe. That covers the
  encoder's 147 weight layers. Its 48 attention MatMuls have no weights, and
  its convolution front end stays FP32.
- File names: the source repo names its FP32 encoder `encoder.int4.onnx`, and
  the server loads that name, so the name stays. The "int4" in the folder name
  refers to the decoder. The int8 decoder is also stored as
  `decoder_merged.int4.onnx`, the name the server looks for.

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

The four ONNX models were run through the server's own code
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

On all 113 inputs, all four ONNX models build exactly the same prompt tokens as
Qwen's own processor. This was checked on every input.

## Results at a glance

| | merged 1.7B (int4) | int8 1.7B | all-int8 1.7B | merged 0.6B |
|---|---|---|---|---|
| Its own noise and command tests | 39/39 | 39/39 | 39/39 | 39/39 |
| Commands identical to the original 1.7B (and to the tests) | 8/8 | 8/8 | 8/8 | 8/8 |
| Speech identical to the original 1.7B | 16/17 | **17/17** | **17/17** | 16/17 |
| Wide set identical to the original 1.7B | 71/76 | **76/76** | **76/76** | 71/76 |
| Next-token agreement with the original 1.7B | 949/958 (99.1%) | **958/958 (100%)** | **958/958 (100%)** | 950/958 (99.2%) |
| Noise output identical to the original 1.7B | 2/12 | 9/12 | 10/12 | 0/12 |
| Noise clips returning nothing (original 1.7B: 5/12) | 5/12 | 5/12 | 5/12 | 12/12 |
| Median time per command | 1.44 s | 2.06 s | 2.09 s | 0.76 s |
| Peak RAM | 4.0 GB | 4.9 GB | 3.7 GB | 1.5 GB |
| On disk | 2.89 GB | 3.76 GB | 2.85 GB | 0.79 GB |

- **Next-token agreement:** the original 1.7B's output tokens are fed through
  each model, and this counts how often the model's own next-token choice
  matches the original's. It is summed over the commands, speech and wide
  groups.
- **Merged 1.7B vs the original:** the only differences are numbers written as
  digits ("16" for "sixteen"). Every word, name and language matches.
- **int8 and all-int8 1.7B vs the original:** no differences on any
  recording. Both take 43–45% more time per command than the int4 decoder.
  The all-int8 build needs less RAM and disk than the int4 build, because its
  encoder shrinks by more than its decoder grows.
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

The int8 and all-int8 1.7B are not in the committed test list. Each was run
separately with its folder added to `_MODELS`: **39/39** on the same tests for
both, and the rest of the suite still passed (292 passed in each run).

## Accuracy

### Commands: identical on all five

All five models return exactly the same text, case and punctuation included,
and it is what each test expects.

| Recording | Language | Text (all five models) |
|---|---|---|
| turn-on-the-office-lamp | en | `Turn on the office lamp.` |
| whats-the-temperature-of-the-ecobee | en | `What's the temperature of the Ecobee?` |
| whats-the-temperature-of-the-ecobee | auto | `What's the temperature of the Ecobee?` |
| living_room_light_en | en | `Turn on the living room light.` |
| living_room_light_de | de | `Schalte das Wohnzimmerlicht ein.` |
| turn-off-the-lights-please | en | `Turn off the lights, please.` |
| nevermind | en | `Never mind.` |
| set-volume-to-30-percent | en | `Set volume to thirty percent.` |

### Speech: 15 of 17 identical on all five

| Recording | original 1.7B | merged 1.7B | int8 1.7B | all-int8 1.7B | merged 0.6B |
|---|---|---|---|---|---|
| set-volume-to-16-percent | `Set volume to sixteen percent.` | `Set volume to 16 percent.` | `Set volume to sixteen percent.` | `Set volume to sixteen percent.` | `Set volume to sixteen percent.` |
| living_room_light_es | `Enciende la luz del salón.` | `Enciende la luz del salón.` | `Enciende la luz del salón.` | `Enciende la luz del salón.` | `utterance` |

The other 15 are identical on all five models:
- `living_room_light` in German, English, Japanese, Korean, Cantonese and
  Chinese;
- `nevermind`;
- `set-volume-to` 13, 15, 30, 50 and 60 percent;
- `turn-off-the-lights-please`;
- `turn-on-the-office-lamp`;
- `whats-the-temperature-of-the-ecobee`.

### Wide set: 66 of 76 identical on all five

Every difference is listed below. The returned and raw outputs are the same
here: the echo guard did not fire on any recording.

| Recording | Prompt | Language | original 1.7B | merged 1.7B | int8 1.7B | all-int8 1.7B | merged 0.6B |
|---|---|---|---|---|---|---|---|
| set-volume-to-16-percent | yes | auto | `Set volume to sixteen percent.` | `Set volume to 16 percent.` | `Set volume to sixteen percent.` | `Set volume to sixteen percent.` | `Set volume to sixteen percent.` |
| set-volume-to-16-percent | yes | en | `Set volume to sixteen percent.` | `Set volume to 16 percent.` | `Set volume to sixteen percent.` | `Set volume to sixteen percent.` | `Set volume to sixteen percent.` |
| set-volume-to-16-percent | no | auto | `Set volume to sixteen percent.` | `Set volume to 16 percent.` | `Set volume to sixteen percent.` | `Set volume to sixteen percent.` | `Set volume to sixteen percent.` |
| set-volume-to-16-percent | no | en | `Set volume to sixteen percent.` | `Set volume to 16 percent.` | `Set volume to sixteen percent.` | `Set volume to sixteen percent.` | `Set volume to sixteen percent.` |
| set-volume-to-60-percent | no | en | `Set volume to sixty percent.` | `Set volume to 60 percent.` | `Set volume to sixty percent.` | `Set volume to sixty percent.` | `Set volume to sixty percent.` |
| living_room_light_es | yes | auto | `Enciende la luz del salón.` | `Enciende la luz del salón.` | `Enciende la luz del salón.` | `Enciende la luz del salón.` | `utterance` |
| living_room_light_ko | no | auto | `거실 불 켜줘.` | `거실 불 켜줘.` | `거실 불 켜줘.` | `거실 불 켜줘.` | `거실 불 켜죠.` |
| living_room_light_ko | no | ko | `거실 불 켜줘.` | `거실 불 켜줘.` | `거실 불 켜줘.` | `거실 불 켜줘.` | `거실 불 켜죠.` |
| whats-the-temperature-of-the-ecobee | no | auto | `What's the temperature of the incubi?` | `What's the temperature of the incubi?` | `What's the temperature of the incubi?` | `What's the temperature of the incubi?` | `What's the temperature of the Igboy?` |
| whats-the-temperature-of-the-ecobee | no | en | `What's the temperature of the incubi?` | `What's the temperature of the incubi?` | `What's the temperature of the incubi?` | `What's the temperature of the incubi?` | `What's the temperature of the Igboy?` |

**Merged 1.7B against the original 1.7B: 71/76.** All 5 differences are numbers
written as digits instead of words. The int8 1.7B, which differs from it only
in the decoder's precision, writes the words, so the int4 decoder is the cause
(see [int8 builds](#int8-builds)). Both forms carry the same meaning; whether
Home Assistant's intent matching treats "16 percent" and "sixteen percent" the
same was not tested here.

**int8 and all-int8 1.7B against the original 1.7B: 76/76 each.**

**Merged 0.6B against the original 1.7B: 71/76.** This is a smaller model, so
it is not expected to match exactly. Its differences:
- **`utterance`** instead of the Spanish command, with the prompt and the
  language auto-detected. This is a wrong transcript.
- **Korean:** `켜죠` instead of `켜줘` in the last syllable, without the prompt.
- **Ecobee without the prompt:** "Igboy" against 1.7B's "incubi". Both are
  wrong without the prompt, and all five models get "Ecobee" right with it.

Next-token agreement with the original 1.7B, by group:

| Group | merged 1.7B | int8 1.7B | all-int8 1.7B | merged 0.6B |
|---|---|---|---|---|
| Commands | 68/68 (100%) | 68/68 (100%) | 68/68 (100%) | 68/68 (100%) |
| Speech | 182/184 (98.9%) | 184/184 (100%) | 184/184 (100%) | 182/184 (98.9%) |
| Wide | 699/706 (99.0%) | 706/706 (100%) | 706/706 (100%) | 700/706 (99.2%) |

## Noise

These are 3 s clips of speech-like noise that Silero VAD accepts as speech:
- shuffled speech: 40 ms pieces in random order;
- babble: six talkers played backwards over each other.

Each was run at three volumes, in English and German, with the prompt. Noise has
no correct transcript, so the question is what reaches Home Assistant.

What the server returns (echo guard on):

| Noise | Volume | Language | original 1.7B | merged 1.7B | int8 1.7B | all-int8 1.7B | merged 0.6B |
|---|---|---|---|---|---|---|---|
| shuffled | 0.1 | en | *(empty)* | *(empty)* | *(empty)* | *(empty)* | *(empty)* |
| shuffled | 0.1 | de | `Klimaschutz, Klimaschutz, Klimaschutz.` | `Ich glaube, ich glaube, …` (×50, to the 256-token limit) | `Klimaklimaklima…` (to the 256-token limit) | `Klimaschutz, Klimaschutz, Klimaschutz.` | *(empty)* |
| shuffled | 0.5 | en | *(empty)* | *(empty)* | *(empty)* | *(empty)* | *(empty)* |
| shuffled | 0.5 | de | `Klimaschutz, Klimaschutz, Klimaschutz.` | *(empty)* | `Klimaschutz, Klimaschutz, Klimaschutz.` | `Klimaschutz, Klimaschutz, Klimaschutz.` | *(empty)* |
| shuffled | 0.9 | en | *(empty)* | *(empty)* | *(empty)* | *(empty)* | *(empty)* |
| shuffled | 0.9 | de | `Klimaschalter, Lichtschalter, Thermostat.` | *(empty)* | `Klimaschutz, Klimaschutz, Klimaschutz.` | `Klimaschalter, Lichtschalter, Thermostat.` | *(empty)* |
| babble | 0.1 | en | *(empty)* | *(empty)* | *(empty)* | *(empty)* | *(empty)* |
| babble | 0.1 | de | *(empty)* | *(empty)* | *(empty)* | *(empty)* | *(empty)* |
| babble | 0.5 | en | `And it's on the TV.` | `And it's 30 degrees on the TV.` | `And it's on the TV.` | `And it's on the TV.` | *(empty)* |
| babble | 0.5 | de | `Ich bin in der Meeting.` | `Der Kühlschrank ist auf der linken Seite.` | `Ich bin in der Meeting.` | `Ich bin gerade auf dem Weg.` | *(empty)* |
| babble | 0.9 | en | `And it's thirty-one degrees.` | `And it's 30 degrees on the meeting.` | `And it's thirty-one degrees.` | `And it's thirty-one degrees.` | *(empty)* |
| babble | 0.9 | de | `Ich bin hier im Meeting.` | `Es ist 30 Grad.` | `Ich bin hier.` | `Ich möchte die Lichter an.` | *(empty)* |

What the models wrote before the guard:

| Model | Read the prompt back | Caught by the guard | Wrote something else |
|---|---|---|---|
| original 1.7B | 5/12 | 5/5 | 7/12 |
| merged 1.7B | 7/12 | 7/7 | 5/12 |
| int8 1.7B | 5/12 | 5/5 | 7/12 |
| all-int8 1.7B | 5/12 | 5/5 | 7/12 |
| merged 0.6B | 12/12 | 12/12 | 0/12 |

What this shows:
- **Every model reads the prompt back on noise**, Qwen's original weights
  included. The echo is how the model itself behaves, not a fault of the ONNX
  builds. The guard caught all 34 echoes.
- **The echoes vary in form:**
  - the merged 0.6B splits "Living Room, Light";
  - the merged 1.7B lowercases names and writes "down stairs";
  - the original, int8 and all-int8 1.7B start their echo at "Upstairs",
    word for word the same.

  The guard matches without regard to case, spacing or punctuation.
- **1.7B, at every precision, turns babble into sentences** that are not
  echoes, so the guard cannot catch them. Examples are "Es ist 30 Grad." and
  "And it's 30 degrees on the TV.". Home Assistant would receive these as
  commands. The 0.6B returned nothing on all 12 clips.
- **The int8 1.7B tracks the original far more closely on noise:** 9 of 12
  identical, against 2 of 12 for the int4 decoder. Noise has no correct answer,
  so small numerical differences send decoding down different paths; the
  closer the weights, the more often the paths stay the same. Its 3
  differences are all German:
  - shuffled 0.1: it loops "Klima" to the token limit, where the original
    stops after three "Klimaschutz";
  - shuffled 0.9: three "Klimaschutz" instead of "Klimaschalter,
    Lichtschalter, Thermostat.";
  - babble 0.9: it drops "im Meeting".
- **The all-int8 1.7B is as close: 10 of 12 identical.** Its 2 differences are
  German babble, where it writes different sentences: "Ich bin gerade auf dem
  Weg." and "Ich möchte die Lichter an." ("I'd like the lights on"). The second
  reads like a real command, and Home Assistant might act on it. Getting 10
  rather than 9 does not make it closer to the original than the int8 1.7B:
  which noise clips happen to stay on the same path is down to chance.

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
| Peak RAM | 4.0 GB | 5.2 GB |
| On disk | 2.89 GB | 4.53 GB |

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

## int8 builds

The int4 decoder is where the merged 1.7B's remaining differences from Qwen's
original weights come from. To check, the same FP32 decoder export was
quantized to int8 with the same recipe (the int8 1.7B). A second build also
quantized the FP32 encoder to int8 with that recipe (the all-int8 1.7B).
Nothing else changed: the embeddings, tokenizer, config and server code are
the same files.

| | merged 1.7B (int4) | int8 1.7B | all-int8 1.7B |
|---|---|---|---|
| Encoder / decoder | FP32 / int4 | FP32 / int8 | int8 / int8 |
| Recordings identical to the original 1.7B (commands + speech + wide) | 95/101 | **101/101** | **101/101** |
| Next-token agreement with the original 1.7B | 949/958 (99.1%) | **958/958 (100%)** | **958/958 (100%)** |
| Noise output identical to the original 1.7B | 2/12 | 9/12 | 10/12 |
| Noise clips returning nothing | 5/12 | 5/12 | 5/12 |
| Median time per command | 1.44 s | 2.06 s (+43%) | 2.09 s (+45%) |
| Noise clip, median / slowest | 1.25 s / 16.23 s | 1.88 s / 24.78 s | 1.96 s / 2.78 s |
| Model load | 4.3 s | 4.7 s | 4.1 s |
| Peak RAM | 4.0 GB | 4.9 GB | 3.7 GB |
| Encoder weights on disk | 1.27 GB | 1.27 GB | 0.36 GB |
| Decoder weights on disk | 0.98 GB | 1.86 GB | 1.86 GB |
| Whole model on disk | 2.89 GB | 3.76 GB | 2.85 GB |

- **Accuracy:** on every recording, both int8 builds give the original model's
  transcript, case and punctuation included. Their next-token choice matches
  the original's at all 958 positions. The int4 decoder's digits
  ("16 percent", "60 percent") go away.
- **The int8 encoder costs nothing measurable:** the all-int8 and int8 1.7B
  return the same text on all 101 recordings. An int4 encoder is a different
  matter: it was found earlier to make the 1.7B write "incubi" for "Ecobee"
  with Ecobee in the prompt, which is why the converter keeps the encoder FP32.
  That earlier finding was not re-measured in this report.
- **Cost:** each command takes about 0.6 s longer than with the int4 decoder.
  The int8 1.7B needs 0.9 GB more RAM and 0.87 GB more disk. The all-int8 1.7B
  needs 0.3 GB *less* RAM and 0.04 GB less disk than the int4 build, because
  its encoder shrinks by 0.91 GB while the decoder grows by 0.88 GB. Both are
  still 1.6× faster than the original weights, and need under 40% of their
  memory.
- **Noise:** int8 does not fix the 1.7B's noise behaviour, because that comes
  from the model itself. Both int8 builds returned text on 7 of 12 clips, as
  the original does. The int8 1.7B's "Klima" loop runs to the 256-token limit
  and takes 24.78 s, the slowest single case in this report. The all-int8 1.7B
  did not loop on any clip, but that is chance: the original does not loop on
  these clips either, and a small difference elsewhere could start one.
- **Not tested:** whether a different int4 recipe (symmetric, smaller blocks,
  or `accuracy_level` other than 4) would get closer at int4's size. The
  `accuracy_level` 4 setting means activations are also rounded to int8 inside
  each layer; int8 matched exactly even so.

## Speed and memory

Timings exclude model loading:
- **Merged models:** the server's `transcribe`, best of 3 runs, with the prompt
  already cached.
- **Original 1.7B:** one `model.generate` call per input, which does not
  include building the inputs.
- **All of them:** 4 threads. The 17 speech recordings are 1.9–3.5 s long.

| | original 1.7B | merged 1.7B | int8 1.7B | all-int8 1.7B | merged 0.6B |
|---|---|---|---|---|---|
| Median time per command | 3.29 s | 1.44 s | 2.06 s | 2.09 s | 0.76 s |
| Mean / slowest | 3.37 s / 4.08 s | 1.41 s / 1.61 s | 2.05 s / 2.39 s | 2.05 s / 2.39 s | 0.74 s / 0.84 s |
| Median real-time factor | 1.26 | 0.51 | 0.74 | 0.76 | 0.25 |
| Noise clip, median | 5.63 s | 1.25 s | 1.88 s | 1.96 s | 0.64 s |
| Noise clip, slowest | 8.89 s | **16.23 s** | **24.78 s** | 2.78 s | 0.66 s |
| Model load | 5.1 s | 4.3 s | 4.7 s | 4.1 s | 2.7 s |
| Peak RAM | 12.8 GB | 4.0 GB | 4.9 GB | 3.7 GB | 1.5 GB |
| On disk | 4.09 GB | 2.89 GB | 3.76 GB | 2.85 GB | 0.79 GB |

- **The merged 1.7B is about 1.9× slower than the 0.6B** (1.44 s against
  0.76 s per command). It is 2.3× faster than the original weights, and needs
  under a third of their memory.
- **The int8 builds are 1.4× slower than the int4 one** (2.06–2.09 s against
  1.44 s per command): decoding a command reads every decoder weight once per
  token, and int8 weights are twice the size. Quantizing the encoder as well
  makes no measurable difference to speed; the encoder runs once per command.
- **The 16.23 s and 24.78 s noise cases** are 1.7B repetition loops that run
  to the 256-token limit: the merged 1.7B repeating "Ich glaube, " and the int8
  1.7B repeating "Klima". The guard does not stop them because the text
  contains no prompt names.
- **The original model's noise times are high** because it has no guard and
  reads the whole prompt back. The merged models stop as soon as the guard
  fires.
- **The unmerged 1.7B** takes 1.87 s per command, with 5.2 GB of peak RAM. Its
  timings are in [Merged vs unmerged 1.7B](#merged-vs-unmerged-17b).

## Open issues

1. **1.7B turns noise into sentences that reach Home Assistant.** It returned
   text on 5 of 12 noise clips, and 4 of those were sentences such as "Es ist
   30 Grad.". Qwen's original weights do the same, so the conversion is not the
   cause. The echo guard cannot catch these; the 0.6B returned nothing on all
   12 clips.
2. **Repetition loops on noise cost 16–25 s** on the 1.7B (seen with the int4
   and int8 decoders), because decoding runs to the 256-token limit.
3. **The merged 1.7B writes some numbers as digits** ("16 percent" where the
   original writes "sixteen percent"). This is the int4 decoder: an int8
   decoder removes it, at 43–45% more time per command. With the encoder also
   in int8, RAM and disk stay below the int4 build's. It was not tested whether
   the digits change how Home Assistant handles the command.
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
  - `retest.py` covers the five main models. Its steps are `prep`,
    `run <model>` and `report`. The original-model run needs the converter's
    environment (PyTorch, transformers ≥ 5.13).
  - `run_split.py` runs the unmerged 1.7B with the handler from a checkout of
    upstream `main` in `local/convert/main-wt`.
  - `quant_int8.py` builds the int8 1.7B from the FP32 decoder that
    `convert_qwen3_asr.py --keep-fp32` leaves behind, and `quant_int8_all.py`
    adds the int8 encoder to make the all-int8 1.7B. `retest.py run int8-1.7b`
    and `retest.py run int8all-1.7b` test them, and `compare_int8.py [model]`
    compares one with the others.
