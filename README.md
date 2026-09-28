<p align="center">
  <img src="assets/logo.png" alt="dots.tts" width="280">
</p>

<p align="center">
  <a href="https://github.com/studio-dots-ai/dots.tts"><img src="https://img.shields.io/badge/GitHub-studio--dots--ai%2Fdots.tts-blue?logo=github" alt="GitHub"></a>
  <a href="https://huggingface.co/collections/dots-studio/dotstts"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-dots.tts%20collection-yellow" alt="Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.07080"><img src="https://img.shields.io/badge/Report-TTS-b31b1b?logo=arxiv&logoColor=white" alt="TTS Report"></a>
  <a href="https://huggingface.co/spaces/dots-studio/dots.tts"><img src="https://img.shields.io/badge/Playground-TTS-orange" alt="TTS Playground"></a>
  <a href="https://studio-dots-ai.github.io/dots.tts-demo/"><img src="https://img.shields.io/badge/Demo%20Page-TTS-red" alt="TTS Demo Page"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.02673"><img src="https://img.shields.io/badge/Report-Edit-b31b1b?logo=arxiv&logoColor=white" alt="Edit Report"></a>
  <a href="https://dots-studio-dots-tts-edit.hf.space/"><img src="https://img.shields.io/badge/Playground-Edit-orange" alt="Edit Playground"></a>
  <a href="https://dots-studio-dots-tts-edit-demo.static.hf.space"><img src="https://img.shields.io/badge/Demo%20Page-Edit-red" alt="Edit Demo Page"></a>
  <a href="https://github.com/studio-dots-ai/doteBench"><img src="https://img.shields.io/badge/Benchmark-Edit-blue?logo=github" alt="Edit Benchmark"></a>
</p>

**dots.tts** is a **2B-parameter fully continuous, end-to-end autoregressive (AR) text-to-speech system**. The backbone pairs a semantic encoder, an LLM, and an autoregressive flow-matching acoustic head over a **48 kHz** AudioVAE, with no discrete tokens anywhere in the pipeline.

dots.tts achieves the best average performance on **Seed-TTS-Eval**, with WERs of **0.94% / 1.30% / 6.60%** and SIM scores of **81.0 / 77.1 / 79.5** on the zh / en / zh-hard test sets, respectively. It further attains the **highest average speaker similarity (83.9)** on the 24-language **MiniMax multilingual** benchmark. Across other benchmarks, dots.tts also consistently demonstrates **open-source state-of-the-art performance**, exhibiting strong generation stability, voice cloning ability, and emotional expressiveness.

### News

* **[2026.09]** 🔥 Released **doteBench V1.0**, an XML-first bilingual benchmark for precise speech editing. Its 2,081 English and Chinese cases cover text, emotion, prosody, pause, and compositional edits, evaluating instruction following, local preservation, and audio quality. [Code](https://github.com/studio-dots-ai/doteBench) · [Release](https://github.com/studio-dots-ai/doteBench/releases/tag/v1.0.0) · [Paper](https://arxiv.org/abs/2608.02673).

* **[2026.08]** 🔥 We have released **dots.tts.edit** for precise, instruction-controlled speech editing — download the [checkpoint](https://huggingface.co/dots-studio/dots.tts.edit), try the [Playground](https://dots-studio-dots-tts-edit.hf.space/), explore the [Demo Page](https://dots-studio-dots-tts-edit-demo.static.hf.space), and read the [paper](https://arxiv.org/abs/2608.02673).

* **[2026.08]** ⚡ Released `dots.tts-mf-2steps`, `dots.tts-mf-1step`, and `dots.tts-mf-2steps-stts` for high-quality voice cloning and double-streaming TTS. These checkpoints build on `dots.tts-mf` with fixed-step train–inference alignment. See [Checkpoints](#checkpoints).

* **[2026.08]** 🚀 [SGLang Omni](https://github.com/sgl-project/sglang-omni) now supports dots.tts (`mf` / `mf-2steps` / `mf-2steps-stts` / `soar` / `base`) with streaming PCM and CUDA-graph backbone decode. MeanFlow, two-step sCM, and STTS use continuous batching. Current Omni STTS serving consumes complete text or an already-collected token-ID sequence when each request starts; it does not yet accept incremental tokens into the same in-flight request. According to the SGLang Omni cookbook, on Seed-TTS-Eval **EN** (1× H100, `dots.tts-mf`, `num_steps=4`), peak throughput reaches **4.76 req/s** / **19.86 audio_s/s** at concurrency 16 (WER **1.35%**). See [SGLang Omni Usage](#sglang-omni-usage) and the [cookbook](https://sgl-project.github.io/sglang-omni/cookbook/dots_tts.html).

* **[2026.07]** 🚀 Shipped a **high-performance inference path** — under `--optimize`, `dots.tts-soar` reaches RTF p50 **0.20 / 0.18** and first-chunk latency **225 ms / 69 ms** (voice cloning / text-only); `dots.tts-mf` reaches **0.15 / 0.13** and **204 ms / 68 ms** respectively. See the [Efficiency](#-efficiency) section for details.

* **[2026.06]** 🔥 We have released **dots.tts** — 2B fully continuous AR TTS, with pretrained / SOAR / MeanFlow-distilled checkpoints and full inference & fine-tuning code under Apache-2.0.

---

## Contents

- [Quick Start](#-quick-start)
  - [Installation](#installation)
  - [CLI](#cli)
  - [Checkpoints](#checkpoints)
  - [Python API](#python-api)
  - [Web Demo (Gradio)](#web-demo-gradio)
  - [Fine-tuning](#fine-tuning)
  - [MeanFlow Distillation](#meanflow-distillation)
  - [SGLang Omni Usage](#sglang-omni-usage)
- [Usage Tips](#-usage-tips)
- [Architecture](#-architecture)
- [Performance](#-performance)
  - [Seed-TTS-Eval](#seed-tts-eval)
  - [MiniMax Multilingual](#minimax-multilingual-24-languages)
  - [CV3-Eval](#cv3-eval)
  - [EmergentTTS-Eval](#emergenttts-eval)
- [Efficiency](#-efficiency)
  - [SGLang Omni Efficiency](#sglang-omni-efficiency)
- [Community Projects](#-community-projects)
- [Risks and Limitations](#%EF%B8%8F-risks-and-limitations)
- [Citation](#-citation)
- [License](#-license)

---

## 🚀 Quick Start

### Installation

We recommend a fresh conda environment (Python 3.10–3.12):

```bash
conda create -n dots_tts python=3.10 -y
conda activate dots_tts
```

Install from PyPI:

```bash
pip install dots.tts
```

Or from source (for local development / editable install):

```bash
git clone https://github.com/studio-dots-ai/dots.tts
cd dots.tts
pip install -e . -c constraints/recommended.txt
```

For training / linting extras:

```bash
pip install 'dots.tts[full]'
# or from source:
pip install -e .[full] -c constraints/recommended.txt
```

The `constraints/recommended.txt` file pins the reproducible versions;
`pyproject.toml` declares compatibility ranges.

To use SGLang Omni for high-performance high-concurrency voice cloning:

```bash
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install -v -e .
```

Detailed installation instructions can be found in this [guidance](https://sgl-project.github.io/sglang-omni/get_started/installation.html).

### CLI

The package installs a `dots.tts` entry point. The examples below use `dots.tts-soar`; replace the model path with any checkpoint in the next section and apply its checkpoint-specific settings.

```bash
# Continuation voice cloning (reference audio + transcript) — recommended, best SIM
dots.tts \
  --model-name-or-path dots-studio/dots.tts-soar \
  --text "Hello, this is a zero-shot voice cloning demonstration." \
  --prompt-audio /path/to/reference.wav \
  --prompt-text "The exact transcript of the reference audio." \
  --output clone.wav

# X-vector-only voice cloning (reference audio only — timbre from speaker x-vector)
dots.tts \
  --model-name-or-path dots-studio/dots.tts-soar \
  --text "Hello, this is a zero-shot voice cloning demonstration." \
  --prompt-audio /path/to/reference.wav \
  --output clone.wav

# Random-voice sampling (no reference) — only meaningful with a fine-tuned
# single-speaker checkpoint
dots.tts \
  --model-name-or-path dots-studio/dots.tts-soar \
  --text "Hello, this is a quick speech synthesis test." \
  --output output.wav
```

Common flags:

| Flag | Description | Default |
|------|-------------|---------|
| `--num-steps` | Sampling steps. Uses an artifact-declared value when present; otherwise `10`. | artifact / `10` |
| `--guidance-scale` | CFG scale. Uses an artifact-declared value when present; otherwise `1.2`. | artifact / `1.2` |
| `--normalize-text` | Apply text normalization before inference (via [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)) | off |
| `--language` | Add an explicit language tag to the input text; accepts `none`, `auto_detect`, language codes such as `EN` / `ZH`, or names such as `english` / `chinese` | `none` |
| `--seed` | RNG seed (fixed seed → deterministic output) | `42` |

`dots.tts --help` lists the full set.

Speech editing uses the separate `dots.tts.edit` entry point. Source audio, a
tagged instruction, and the output path are required. Source and target
transcripts are optional: when omitted or blank, both are derived from the
instruction.

```bash
dots.tts.edit \
  --model-name-or-path dots-studio/dots.tts.edit \
  --source-audio /path/to/source.wav \
  --instruction 'Hello <sub targ="small">brave</sub> world.' \
  --output edited.wav
```

Explicit non-empty `--source-text` and `--target-text` values override the
derived transcripts. Edit speaker guidance defaults to `auto`: it is disabled
when the instruction contains at least one operation and every operation is
`emo`, `bg`, or `enhance`, and enabled for text, pitch, rate, pause, speaker
transfer, or mixed edits. Pass bare `--use-xvector` (or `--use-xvector on`) to
force it on, and `--use-xvector off` to force it off. TTS speaker guidance
remains enabled when reference audio is provided. Supported structural tags
include `<del>`, `<ins>`, `<sub targ="replacement">`, `<emo>`, `<pitch>`,
`<rate>`, `<enhance>`, `<bg>`, `<pause/>`, and `<spk_transfer/>`. Malformed
instructions and instructions that derive an empty transcript are rejected.

Notes:

- `--prompt-audio` selects the speaker voice — continuation cloning when paired with `--prompt-text`, x-vector-only cloning when used alone. Omitting `--prompt-audio` falls back to random-voice sampling, which is only meaningful on a fine-tuned single-speaker checkpoint.
- `--language` is useful for multilingual or code-switched text when you want to force the model-side language tag. For example, pass `--language EN` for English, `--language ZH` for Mandarin, `--language Cantonese` for Cantonese, or `--language auto_detect` to infer the tag from `--text`.
- Pass either a local model directory or a Hugging Face repo id.

### Checkpoints

Seven pretrained checkpoints are released on Hugging Face. They share the same backbone; choose by task and runtime entry point.

| Model and entry point | Recommended use | Settings | Description |
|---|---|---|---|
| [`dots-studio/dots.tts-base`](https://huggingface.co/dots-studio/dots.tts-base)<br>CLI: `dots.tts`<br>Python: `DotsTtsRuntime` | Pretraining baseline; fine-tuning base. | NFE `10`–`32` (default `10`); CFG `1.2`. | Base pretrained checkpoint. |
| [`dots-studio/dots.tts-soar`](https://huggingface.co/dots-studio/dots.tts-soar)<br>CLI: `dots.tts`<br>Python: `DotsTtsRuntime` | Highest speaker similarity; high-quality voice cloning; fine-tuning. | NFE `10`–`32` (default `10`); CFG `1.2`. | SOAR checkpoint on top of `dots.tts-base`. |
| [`dots-studio/dots.tts-mf`](https://huggingface.co/dots-studio/dots.tts-mf)<br>CLI: `dots.tts`<br>Python: `DotsTtsRuntime` | Latency- or concurrency-sensitive TTS. | NFE `4` recommended; CFG fused. | MeanFlow-distilled student from `dots.tts-soar`. |
| [`dots-studio/dots.tts-mf-2steps`](https://huggingface.co/dots-studio/dots.tts-mf-2steps)<br>CLI: `dots.tts`<br>Python: `DotsTtsRuntime` | Latency- or concurrency-sensitive TTS. | Omit sampling options; fixed NFE `2` sCM. | Built on `dots.tts-mf` with a fixed two-step schedule for exact train–inference alignment and additional refinements. Uses the dedicated sCM solver at inference. |
| [`dots-studio/dots.tts-mf-1step`](https://huggingface.co/dots-studio/dots.tts-mf-1step)<br>CLI: `dots.tts`<br>Python: `DotsTtsRuntime` | Latency- or concurrency-sensitive TTS. | Omit sampling options; fixed NFE `1`. | Built on `dots.tts-mf`, extending fixed-step training to one-step generation with further training refinements. |
| [`dots-studio/dots.tts-mf-2steps-stts`](https://huggingface.co/dots-studio/dots.tts-mf-2steps-stts)<br>CLI: -<br>Python: `DotsTtsRuntimeDoubleStreaming` | LLM interaction and duplex dialogue. | Omit sampling options; streaming cadence is artifact-defined. | Streaming-TTS checkpoint built for double-streaming use. Uses the same fixed two-step sCM sampling contract as `dots.tts-mf-2steps`. |
| [`dots-studio/dots.tts.edit`](https://huggingface.co/dots-studio/dots.tts.edit)<br>CLI: `dots.tts.edit`<br>Python: `DotsTtsEditRuntime` | Speech editing. | NFE `10`–`32` (default `10`); CFG `1.2`. | Instruction-controlled speech editing checkpoint built on `dots.tts-base`. |

Pass the repo id directly to the entry point shown above; the snapshot is fetched on first use and cached locally. Fixed-step artifacts reject incompatible sampling overrides.

### Python API

#### Basic TTS

```python
from dots_tts.runtime import DotsTtsRuntime
import soundfile as sf

runtime = DotsTtsRuntime.from_pretrained(
    "dots-studio/dots.tts-soar",
    precision="bfloat16",
    optimize=True,  # torch.compile acceleration (warmup at load, faster steady-state)
)

result = runtime.generate(
    text="Hello, this is a quick speech synthesis test.",
    prompt_audio_path="/path/to/reference.wav",
    prompt_text="The exact transcript of the reference audio.",
    num_steps=10,
    guidance_scale=1.2,
)

sf.write("output.wav", result["audio"].float().cpu().squeeze().numpy(), result["sample_rate"])
```

The fixed-step MeanFlow artifacts read their sampling contracts directly from
the model configuration, so CLI and Python calls do not need sampling options.

For low-latency playback or streaming to a client, use `generate_stream` instead — it yields audio chunks (`torch.Tensor`, shape `(1, samples)`) as they are produced. Arguments are identical to `generate`:

```python
import torch

stream = runtime.generate_stream(
    text="Hello, this is a streaming speech synthesis test.",
    prompt_audio_path="/path/to/reference.wav",
    prompt_text="The exact transcript of the reference audio.",
    num_steps=10,
    guidance_scale=1.2,
)

chunks = []
for chunk in stream:
    chunks.append(chunk.detach().float().cpu())
    # handle_chunk(chunk)  # push to a player / websocket / etc.

audio = torch.cat(chunks, dim=-1).squeeze().numpy()
sf.write("output_stream.wav", audio, runtime.sample_rate)
```

#### Double Streaming

For duplex dialogue systems where an upstream LLM emits text tokens incrementally, use the double-streaming runtime. It accepts one text token at a time and returns either an audio chunk or `None` when the acoustic stream needs more text context.

The recommended released checkpoint for this path is [`dots-studio/dots.tts-mf-2steps-stts`](https://huggingface.co/dots-studio/dots.tts-mf-2steps-stts). Its sampling and streaming settings are stored in the artifact, so callers only provide text and optional prompt audio/text.

The same contract is available from Python:

```python
import torch
import soundfile as sf

from dots_tts.runtime_double_streaming import DotsTtsRuntimeDoubleStreaming

runtime = DotsTtsRuntimeDoubleStreaming.from_pretrained(
    "dots-studio/dots.tts-mf-2steps-stts",
    precision="bfloat16",
    optimize=True,
    max_generate_length=500,
)

text = "你好呀，今天想聊点什么？或者有什么我能帮你的？"
text_token_ids = runtime.model.tokenizer.encode(text, add_special_tokens=False)

session = runtime.start_double_streaming(
    prompt_audio_path="/path/to/reference.wav",
    prompt_text="The exact transcript spoken in the reference audio.",
)

chunks = []
for token_id in text_token_ids:
    chunk = session.push_text_token(token_id)
    if chunk is not None:
        chunks.append(chunk.detach().cpu())

for chunk in session.finish_text():
    chunks.append(chunk.detach().cpu())

audio = torch.cat(chunks, dim=-1).float().squeeze().numpy()
sf.write("double_streaming.wav", audio, runtime.sample_rate)
```

`--optimize` is strongly recommended for double streaming. It adds a one-time `torch.compile` warmup at load time, but the steady-state path uses cached/compiled LLM, DiT, and vocoder steps and substantially reduces RTF and streaming gaps.

For a complete command-line example, see [`scripts/example_double_streaming.py`](scripts/example_double_streaming.py).

#### Speech Editing

The same edit contract is available from Python:

```python
from dots_tts.edit_runtime import DotsTtsEditRuntime

edit_runtime = DotsTtsEditRuntime.from_pretrained(
    "dots-studio/dots.tts.edit",
    precision="bfloat16",
)
result = edit_runtime.generate_edit(
    source_audio_path="/path/to/source.wav",
    instruction='Hello <sub targ="small">brave</sub> world.',
    # source_text and target_text are optional overrides.
    # use_xvector defaults to "auto"; pass True or False to override it.
    num_steps=10,
    guidance_scale=1.2,
)
sf.write("edited.wav", result["audio"].float().cpu().squeeze().numpy(), result["sample_rate"])
```

### Web Demo (Gradio)

```bash
python apps/gradio/app.py \
  --model-name-or-path dots-studio/dots.tts-soar \
  --optimize
```

Defaults to `http://0.0.0.0:7860`. With `--optimize` the first launch runs warmup (slower startup, faster steady-state).

For the local Edit Playground, build the frontend once with Node.js 20+ and
then launch the application:

```bash
cd apps/edit_playground/frontend
npm ci
npm run build
cd ../../..
python apps/edit_playground/app.py \
  --model-name-or-path dots-studio/dots.tts.edit \
  --optimize
```

The Edit Playground ships without voice, edit-source, or noise audio presets;
upload your own source/reference audio. Optional local transcription requires
the ASR extra:

```bash
python -m pip install -e '.[edit_playground_asr]' -c constraints/recommended.txt
python apps/edit_playground/app.py \
  --model-name-or-path dots-studio/dots.tts.edit \
  --asr-model Qwen/Qwen3-ASR-1.7B
```

The generated `frontend/dist` directory is intentionally not committed. Pass
`--rebuild-frontend` to install dependencies and rebuild it explicitly during
launch. See [`apps/edit_playground/README.md`](apps/edit_playground/README.md)
for the frontend test, build, and local development workflow.

### Fine-tuning

This repo exposes fine-tuning and MeanFlow distillation entry points. Fine-tune from a released checkpoint with:

```bash
accelerate launch scripts/train_dots_tts.py --config configs/dots_tts.yaml
```

`configs/dots_tts.yaml` is a smoke configuration that verifies the pipeline runs end-to-end on commodity hardware. Replace `train.pretrained_model_path`, `train_data.sources` / `val_data.sources`, `train.output_dir`, and `train.max_train_steps` with your own values to use it.

A helper script downloads LJSpeech-1.1-48kHz and emits a train/valid JSONL manifest for the smoke run:

```bash
python scripts/prepare_train_jsonl_manifest.py --output-dir downloaded_data
```

Manifest format — one JSON per line, minimum three fields:

```json
{"fid": "sample-0001", "audio": "/abs/path/to/audio.wav", "text": "hello world"}
```

### MeanFlow Distillation

MeanFlow distillation trains a MeanFlow DiT student against a frozen flow-matching teacher. The teacher can be the released SOAR checkpoint or any compatible flow-matching dots.tts checkpoint you have fine-tuned yourself.

To use SOAR as the teacher, download it first:

```bash
huggingface-cli download dots-studio/dots.tts-soar \
  --local-dir pretrained_models/dots.tts-soar
```

Then launch distillation with the MeanFlow config:

```bash
accelerate launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  scripts/train_dots_tts_meanflow.py \
  --config configs/dots_tts_meanflow.yaml \
  --teacher-model-path pretrained_models/dots.tts-soar
```

To distill from your own fine-tuned teacher, pass that checkpoint instead:

```bash
accelerate launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  scripts/train_dots_tts_meanflow.py \
  --config configs/dots_tts_meanflow.yaml \
  --teacher-model-path /path/to/your_finetuned_teacher
```

`configs/dots_tts_meanflow.yaml` is a conservative smoke configuration that uses the same LJSpeech manifests produced by `scripts/prepare_train_jsonl_manifest.py`. Replace `train.pretrained_model_path`, `--teacher-model-path`, `train_data.sources` / `val_data.sources`, `train.output_dir`, and `train.max_train_steps` for your own distillation run.

By default, the script initializes the student from `train.pretrained_model_path`, adds the MeanFlow duration embedding, freezes the non-DiT modules, and trains `student.core.velocity_field_predictor`. MeanFlow does not run a separate CFG branch at inference time; the default `fused` mode distills the guided teacher target into the student. Training checkpoints save the MeanFlow student only; the frozen teacher is not written into the checkpoint model directory. Pass `--train-all-parameters` only if you want to update the full dots.tts model.

Common MeanFlow flags:

| Flag | Description | Default |
|------|-------------|---------|
| `--teacher-model-path` | Frozen flow-matching teacher directory. Defaults to `train.pretrained_model_path` if omitted. | `train.pretrained_model_path` |
| `--teacher-steps` | Teacher rollout steps used to build the distillation target. Higher is slower and usually stronger. | `8` |
| `--teacher-solver` | Teacher ODE solver: `euler`, `midpoint`, or `rk4`. | `euler` |
| `--cfg-distill-mode` | `fused` distills a guided teacher target into the student; `natural` trains on sampled conditional/unconditional masks without fusing CFG. | `fused` |
| `--distill-cfg-scale` | Extra CFG coefficient used when `--cfg-distill-mode fused` is enabled. It matches inference `guidance_scale` semantics: `teacher_cond + scale * (teacher_cond - teacher_uncond)`. | `1.2` |
| `--anchor-prob` | Probability of using a zero-duration anchor sample in MeanFlow training. | `0.5` |
| `--debug` | Print the first few batch summaries and gradient diagnostics. | off |

---

### SGLang Omni Usage

[SGLang Omni](https://github.com/sgl-project/sglang-omni) serves dots.tts behind an OpenAI-compatible `/v1/audio/speech` API with continuous batching for MeanFlow, two-step sCM, and prebuilt STTS interleave schedules, plus streaming PCM and CUDA-graph backbone decode. SGLang Omni does not yet expose the model's end-to-end, same-request token/audio double-streaming runtime. Full details live in the [SGLang Omni dots.tts cookbook](https://sgl-project.github.io/sglang-omni/cookbook/dots_tts.html).

Install Omni as in [SGLang Omni Installation](https://sgl-project.github.io/sglang-omni/get_started/installation.html), then from the `sglang-omni` checkout:

```bash
hf download dots-studio/dots.tts-mf

sgl-omni serve \
  --model-path dots-studio/dots.tts-mf \
  --config examples/configs/dots_tts.yaml \
  --port 8000
```

Use the model-specific config for two-step sCM or STTS:

```bash
# Fixed two-step sCM
sgl-omni serve \
  --model-path dots-studio/dots.tts-mf-2steps \
  --config examples/configs/dots_tts_scm.yaml \
  --port 8000

# Streaming TTS with artifact-defined text/audio interleave
sgl-omni serve \
  --model-path dots-studio/dots.tts-mf-2steps-stts \
  --config examples/configs/dots_tts_stts.yaml \
  --port 8000
```

| Checkpoint | Omni config | Notes |
|---|---|---|
| [`dots-studio/dots.tts-mf`](https://huggingface.co/dots-studio/dots.tts-mf) | `examples/configs/dots_tts.yaml` | MeanFlow. Continuous batching (`max_running_requests=16`), `num_steps=4`. **Recommended for serving.** |
| [`dots-studio/dots.tts-mf-2steps`](https://huggingface.co/dots-studio/dots.tts-mf-2steps) | `examples/configs/dots_tts_scm.yaml` | Artifact-defined sCM (Euler, NFE `2`, CFG `0`). Continuous batching (`max_running_requests=16`). |
| [`dots-studio/dots.tts-mf-2steps-stts`](https://huggingface.co/dots-studio/dots.tts-mf-2steps-stts) | `examples/configs/dots_tts_stts.yaml` | Artifact-defined sCM and text/audio cadence. Continuous batching over prebuilt interleave schedules (`max_running_requests=16`); no in-flight token append. |
| [`dots-studio/dots.tts-soar`](https://huggingface.co/dots-studio/dots.tts-soar) | `examples/configs/dots_tts_soar.yaml` | Flow matching + CFG. Single request at a time (`max_running_requests=1`), `num_steps=10`. |
| [`dots-studio/dots.tts-base`](https://huggingface.co/dots-studio/dots.tts-base) | `examples/configs/dots_tts_soar.yaml` | Same as SOAR; pass `--model-path dots-studio/dots.tts-base`. |

Use the config file — it enables the compiled acoustic tail / vocoder and backbone decode CUDA graph. MeanFlow, sCM, and STTS use continuous batching; SOAR/base remain single-request because their CFG conditional/unconditional branches are not yet implemented by the batched acoustic tail.

Voice cloning (reference audio + transcript required):

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dots-studio/dots.tts-mf",
    "input": "Have a nice day and enjoy south california sunshine.",
    "references": [{
      "audio_path": "docs/_static/audio/male-voice.wav",
      "text": "Hey, Adam here. Let'\''s create something that feels real, sounds human, and connects every time."
    }],
    "seed": 42
  }' \
  --output output.wav
```

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "dots-studio/dots.tts-mf",
        "input": "Have a nice day and enjoy south california sunshine.",
        "references": [{
            "audio_path": "docs/_static/audio/male-voice.wav",
            "text": "Hey, Adam here. Let's create something that feels real, sounds human, and connects every time.",
        }],
        "seed": 42,
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

`ref_audio` / `ref_text` are accepted as a shorthand for `references[0].audio_path` / `references[0].text`.

The same request format works with `dots.tts-mf-2steps` and `dots.tts-mf-2steps-stts`. STTS requires reference audio plus its transcript and reads its interleave cadence from the checkpoint. Each Omni request must start with complete text or an already-collected `text_token_ids` (or `input_ids`) array. The `/v1/audio/speech/stream` WebSocket can receive text chunks, but it buffers them and starts separate TTS requests at sentence/clause boundaries; it does not inject new tokens into an in-flight STTS request. Therefore Omni does not yet provide same-request, token-by-token double-streaming inference.

Streaming (raw 48 kHz PCM; set `"stream": true` and `"response_format": "pcm"`):

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dots-studio/dots.tts-mf",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "docs/_static/audio/female-voice.wav",
      "text": "By repeating what students say, teachers can demonstrate that they are listening. By extending what students say."
    }],
    "stream": true,
    "response_format": "pcm",
    "seed": 42
  }' \
  --output output.pcm

ffmpeg -f s16le -ar 48000 -ac 1 -i output.pcm output.wav
```

Solver knobs (`speaker_scale`, `guidance_scale`, `eos_threshold`, `num_steps`, …) go under `stage_params.latent_engine` — not as top-level fields. `temperature` / `top_p` / `top_k` do not apply (continuous latent; no token sampler). MeanFlow fixes `num_steps=4` engine-wide for continuous batching. The two-step sCM and STTS checkpoints read Euler, NFE `2`, CFG `0`, `tau_mid`, and (for STTS) cadence from the artifact and reject incompatible overrides.

## 💡 Usage Tips

- **Keep the reference audio around 10s**. Longer audio won't yield better results.
- **`--prompt-text` should match what's actually spoken in the reference audio**. Mismatches degrade stability and may cause word-level errors.
- **Higher-quality references give better clones** — prefer a high sample rate, low background noise, no trailing noise, and natural-sounding speech.
- **Try different `--seed` values for prosody variation**. Each seed produces a different rhythm and intonation — resample a few times if the default doesn't feel right.
- **For flow-matching checkpoints, increase `--num-steps` if quality isn't good enough**. Fixed-step MeanFlow artifacts reject incompatible sampling overrides.
- **Force a pronunciation with Pinyin for polyphones.** Replace the character in the input text with its tone-marked pinyin — e.g. write `我生平不hào此道` to force `好` to be read as `hào`. Use tone-marked pinyin only (`hǎo`, `hào`, `bā`); numbered forms like `hao4` or `ha4o` are **not** recognized. Useful when reseeding doesn't fix a polyphone misread.

---

## 🏛 Architecture

A frozen **AudioVAE** encodes 48 kHz mono waveform into a continuous latent and decodes it back via a BigVGAN-style causal decoder. An **autoregressive backbone** predicts that latent one patch at a time, in three components:

- **Semantic encoder** — re-encodes each newly generated VAE patch into a compact embedding for the LLM, stripping high-variance acoustic detail.
- **LLM** — initialized from **Qwen2.5-1.5B-Base**, consumes BPE text directly (no phonemes), and emits one hidden state per audio step.
- **AR flow-matching head** — a DiT that conditions on the LLM hidden state and the AR prefix to denoise the next VAE patch, with a frozen CAM++ speaker x-vector as side input.

Two sequence layouts: *plain mode* places the full text as a prefix before the audio span (standard TTS); *[double-streaming interleaved mode](scripts/example_double_streaming.py)* lets a caller push BPE text tokens incrementally while audio patches are decoded online using the checkpoint's declared streaming cadence. See the technical report for full architectural and training details.

---

## 📊 Performance

Baselines are taken from original publications or default-configuration open-source releases.

### Seed-TTS-Eval

Zero-shot, ~3 s reference prompt, scored by the benchmark's reference ASR and WavLM-SV similarity.

| Model | Params | test-en WER↓ / SIM↑ | test-zh WER↓ / SIM↑ | test-zh-hard WER↓ / SIM↑ | **Avg WER↓ / SIM↑** |
|---|---:|:---:|:---:|:---:|:---:|
| CosyVoice 3 | 1.5B | 2.22 / 72.0 | 1.12 / 78.1 | **5.83** / 75.8 | 3.06 / 75.3 |
| DiTAR | 0.6B | 1.69 / 73.5 | 1.02 / 75.3 | — | — |
| F5-TTS | 0.3B | 2.00 / 67.0 | 1.53 / 76.0 | 8.67 / 71.3 | 4.10 / 71.4 |
| FireRedTTS-2 | 1.5B | 1.95 / 66.5 | 1.14 / 73.6 | 8.98 / 70.3 | 4.02 / 70.1 |
| IndexTTS 2 | 1.5B | 2.23 / 70.6 | 1.03 / 76.5 | 7.12 / 75.5 | 3.46 / 74.2 |
| MegaTTS 3 | 0.5B | 2.79 / 77.1 | 1.52 / 79.0 | — | — |
| MiniMax-Speech | — | 1.65 / 69.2 | **0.83** / 78.3 | — | — |
| Qwen3-TTS | 1.7B | **1.23** / 71.7 | 1.22 / 77.0 | 6.76 / 74.8 | 3.07 / 74.5 |
| Seed-TTS | — | 2.25 / 76.2 | 1.12 / 79.6 | 7.59 / 77.6 | 3.65 / 77.8 |
| VibeVoice | 1.5B | 3.04 / 68.9 | 1.16 / 74.4 | — | — |
| VoxCPM 2 | 2B | 1.84 / 75.3 | 0.97 / 79.5 | 8.13 / 75.3 | 3.65 / 76.7 |
| **dots.tts (Pretrain)** | **2B** | 1.34 / 76.8 | 0.96 / 80.5 | 6.46 / 79.2 | **2.92** / 78.8 |
| **dots.tts (SOAR)** | **2B** | 1.30 / **77.1** | 0.94 / **81.0** | 6.60 / **79.5** | 2.95 / **79.2** |
| **dots.tts (MF, NFE=4)** | **2B** | 1.29 / 76.2 | 0.94 / 80.0 | 6.60 / 78.5 | 2.94 / 78.2 |
| **dots.tts (MF-2steps)** | **2B** | 1.64 / 76.4 | 1.00 / 80.4 | 6.43 / 78.5 | 3.02 / 78.4 |
| **dots.tts (MF-1step)** | **2B** | 1.59 / 76.6 | 1.02 / 80.2 | 6.63 / 78.1 | 3.08 / 78.3 |
| **dots.tts (MF-2steps-STTS)** | **2B** | 1.41 / 75.3 | 1.04 / 79.2 | 7.83 / 76.9 | 3.43 / 77.1 |

### MiniMax Multilingual (24 languages)

Per-language WER / SIM on the MiniMax-Speech multilingual test set (100 utterances × 2 reference speakers per language). **Highest average SIM (83.9, SOAR)**, with a dots.tts variant taking the per-language SIM lead outright on 19 of 24 languages and tying on 2 more. Content fidelity is on par with the strongest systems on high-resource / Western European splits, and trails on low-resource long-tail languages where SIM is still preserved.

<details>
<summary><b>Per-language WER / SIM (click to expand)</b></summary>

| Language | MiniMax | ElevenLabs | Fish-Audio S2 | VoxCPM 2 | **dots.tts (Pre.)** | **dots.tts (SOAR)** | **dots.tts (MF$_4$)** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Arabic | **1.67** / 73.6 | **1.67** / 70.6 | 3.50 / 75.0 | 13.05 / **79.1** | 37.91 / 77.5 | 36.19 / **79.1** | 39.65 / 77.6 |
| Cantonese* | 34.11 / 77.8 | 51.51 / 67.0 | 30.67 / 80.5 | 38.58 / 83.5 | 37.91 / 84.7 | 42.32 / **85.0** | 37.82 / 84.0 |
| Chinese | 2.25 / 78.0 | 16.03 / 67.7 | **0.73** / 81.6 | 1.14 / **82.5** | 1.08 / 82.3 | 0.77 / **82.5** | 1.01 / 81.8 |
| Czech | 3.88 / 79.6 | **2.11** / 68.5 | 2.84 / 79.8 | 24.13 / 78.3 | 5.05 / 83.8 | 4.25 / **84.2** | 5.67 / 83.9 |
| Dutch | 1.14 / 73.8 | **0.80** / 68.0 | 0.99 / 73.0 | 0.91 / 80.8 | 1.20 / 81.4 | 1.39 / **82.2** | 1.30 / 82.1 |
| English | 2.16 / 75.6 | 2.34 / 61.3 | 1.62 / 79.7 | 2.29 / 85.4 | 1.06 / 86.9 | **1.03** / **87.5** | 1.09 / 86.9 |
| Finnish | 4.67 / 83.5 | 2.96 / 75.9 | 3.33 / 81.9 | **2.63** / **89.0** | 3.44 / 88.0 | 4.08 / 88.3 | 3.61 / 88.3 |
| French | 4.10 / 62.8 | 5.22 / 53.5 | **3.05** / 69.8 | 4.53 / 73.5 | 3.82 / 78.2 | 3.56 / **78.6** | 3.26 / 78.5 |
| German | 1.91 / 73.3 | 0.57 / 61.4 | **0.55** / 76.7 | 0.68 / 80.3 | 1.03 / 79.5 | 1.70 / **80.6** | 0.91 / 79.5 |
| Greek | 2.02 / 82.6 | **0.99** / 73.3 | 5.74 / 79.5 | 2.84 / 86.0 | 2.97 / **87.6** | 3.00 / **87.6** | 3.19 / 87.3 |
| Hindi | 6.96 / 81.8 | **5.83** / 73.0 | 14.64 / 82.1 | 19.70 / **85.6** | 14.32 / 84.5 | 14.24 / 84.7 | 14.75 / 84.8 |
| Indonesian | 1.24 / 72.9 | **1.06** / 66.0 | 1.46 / 76.3 | 1.08 / 80.0 | 2.71 / 80.8 | 2.96 / 80.8 | 3.91 / **81.2** |
| Italian | 1.54 / 69.9 | 1.74 / 57.9 | **1.27** / 74.7 | 1.56 / 78.0 | 3.16 / 84.5 | 3.12 / **84.7** | 2.16 / 84.3 |
| Japanese | 3.52 / 77.6 | 10.65 / 73.8 | **2.76** / 79.6 | 4.63 / 82.8 | 7.16 / 83.1 | 5.28 / **83.7** | 5.17 / 83.1 |
| Korean | 1.75 / 77.6 | 1.87 / 70.0 | **1.18** / 81.7 | 1.96 / 83.3 | 5.30 / 84.3 | 5.66 / 83.6 | 3.93 / **84.9** |
| Polish | 1.42 / 80.2 | **0.77** / 72.9 | 1.26 / 81.9 | 1.14 / **88.4** | 2.72 / 87.3 | 3.59 / 87.8 | 3.42 / 87.5 |
| Portuguese | 1.88 / 80.5 | 1.33 / 71.1 | **1.14** / 78.1 | 1.94 / 83.7 | 1.64 / 83.1 | 2.00 / **84.3** | 2.40 / 83.1 |
| Romanian | 2.88 / 80.9 | **1.35** / 69.9 | 10.74 / 73.3 | 21.58 / 79.7 | 3.36 / 86.2 | 3.87 / **87.1** | 3.38 / 86.1 |
| Russian | 4.28 / 76.1 | 3.88 / 67.6 | **2.40** / 79.0 | 3.63 / 81.1 | 3.64 / 83.0 | 4.28 / **83.2** | 4.42 / **83.2** |
| Spanish | 1.03 / 76.2 | 1.08 / 61.5 | 0.91 / 77.6 | 1.44 / 83.1 | 0.96 / 83.9 | 1.27 / **84.0** | **0.80** / **84.0** |
| Thai | **2.70** / 80.0 | 73.94 / 58.8 | 4.23 / 78.6 | 2.96 / 84.0 | 7.45 / 83.8 | 7.86 / 83.9 | 8.03 / **84.2** |
| Turkish | 1.52 / 77.9 | **0.70** / 59.6 | 0.87 / 83.5 | 0.82 / 87.1 | 5.45 / **87.4** | 4.96 / 87.3 | 6.20 / 86.8 |
| Ukrainian | 1.08 / 73.0 | **1.00** / 64.7 | 2.30 / 74.7 | 6.32 / 79.8 | 1.61 / 80.5 | 1.27 / **81.2** | 1.66 / 80.0 |
| Vietnamese | **0.88** / 74.3 | 73.42 / 36.9 | 7.41 / 74.0 | 3.31 / 80.6 | 3.85 / 80.7 | 3.89 / **81.6** | 5.43 / 80.5 |
| **Average** | **2.8** / 76.6 | 7.5 / 65.5 | 3.7 / 78.0 | 5.7 / 82.3 | 6.6 / 83.5 | 6.8 / **83.9** | 6.8 / 83.5 |

</details>

<sub>*Cantonese WER reflects an ASR-faithfulness floor common to all systems; SIM remains comparable.</sub>

### CV3-Eval

Hard-subset Chinese/English plus a cross-lingual voice-cloning split. **Takes the table top on hard-en (MF$_4$ at 4.37) and leads both cross-lingual SIM subsets (SOAR at 75.0 / 72.8)**, with the post-trained variants bracketing the prior leader on the hardest English subset.

| Model | zh W↓ | en W↓ | hard-zh W↓ | hard-en W↓ | en→zh W↓ / S↑ | zh→en W↓ / S↑ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| CosyVoice 2 | 4.08 | 6.32 | 12.58 | 11.96 | 13.50 / 63.3 | 6.47 / 64.3 |
| CosyVoice 3 (1.5B) | 3.91 | 4.99 | 9.77 | 10.55 | **8.01** / 66.9 | **4.32** / 66.4 |
| Fish-Audio S2 | **2.65** | **2.43** | 9.10 | 4.40 | — | — |
| VoxCPM 2 | 3.65 | 5.00 | **8.55** | 8.48 | — | — |
| **dots.tts (Pretrain)** | 3.51 | 5.24 | 9.69 | 5.99 | 10.88 / 74.6 | 4.97 / 71.9 |
| **dots.tts (SOAR)** | 3.71 | 4.50 | 9.22 | 4.49 | 10.75 / **75.0** | 5.66 / **72.8** |
| **dots.tts (MF, NFE=4)** | 3.95 | 4.05 | 9.10 | **4.37** | 10.73 / 73.8 | 5.24 / 70.9 |

### EmergentTTS-Eval

Win-rate judged head-to-head against `gpt-4o-mini-tts` by Gemini-2.5-Pro-0506 across six expressiveness-oriented scenarios. **SOAR takes the top Syntactic Complexity score in the table (65.7%) — above every closed-source system** — and Pretrain posts the **best Emotions score among open-source systems (72.7%)**.

| Model | Voice | WER↓ | Overall↑ | Emotions↑ | Paraling.↑ | Foreign↑ | C. Pron.↑ | Quest.↑ | Syntax↑ |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gemini-2.5-Flash-TTS\* | Zephyr | 10.39 | **70.7%** | **95.9%** | **91.3%** | 58.5% | 55.7% | **63.0%** | 57.9% |
| Gemini-2.5-Pro-TTS\* | Zephyr | 11.79 | 69.3% | 86.9% | 82.3% | 58.2% | **64.8%** | 61.3% | 61.8% |
| gpt-4o-audio-preview\* | Ballad | 11.87 | 65.2% | 88.8% | 82.1% | **60.2%** | 40.4% | 57.0% | 59.5% |
| gpt-4o-mini-tts\* | Alloy | 10.76 | 56.3% | 59.2% | 58.8% | 57.3% | 52.4% | 52.7% | 57.1% |
| *baseline: gpt-4o-mini-tts* | Alloy | 10.61 | 50.0% | — | — | — | — | — | — |
| **dots.tts (Pretrain)** | basic\_ref\_en | 10.86 | 49.2% | 72.7% | 54.7% | 39.5% | 18.0% | 48.4% | 58.4% |
| **dots.tts (MF4)** | basic\_ref\_en | 11.75 | 47.9% | 59.8% | 55.2% | 36.3% | 16.7% | 50.5% | 64.8% |
| **dots.tts (SOAR)** | basic\_ref\_en | 10.45 | 47.6% | 63.9% | 52.7% | 39.4% | 16.4% | 47.0% | **65.7%** |
| Qwen3-TTS | basic\_ref\_en | 17.32 | 42.8% | 39.8% | 50.7% | 25.4% | 30.0% | 48.9% | 60.4% |
| HumeAI\* | — | 12.85 | 42.7% | 61.6% | 36.9% | 34.6% | 34.3% | 43.2% | 44.6% |
| Qwen3-TTS | Ryan | 19.65 | 42.3% | 60.5% | 62.7% | 17.1% | 9.8% | 56.4% | 43.0% |
| VoxCPM 2 | basic\_ref\_en | 11.84 | 41.1% | 42.3% | 44.1% | 33.3% | 18.6% | 53.4% | 52.3% |
| MiniMax/speech-02-hd\* | EN-narr | **10.02** | 36.6% | 40.9% | 34.3% | 34.3% | 16.3% | 47.3% | 43.9% |
| 11Labs Multilingual v2\* | Brian | 11.19 | 33.9% | 30.4% | 45.5% | 35.5% | 14.5% | 39.5% | 35.5% |
| F5-TTS | basic\_ref\_en | 16.47 | 15.3% | 26.8% | 21.6% | 1.8% | 1.4% | 14.8% | 23.8% |

<sub>\* Closed-source / commercial. Table shows a selected subset for brevity — for the full leaderboard, see [EmergentTTS-Eval-public](https://github.com/boson-ai/EmergentTTS-Eval-public/blob/main/LEADERBOARD_gemini-2.5-pro-05-06.md).</sub>

---

## ⚡ Efficiency

Streaming-inference benchmarks under `--optimize` on a Seed-TTS-Eval mix (100 utterances across zh / en / zh-hard, first post-warmup request excluded, N=99 per group). `voice_cloning` uses reference audio + transcript; `text_only` uses text with no reference. Common config: `precision=bfloat16`, `guidance_scale=1.2`, `seed=42`; SOAR uses `num_steps=10`, MF uses `num_steps=4`. Hardware / stack: single H800, torch 2.8 + CUDA 12.8. The `--optimize` path also accelerates non-streaming `generate()` calls; numbers below are the streaming path.

> **Note:** `--optimize` triggers a one-shot `torch.compile` warmup that walks every DiT compile bucket + KvPrefill + vocoder chunk sizes. Cold start takes **~3 minutes** on H800; every subsequent request runs at the steady-state RTF above. Pass `warmup_on_optimize=False` to `DotsTtsRuntime` if you want to skip warmup and accept the first request paying the compile cost.

### Steady-State Latency

| Group | audio mean (s) | latency p50 / p90 (s) | first-chunk p50 / p90 (ms) | RTF mean / p50 / p90 | peak alloc (GB) |
|---|---:|---:|---:|---:|---:|
| SOAR / voice_cloning | 7.57 | 1.13 / 3.04 | 225 / 404 | 0.21 / 0.20 / 0.26 | 7.86 |
| SOAR / text_only     | 7.78 | 0.95 / 3.02 |  69 /  79 | 0.18 / 0.18 / 0.20 | 7.85 |
| MF   / voice_cloning | 7.46 | 0.88 / 1.73 | 204 / 381 | 0.16 / 0.15 / 0.21 | 5.74 |
| MF   / text_only     | 7.65 | 0.68 / 2.06 |  68 /  78 | 0.13 / 0.13 / 0.15 | 5.73 |


### Memory Footprint by Length Bucket

Bucket = total prompt + generated audio in latent patches (one patch ≈ 160 ms).

| Bucket | Total audio cap | SOAR / voice_cloning | SOAR / text_only | MF / voice_cloning | MF / text_only |
|---|---:|---:|---:|---:|---:|
| <64 patches  | <10.24s | 5.65 GB | 5.64 GB | 5.30 GB | 5.29 GB |
| <128 patches | <20.48s | 6.53 GB | 6.52 GB | 5.47 GB | 5.46 GB |
| <256 patches | <40.96s | 7.86 GB | 7.85 GB | 5.74 GB | 5.73 GB |
| <512 patches | <81.92s | 10.51 GB\* | 10.51 GB\* | 6.29 GB\* | N/A\*\* |

\* From explicit long-audio probes (actual spans within 256–512 patches).  
\*\* `mf / text_only` did not reach the 256–512 bucket under either synthetic (x4) or real long-text probes; longest observed 237 patches at 5.73 GB.

### SGLang Omni Efficiency

Serving throughput on Seed-TTS-Eval **EN** against a single Omni server started from `examples/configs/dots_tts.yaml` (`max_running_requests=16`, bf16, `num_steps=4`, backbone decode CUDA graph + graph-captured acoustic tail). Each row is the mean of two runs, seed 42. Hardware: **1× H100**. Full write-up: [SGLang Omni cookbook — Performance](https://sgl-project.github.io/sglang-omni/cookbook/dots_tts.html#performance).


| Concurrency | Samples | Throughput (req/s) | audio_s/s | Mean latency | RTF (per-req)| WER |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1088 | 0.935 | 3.726 | 1.070 | 0.275 | 1.241% |
| 2 | 1088 | 1.556 | 6.493 | 1.286 | 0.314 | 1.256% |
| 4 | 1088 | 2.493 | 10.407 | 1.603 | 0.390 | 1.264% |
| 8 | 1088 | 3.875 | 16.173 | 2.062 | 0.502 | 1.323% |
| 16 | 1088 | 4.760 | 19.859 | 3.344 | 0.812 | 1.348% |
| 32 | 1088 | 4.988 | 20.818 | 6.344 | 1.596 | 1.331% |

Zero failed requests in every run, and no sample above 50% WER. WER is measured with `Qwen/Qwen3-ASR-1.7B` on the first run of each row.

To reproduce (server already running as above):

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model dots-studio/dots.tts-mf \
  --ref-format references \
  --base-url http://127.0.0.1:8000 --port 8000 \
  --lang en --max-concurrency 16 --warmup 8 --seed 42 \
  --generate-only --use-existing-server \
  --output-dir results/dots-seedtts-en-c16

python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model dots-studio/dots.tts-mf \
  --ref-format references --lang en --seed 42 \
  --transcribe-only --port 8000 \
  --output-dir results/dots-seedtts-en-c16
```

## 🤝 Community Projects

Third-party ports and integrations of dots.tts, maintained by the community.

| Project | Description | Maintainer |
|---|---|---|
| [sglang-omni](https://github.com/sgl-project/sglang-omni) | High-concurrency serving for dots.tts ([cookbook](https://sgl-project.github.io/sglang-omni/cookbook/dots_tts.html)) | [@sgl-project](https://github.com/sgl-project) |
| [audio.cpp](https://github.com/0xShug0/audio.cpp) | ggml-based unified C++ inference framework — CPU/CUDA/Vulkan/Metal, CLI & server, no Python | [@0xShug0](https://github.com/0xShug0)
| [dots-tts-mlx](https://github.com/sb1992/dots-tts-mlx) | Pure-MLX inference port for Apple Silicon (Python) | [@sb1992](https://github.com/sb1992) |
| [mlx-swift-dots-tts](https://github.com/sammcj/mlx-swift-dots-tts) | Native MLX Swift port for Apple Silicon (no Python runtime) | [@sammcj](https://github.com/sammcj) |
| [Dots-TTS-ComfyUI](https://github.com/Saganaki22/Dots-TTS-ComfyUI) | ComfyUI custom nodes for TTS, voice cloning, and Whisper transcription | [@Saganaki22](https://github.com/Saganaki22) |

---

## ⚠️ Risks and Limitations

- **Misuse risk.** High-fidelity zero-shot voice cloning can produce highly realistic synthetic speech. The released checkpoints are intended for research and authorized deployment. Do **not** use dots.tts for impersonation, fraud, or disinformation. Combine downstream use with consent-aware reference-audio policies, robust synthetic-speech detection, and content watermarking. Clearly mark AI-generated audio.
- **Low-resource WER gap.** A BPE backbone inherits the text LLM's language coverage at the cost of a higher data appetite. On script-divergent and under-represented languages (Arabic, Hindi, Turkish, Vietnamese) the WER gap visible on the MiniMax benchmark reflects this, and the same long tail surfaces on the Foreign Words and Complex Pronunciation scenarios of EmergentTTS-Eval. Speaker similarity is preserved across these languages.
- **Speech-heavy training.** Although the AudioVAE is trained at 48 kHz and is modality-agnostic in principle, the backbone is trained on a speech-heavy mixture. Singing and unified speech + sound generation are not covered in this release.

---

## 📖 Citation

If you find dots.tts or dots.tts.edit useful, please consider citing the corresponding technical report and starring the repository.

```bibtex
@article{dotstts2026,
  title         = {dots.tts Technical Report},
  author        = {dots.tts Team},
  year          = {2026},
  eprint        = {2606.07080},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
}

@article{wang2026dotsttsedit,
  title         = {dots.tts.edit: Precisely Controlled Speech Editing with a Continuous Autoregressive Model},
  author        = {Wang, Hankun and Li, Bohan and Lian, Shi and Gu, Xiaoyu and Peng, Jing and Zheng, Da and Zhang, Colin and Yu, Kai},
  year          = {2026},
  eprint        = {2608.02673},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
}
```

## 📄 License

dots.tts code and released checkpoints are licensed under [Apache-2.0](LICENSE).

## 🙏 Acknowledgments

- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) — LLM backbone initialization.
- [DiTAR](https://arxiv.org/abs/2502.03930) and [ARDiT](https://arxiv.org/abs/2406.05551) — for the continuous-AR + per-patch diffusion design.
- [HoliTok](https://github.com/bovod-sjtu/HoliTok) — for the AudioVAE design.
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) — for the vocoder design.
- [CAM++](https://github.com/alibaba-damo-academy/3D-Speaker) — for speaker x-vector encoder.
