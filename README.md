# Local Image Edit — Apple Silicon (M5)

Local, fully offline, text-guided image editing. Upload a photo, say what you
want changed, and keep refining — each edit builds on the last, like a
conversation with the image.

Runs **FLUX.2 Klein 9B at 8-bit** on the Apple GPU via Metal — or on an NVIDIA
GPU under Linux. No cloud APIs, no telemetry. Once the weights are downloaded
you can unplug the network.

Qwen-Image-Edit is also supported, but does not fit in 24 GB — see
[Choosing a model](#choosing-a-model).

```
python app.py
```

---

## What "interactive" means here

The editing loop is a conversation, not a one-shot form:

```
┌── Source ────────┬── Thread ──────────────────┬── Compare ────────┐
│                  │  you: make the sky sunset  │  ◀ before│after ▶ │
│   [ upload ]     │       ✓ 42s                │                   │
│                  │  you: now add birds        │  ○ vs previous    │
│  Settings ▸      │       ⏳ step 11/25 ~38s   │  ○ vs original    │
│                  │       [ live preview ]     │  ○ vs pinned      │
│  Other branches  ├────────────────────────────┤                   │
│  [▪][▪][▪]       │  next instruction…  [Stop] │  [⬇ Save]         │
└──────────────────┴────────────────────────────┴───────────────────┘
```

* **Chained edits.** Every instruction operates on the result of the previous
  one. Ask for a sunset, then add birds, then make it cinematic.
* **Branching, not truncation.** Click any earlier step and edit from there.
  The work you navigated away from is *not* destroyed — it stays reachable in
  the "Other branches" gallery. Each edit costs 30–90 seconds, so silently
  discarding steps would be punitive.
* **Live previews.** The latents are decoded at roughly 20%, 45%, and 75% of
  the run. If the model misread your instruction you can see it — and press
  Stop — about a minute before the final image would have arrived.
* **Stop keeps the result.** Stopping decodes the partially denoised image and
  files it as a normal step rather than throwing the work away.
* **Safe mid-run navigation.** The parent of an edit is bound when you press
  Edit. Reverting or browsing while the GPU is busy can't misattach the result.

Buttons under the composer (`Revert here`, `Pin`, `Reuse prompt`, `Undo`,
`Save`) act on whichever message you last clicked in the thread.

### Example instructions

```
Replace the background with a futuristic city
Remove the person in the background
Change the shirt colour to blue
Turn this photo into a cinematic still with dramatic lighting
```

Qwen-Image-Edit responds well to explicit preservation clauses — adding
"Keep everything else unchanged" measurably reduces unwanted drift.

---

## Requirements

| | |
|---|---|
| Hardware | Apple Silicon Mac (M1 or later). Developed and tested on **M5, 24 GB**. |
| macOS | 13 Ventura or later (26.x tested) |
| Python | 3.11+ (3.12 recommended) |
| Disk | ~22 GB free — the default model is ~18 GB |
| RAM | 16 GB workable; **24 GB comfortable** with the default model |

Intel Macs are not supported for practical use: there is no Metal GPU path, and
CPU-only inference takes well over an hour per image.

### Linux

Linux works too, with caveats. MLX publishes Linux wheels
(`manylinux_2_35_x86_64` and `aarch64`) and mflux declares `mlx[cuda13]` on
Linux, so the same 6-bit weights run on an **NVIDIA GPU via CUDA 13**. `setup.sh`
and `run.sh` detect this and install the right wheel automatically —
`requirements.txt` needs no changes, because mflux's own dependency markers
select the CUDA build.

| | |
|---|---|
| GPU | NVIDIA, CUDA 13. **≥24 GB VRAM** for 768 px; ≥40 GB to run without `memory.mode: low` |
| Driver | Recent enough for CUDA 13 |
| Python | 3.11+ |

Two honest caveats:

* **Untested by the author.** Everything here was verified on an M5. The Linux
  path follows from MLX's published platform support and is exercised by unit
  tests, but no image has been generated on it.
* **VRAM, not unified memory.** On Apple Silicon the 32 GB working set spills
  into system RAM and merely gets slow. On a discrete GPU it OOMs instead.
  `memory.mode: low` matters more here, not less — it bounds the denoise phase
  to ~16.9 GB, which is what makes a 24 GB card viable at all.

AMD/ROCm and Apple Intel have no MLX GPU backend, so both fall back to CPU.

---

## Install

On a new machine, one command does everything — environment, dependencies, and
the ~18 GB of weights:

```bash
./setup.sh
```

It is safe to interrupt and re-run; both the dependency install and the
download resume rather than restarting.

| Command | Use when |
|---|---|
| `./setup.sh` | Fresh machine, full setup |
| `./setup.sh --no-model` | Set up the environment, fetch weights later |
| `./setup.sh --model-only` | Weights only (dependencies already installed) |
| `./setup.sh --mirror <url>` | huggingface.co is slow or blocked |
| `./setup.sh --import-from <dir>` | Copy weights from another machine |
| `./setup.sh --check` | Verify an existing install |
| `./setup.sh --jobs 8` | More parallel download workers |
| `./setup.sh --no-xet` | Force plain HTTP transfer |

### Manual install

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python app.py
```

The UI opens at <http://127.0.0.1:7860>.

`./run.sh` creates the virtualenv, installs dependencies only when
`requirements.txt` changes, and forwards any flags to `app.py`.

### Setting up on a slow or restricted network

The download retries with backoff, resumes from partial files, and drops to
plain HTTP automatically if a transfer makes no headway (HuggingFace's Xet
backend has been observed throttling to tens of KB/s). If it still won't
complete:

```bash
# Use a Hub mirror
./setup.sh --model-only --mirror https://hf-mirror.com

# Or go through a proxy — standard env vars are respected
HTTPS_PROXY=http://proxy.internal:3128 ./setup.sh --model-only
```

**Sneakernet.** If the machine has poor connectivity, copy the weights from one
that already has them. On the source machine the files live in `models/hf/`:

```bash
# On the machine that already works
rsync -a --info=progress2 models/ user@newmachine:/tmp/qwen-weights/

# On the new machine
./setup.sh --import-from /tmp/qwen-weights
```

`--import-from` accepts either a full cache tree or a bare snapshot directory
containing `transformer/`, `text_encoder/`, and `vae/`. It verifies every shard
before and after copying, so a bad USB copy fails loudly instead of surfacing
as a cryptic error at load time.

### First-time model download

Nothing is downloaded at install time. On the **first edit** the app fetches
~18 GB from HuggingFace into `models/hf/` (configurable via `model.cache_dir`).
Expect 20–45 minutes on a fast connection. Downloads resume if interrupted.

To fetch the weights up front instead of on first edit:

```bash
python app.py --preload
```

**Monitoring the download.** The download runs inside the app, so there is no
progress bar in your shell. From a second terminal:

```bash
./watch-download.py
```

```
  text_encoder  ███████████░░░░░░░░░  59.0%    9.14 / 15.49 GB
  tokenizer     ████████████████████ 100.0%    0.01 /  0.01 GB
  transformer   ░░░░░░░░░░░░░░░░░░░░   0.0%    0.00 / 16.61 GB
  vae           ░░░░░░░░░░░░░░░░░░░░   0.0%    0.00 /  0.25 GB

  Total         9.15 / 32.37 GB  (28.3%)
  In flight     9.80 GB partially written
  7.3 MB/s · ETA ~31 min · elapsed 4 min
```

It redraws in place, smooths the throughput estimate (HuggingFace rates are
spiky), and warns if nothing has moved for two minutes. Watching is read-only —
Ctrl-C never affects the download itself. No activation needed; it re-execs
into `venv/` automatically.

| Command | Behaviour |
|---|---|
| `./watch-download.py` | Live, refreshes every 5 s until complete |
| `./watch-download.py --interval 30` | Gentler refresh |
| `./watch-download.py --once` | Print once and exit |
| `python app.py --status` | Same one-shot report, via the main CLI |
| `python app.py --status --watch` | Same live view, via the main CLI |

Exit codes make it scriptable — `0` complete, `1` still downloading, `130`
interrupted:

```bash
./watch-download.py && python app.py    # launch the moment weights land
```

Two things that look like problems but aren't:

* **`transformer` sits at 0% for a long time.** Files download alphabetically,
  so `text_encoder` (15.5 GB) completes before `transformer` (16.6 GB) starts.
* **"In flight" is large while "Total" barely moves.** A 2 GB shard counts as
  0% complete until its last byte lands. The in-flight figure is the honest
  view of current progress.

Other useful checks:

```bash
python app.py --check     # device/backend report, no model load
python app.py --verify    # confirm cached weight shards aren't truncated
```

If a download stalls, interrupt it and re-run — it resumes from where it
stopped. HuggingFace's Xet backend has been known to throttle; setting
`HF_HUB_DISABLE_XET=1` falls back to plain HTTP, which is often faster.

---

## Recommended settings for M5 (24 GB)

| Setting | Value | Note |
|---|---|---|
| Resolution | **Balanced — match input, 0.6 MP** | Best quality-per-second on a 10-core GPU |
| Steps | **28** | 12–16 for quick iteration, 40+ for finals |
| Guidance | **1.0** | FLUX.2 is distilled — it wants ~1.0, not the 4.0 a Qwen/SD model needs |
| Memory mode | **`balanced`** | The model fits; nothing needs evicting |

FLUX.2 **does not accept a negative prompt** — the field is ignored for this
family. Describe what you *want* instead of what you don't.

The "match input" presets size to a pixel budget while **preserving the source
aspect ratio** — a landscape photo stays landscape. Prefer these; the fixed
`Square` / `Portrait` / `Landscape` presets are for deliberately reframing, and
will stretch an image whose shape doesn't match.

Workflow that wastes the least time: explore wording at *Draft — 0.3 MP*, then
re-run the instruction you settled on at *Quality — 1.0 MP* with the same seed.

---

## Choosing a model

Models are switchable at runtime — pick one from the **Model** dropdown in the
UI and the current model is unloaded and the new one loaded, no restart needed.
Weights download on first selection.

```bash
python app.py --list-models        # what's available, sizes, what's cached
python app.py --model flux2-klein-4b
```

```
  Selectable models  (machine has 24 GB)

  * flux2-klein-9b    17.8 GB  fits     cached          FLUX.2 Klein 9B (8-bit)
      Best quality that fits a 24 GB machine. Recommended.
    flux2-klein-4b     6.7 GB  fits     not downloaded  FLUX.2 Klein 4B (6-bit)
      Fastest. Lower fidelity; good for exploring wording.
    qwen-6bit         32.4 GB  TOO BIG  cached          Qwen-Image-Edit 2509 (6-bit)
      Needs 40 GB+. On 24 GB it swaps badly — ~116 s per step.
```

The dropdown labels carry the size and a warning when a model is larger than
the machine, so the consequence is visible before you pick it.

Selecting a model also moves the settings that belong to it: FLUX.2 wants
`guidance ≈ 1.0` and **rejects negative prompts**, so that field is hidden;
Qwen wants `guidance ≈ 4.0` and keeps it. `memory.mode: auto` re-decides
per model whether the text encoder needs evicting.

Add your own in `config.yaml` — `repo_id` and `family` must match, and startup
validates that they do:

```yaml
model:
  active: "flux2-klein-9b"
  catalog:
    - id: "flux2-klein-9b"
      label: "FLUX.2 Klein 9B (8-bit)"
      repo_id: "mlx-community/flux2-klein-9b-8bit"
      family: "flux2-klein-edit"
      notes: "Best quality that fits a 24 GB machine."
```

| `family` | Pipeline | Working set |
|---|---|---|
| `flux2-klein-edit` | FLUX.2 Klein 9B | 17.9 GB |
| `flux2-klein-edit-4b` | FLUX.2 Klein 4B | 6.7 GB |
| `qwen-image-edit` | Qwen-Image-Edit 2509 | 32.4 GB |

### Running out of GPU memory

On Apple Silicon an oversized model spills into system RAM and merely gets
slow. On a **discrete GPU there is no swap** — exceeding VRAM aborts the
process from inside the CUDA allocator:

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  cudaMallocAsync(&data, size, stream) failed: out of memory
```

That is a C++ abort, not a Python exception, so no handler can catch it. The
app therefore does two things on a discrete GPU:

* **Preflight.** Before loading, it compares the model's working set (or its
  denoise peak in `low` mode) plus ~2 GB of activation headroom against
  available VRAM, and refuses with an explanation naming smaller models.
* **A memory limit.** `mx.set_memory_limit()` is set to ~92% of VRAM, so MLX
  raises a catchable Python error rather than letting the allocator abort.

If you hit this anyway, in order of effectiveness: switch to a smaller model,
set `memory.mode: low`, drop the resolution preset, lower `steps`.

### Why Qwen-Image-Edit does not fit in 24 GB

This is worth spelling out, because quantising harder does not fix it.

| Component | Qwen-Image-Edit 6-bit | FLUX.2 Klein 9B 8-bit |
|---|---|---|
| Transformer | 16.6 GB (6-bit) | 9.6 GB (8-bit) |
| Text encoder | **15.5 GB — bf16, not quantised** | 8.0 GB (8-bit) |
| VAE | 0.25 GB | 0.2 GB |
| **Total** | **32.4 GB** | **17.9 GB** |

mflux marks Qwen's text encoder `skip_quantization=True` because quantising it
noticeably degrades prompt understanding. That is a defensible trade, but it
sets a hard floor: even the 4-bit Qwen export still totals **27.3 GB**, because
only the transformer shrinks. Every FLUX.2 component is quantised, which is the
entire difference.

**Measured on an M5 / 24 GB**, Qwen-Image-Edit at 512 px, 12 steps:

```
peak memory   32.2 GB      ← the full working set, on a 24 GB machine
swap used     17.0 GB
pageins       20.5 million
per step      116 s        → 23 minutes for one edit
```

The output was correct; it was just unusable. If you have 40 GB or more,
`family: qwen-image-edit` is a reasonable choice.

### A note on lazy evaluation

If you run Qwen on a constrained machine, one detail matters enough to record.

MLX evaluates lazily. `_encode_prompts_with_images` does not return
embeddings — it returns an *unevaluated graph* that still references all
15.5 GB of the text encoder's weights. Dropping the module afterwards therefore
frees nothing: the weights stay alive until the graph is finally forced, at the
first `mx.eval()` **inside the denoise loop** — exactly when the transformer is
also materialising. Both are resident at once.

`ImageEditor._install_encode_barrier` inserts an explicit `mx.eval` on the
embeddings before eviction, collapsing the graph while the transformer is still
untouched. Without it, `memory.mode: low` silently does nothing.

---

## How it works, and why it isn't PyTorch MPS

The brief asked for PyTorch MPS. That path cannot deliver a quantised model on
Apple Silicon today:

* `bitsandbytes` is CUDA-only.
* `torchao` has no 6-bit kernels on Apple GPUs.
* diffusers' GGUF loader dequantises back to bf16 at compute time, so it saves
  disk, not memory.

These transformers are 9–20B parameters — tens of GB at bf16, well beyond
24 GB of unified memory. The quantised weights that do fit are published in
**mflux/MLX** format, and MLX runs on the same Apple GPU through the same Metal
backend. So the compute path is MLX/Metal rather than torch/MPS; the hardware
being driven is identical.

PyTorch MPS availability is still probed and printed at startup, because torch
comes along as an mflux dependency and it is the first thing people check when
asking "is my GPU actually being used?"

### Memory modes

| Mode | Behaviour | Use when |
|---|---|---|
| `auto` | Pick per model: `balanced` if it fits, `low` if not | **Default** — correct as you switch models |
| `balanced` | Everything stays resident | The model fits |
| `low` | Evict the text encoder after encoding, reload next turn | The model does not fit (Qwen under ~40 GB) |
| `off` | No management at all | Debugging |

> mflux ships a `MemorySaver` that performs the same eviction but never
> reloads, so a second edit with a new prompt would fail. This app owns that
> lifecycle itself (`src/model_loader.py`).

---

## Configuration

Everything lives in `config.yaml`; there are no hardcoded paths. Relative paths
resolve against the project directory, and `~` / `${ENV_VAR}` are expanded.

```yaml
model:
  repo_id: "mlx-community/flux2-klein-9b-8bit"   # or a local directory
  family:  "flux2-klein-edit"
  cache_dir: "./models/hf"

memory:
  mode: "auto"                # auto | balanced | low | off

generation:
  steps: 28
  guidance: 1.0

ui:
  server_port: 7860
```

Use a different file with `python app.py --config other.yaml` or
`QWEN_EDIT_CONFIG=other.yaml`.

### Verify weights after switching models

Repos can be published broken. `Norton0924/Qwen-Image-Edit-2509-6bit` ships
`text_encoder/2.safetensors` at 67 MB while its own header declares 2.06 GB of
tensor data — it cannot load, and MLX reports it as an opaque "invalid data
offsets" error. After changing `repo_id`, run:

```bash
python app.py --verify
```

which names the offending file instead.

---

## Local REST API

Enabled by default (`api.enabled`), bound to the same host and port as the UI.

```bash
# Edit an image, get a PNG back
curl -X POST http://127.0.0.1:7860/api/edit \
  -F image=@photo.jpg \
  -F 'instruction=change the shirt colour to blue' \
  -F steps=25 -F guidance=4.0 \
  -o edited.png

# Or JSON in / JSON out (base64)
curl -X POST http://127.0.0.1:7860/api/edit \
  -H 'Content-Type: application/json' \
  -d '{"image":"<base64>","instruction":"make it night time","steps":25}'

curl http://127.0.0.1:7860/api/status
```

`/api/status` reports device, backend, quantisation, memory mode, and whether a
generation is in flight.

---

## Outputs

```
outputs/
├── 20260729-201455-edit-make-the-sky-sunset.png
├── 20260729-201455-edit-make-the-sky-sunset.json   # prompt, seed, steps, lineage
├── sessions/<session-id>/                          # full edit tree, resumable
└── app.log
```

The metadata sidecar records the full `lineage` — every instruction that led to
that image — so a result five edits deep is reproducible.

---

## Project layout

```
qwen-image-edit-m5/
├── app.py              entrypoint, CLI flags, startup banner
├── config.yaml         all settings
├── requirements.txt
├── setup.sh            one-command install (deps + weights)
├── run.sh              venv bootstrap + launch
├── watch-download.py   live model-download monitor
├── models/             weight cache (gitignored)
├── outputs/            images, metadata, sessions, logs
├── src/
│   ├── config.py       typed config loading + validation
│   ├── families.py     per-model API/memory differences
│   ├── device.py       Apple Silicon detection, Metal probe, banner
│   ├── model_loader.py loading, memory modes, integrity verification
│   ├── inference.py    generation loop, previews, cancellation
│   ├── session.py      edit tree (branch/undo/pin/persist)
│   ├── ui.py           Gradio interface
│   ├── api.py          local REST API
│   └── utils.py        image IO, seeds, formatting
└── tests/              session + utility tests (no GPU needed)
```

```bash
python -m pytest tests/ -q
```

---

## Troubleshooting

**"Metal unavailable" / falls back to CPU**

```bash
python app.py --check
python -c "import mlx.core as mx; print(mx.metal.is_available())"
```

If this prints `False` on an Apple Silicon Mac, MLX installed a wrong-arch
wheel — usually from running under Rosetta. Confirm with
`python -c "import platform; print(platform.machine())"`; it must say `arm64`,
not `x86_64`. Rebuild the virtualenv with a native Python:

```bash
rm -rf venv && /opt/homebrew/bin/python3 -m venv venv
source venv/bin/activate && pip install -r requirements.txt
```

**Checking PyTorch MPS specifically**

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

Torch is only a transitive dependency here — inference does not use it — but
the startup banner reports its status since it's a common sanity check.

**Out of memory / heavy swapping**

First check which model you are on — `python app.py --check` prints the working
set and whether it fits. If it says *Tight* or *Constrained*, you are on a model
too large for this machine; switch to `flux2-klein-edit` (see
[Choosing a model](#choosing-a-model)).

If the model does fit and you are still short: drop to a smaller resolution
preset, lower `steps`, and quit other large apps. Watch the live figure in the
status line under the composer, or in `/api/status`.

**"invalid data offsets" / "incomplete download"**

A weight shard is truncated. Diagnose precisely with:

```bash
python app.py --verify
```

Delete the reported snapshot under `models/hf/hub/` and re-run to re-download.
If it fails again on the same file, the upstream repo itself is broken — switch
`model.repo_id`.

**First edit takes forever**

The first edit downloads the weights (~18 GB by default) and then loads them. Later launches load from
cache in tens of seconds. Use `python app.py --preload` to front-load this.

**Port already in use**

```bash
python app.py --port 7861
```

**Edits ignore part of the instruction**

Raise guidance to 5–6, increase steps to 30–40, and be explicit about what
should *not* change ("Keep the pose, background, and lighting unchanged").
Chaining two simple instructions usually beats one compound instruction.

---

## Licence and attribution

* Default model: [FLUX.2 Klein](https://huggingface.co/black-forest-labs), 8-bit MLX export by [mlx-community](https://huggingface.co/mlx-community/flux2-klein-9b-8bit)
* Also supported: [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) (Apache 2.0), 6-bit export by [OsaurusAI](https://huggingface.co/OsaurusAI/Qwen-Image-Edit-mflux-q6)
* Inference: [mflux](https://github.com/filipstrand/mflux) — MLX implementations of generative image models
* UI: [Gradio](https://gradio.app)
