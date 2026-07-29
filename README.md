# Qwen Image Edit — Apple Silicon (M5)

Local, fully offline, text-guided image editing. Upload a photo, say what you
want changed, and keep refining — each edit builds on the last, like a
conversation with the image.

Runs **Qwen-Image-Edit-2509 at 6-bit** on the Apple GPU via Metal — or on an
NVIDIA GPU under Linux. No cloud APIs, no telemetry. Once the weights are
downloaded you can unplug the network.

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
| Disk | ~35 GB free — the model is ~32 GB |
| RAM | 24 GB workable with `memory.mode: low`; 32 GB+ comfortable |

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
the ~32 GB of weights:

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
~32 GB from HuggingFace into `models/hf/` (configurable via `model.cache_dir`).
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
| Steps | **25** | 12–15 for quick iteration, 30+ for finals |
| Guidance | **4.0** | 2.5–3.0 for subtle edits, 5–6 to force compliance |
| Memory mode | **`low`** | Essential at 24 GB |

The "match input" presets size to a pixel budget while **preserving the source
aspect ratio** — a landscape photo stays landscape. Prefer these; the fixed
`Square` / `Portrait` / `Landscape` presets are for deliberately reframing, and
will stretch an image whose shape doesn't match.

Workflow that wastes the least time: explore wording at *Draft — 0.3 MP*, then
re-run the instruction you settled on at *Quality — 1.0 MP* with the same seed.

---

## How it works, and why it isn't PyTorch MPS

The brief asked for PyTorch MPS. That path cannot deliver a 6-bit model today:

* `bitsandbytes` is CUDA-only.
* `torchao` has no 6-bit kernels on Apple GPUs.
* diffusers' GGUF loader dequantises back to bf16 at compute time, so it saves
  disk, not memory.

Qwen-Image-Edit's transformer is 20B parameters — roughly **40 GB at bf16**,
which does not fit in 24 GB of unified memory. The only real 6-bit
Qwen-Image-Edit weights are published in **mflux/MLX** format, and MLX runs on
the same Apple GPU through the same Metal backend. So the compute path is
MLX/Metal rather than torch/MPS; the hardware being driven is identical.

PyTorch MPS availability is still probed and printed at startup, because torch
comes along as an mflux dependency and it is the first thing people check when
asking "is my GPU actually being used?"

### Memory

The model does not split evenly:

| Component | Size | Precision |
|---|---|---|
| Transformer | 16.6 GB | 6-bit (U32-packed + bf16 scales) |
| Text encoder | 15.5 GB | **bf16 — deliberately not quantised** |
| VAE | 0.25 GB | bf16 |
| **Total** | **~32.4 GB** | |

The text encoder is left at bf16 on purpose: mflux marks it
`skip_quantization=True` because 6-bit noticeably degrades prompt
understanding. That is the right trade, but it means a 32 GB working set on a
24 GB machine.

The text encoder only runs once per edit, at the start; the transformer runs on
every denoise step. So `memory.mode: low` evicts the encoder the moment the
prompt is encoded and lazily reloads it on the next turn:

| Mode | Denoise peak | Trade-off |
|---|---|---|
| `low` | **~16.9 GB** | Reloads the encoder (~10–25 s) on turns that re-encode. Best for 24 GB. |
| `balanced` | ~32 GB | No reload; relies on macOS paging the idle encoder out. Best for 32 GB+. |
| `off` | ~32 GB | No management at all. |

> mflux ships a `MemorySaver` that performs the same eviction but never
> reloads, so a second edit with a new prompt would fail. This app owns that
> lifecycle itself (`src/model_loader.py`).

---

## Configuration

Everything lives in `config.yaml`; there are no hardcoded paths. Relative paths
resolve against the project directory, and `~` / `${ENV_VAR}` are expanded.

```yaml
model:
  repo_id: "OsaurusAI/Qwen-Image-Edit-mflux-q6"   # or a local directory
  cache_dir: "./models/hf"

memory:
  mode: "low"                 # low | balanced | off

generation:
  steps: 25
  guidance: 4.0

ui:
  server_port: 7860
```

Use a different file with `python app.py --config other.yaml` or
`QWEN_EDIT_CONFIG=other.yaml`.

### Model choice

`OsaurusAI/Qwen-Image-Edit-mflux-q6` is the default because it is a complete
6-bit mflux export of `Qwen/Qwen-Image-Edit-2509`.

> The other 6-bit export, `Norton0924/Qwen-Image-Edit-2509-6bit`, is **broken
> upstream**: its `text_encoder/2.safetensors` is published at 67 MB while its
> own header declares 2.06 GB of tensor data. It cannot load. Run
> `python app.py --verify` after switching `repo_id` to catch this class of
> problem before it turns into an opaque MLX error.

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

Set `memory.mode: low`, drop to 512 or 768, and lower `steps`. Quit other large
apps: 24 GB is genuinely tight for a 32 GB working set. Watch the real number
with `python app.py` and the memory line in `/api/status`.

**"invalid data offsets" / "incomplete download"**

A weight shard is truncated. Diagnose precisely with:

```bash
python app.py --verify
```

Delete the reported snapshot under `models/hf/hub/` and re-run to re-download.
If it fails again on the same file, the upstream repo itself is broken — switch
`model.repo_id`.

**First edit takes forever**

The first edit downloads ~32 GB and then loads it. Later launches load from
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

* Model: [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) (Apache 2.0), 6-bit MLX export by [OsaurusAI](https://huggingface.co/OsaurusAI/Qwen-Image-Edit-mflux-q6)
* Inference: [mflux](https://github.com/filipstrand/mflux) — MLX implementations of generative image models
* UI: [Gradio](https://gradio.app)
