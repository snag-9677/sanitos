"""Gradio interface: chat-style iterative image editing.

The interaction model, and why it is shaped this way:

* The edit history is a **tree**, but the thread renders only the root-to-head
  path so it reads like an ordinary conversation. Off-path branches appear as a
  small gallery instead of a graph.
* The parent of an edit is **captured at submit time**. If the user reverts to
  an older step while the GPU is busy, the running edit still attaches where it
  was launched from rather than grafting onto the new head.
* Generation is **single-flight** and says so: Send becomes Stop. Selecting
  steps, comparing, and browsing branches stay responsive on other event lanes
  while an edit runs.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Iterator

import gradio as gr
from PIL import Image

from .config import AppConfig
from .device import DeviceInfo
from .inference import (
    EditProgress,
    EditRequest,
    ImageEditor,
    InferenceError,
    memory_snapshot,
)
from .families import get_family
from .model_loader import EditModel, is_model_cached
from .session import EditSession
from .utils import (
    ImageError,
    coerce_seed,
    dimensions_for_megapixels,
    format_duration,
    load_image,
    random_seed,
)

logger = logging.getLogger(__name__)

COMPARE_CHOICES = ["vs previous", "vs original", "vs pinned"]
_COMPARE_MODES = {"vs previous": "previous", "vs original": "original", "vs pinned": "pinned"}

CSS = """
.qe-thread { min-height: 420px; }
.qe-status { font-size: 0.86rem; opacity: 0.78; min-height: 1.4em; }
.qe-hint   { font-size: 0.82rem; opacity: 0.65; }
footer { display: none !important; }
"""


def launch_style() -> dict[str, Any]:
    """Styling for ``Blocks.launch``.

    Gradio 6 moved ``css`` and ``theme`` off the ``Blocks`` constructor, so the
    entrypoint passes these through at launch time.
    """
    return {"css": CSS, "theme": gr.themes.Soft()}


class EditorUI:
    """Builds the Gradio app and owns the per-tab state wiring."""

    def __init__(self, config: AppConfig, device: DeviceInfo, model: EditModel) -> None:
        self.config = config
        self.device = device
        self.model = model
        self.editor = ImageEditor(model)
        # One cancel flag per running job; replaced at the start of each edit.
        self._cancel = threading.Event()

    # ------------------------------------------------------------- rendering

    def _thread_messages(self, session: EditSession | None) -> tuple[list[dict], list[str]]:
        """Render the root-to-head path as chat messages.

        Returns the messages plus a parallel list mapping message index to step
        id, which is how ``.select`` resolves clicks back to nodes.
        """
        if session is None:
            return [], []

        messages: list[dict] = []
        id_map: list[str] = []

        for step in session.active_path():
            if step.is_root:
                messages.append(
                    {"role": "assistant", "content": gr.Image(step.image, label="Original")}
                )
                id_map.append(step.id)
                continue

            messages.append({"role": "user", "content": step.instruction})
            id_map.append(step.id)

            if step.error:
                messages.append({"role": "assistant", "content": f"⚠️ {step.error}"})
            else:
                messages.append(
                    {"role": "assistant", "content": gr.Image(step.image, label=None)}
                )
            id_map.append(step.id)

            caption = step.settings_caption()
            if step.duration:
                caption += f" · {format_duration(step.duration)}"
            messages.append({"role": "assistant", "content": f"<sub>{caption}</sub>"})
            id_map.append(step.id)

        return messages, id_map

    def _compare(self, session: EditSession | None, label: str) -> tuple[Image.Image, Image.Image] | None:
        if session is None:
            return None
        return session.compare_pair(_COMPARE_MODES.get(label, "previous"))

    def _gallery(self, session: EditSession | None) -> list[tuple[Image.Image, str]]:
        return session.branch_gallery() if session else []

    def _status(self, session: EditSession | None, extra: str = "") -> str:
        if session is None:
            return extra or "Upload an image to begin."
        head = session.head
        position = f"Step {session.depth_of(head.id)} of {session.edit_count} edit(s)"
        branches = len(session.other_branch_leaves())
        if branches:
            position += f" · {branches} other branch{'es' if branches > 1 else ''}"
        pinned = session.pinned
        if pinned:
            position += f" · pinned: step {session.depth_of(pinned.id)}"
        if extra:
            position += f" · {extra}"
        # Memory is the binding constraint on 24 GB, so keep it visible.
        if self.model.is_loaded:
            position += f"  ·  {memory_snapshot()}"
        return position

    def _refresh(
        self, session: EditSession | None, compare_label: str, extra: str = ""
    ) -> tuple[Any, ...]:
        """Standard bundle of outputs after any state change."""
        messages, id_map = self._thread_messages(session)
        return (
            messages,
            id_map,
            self._compare(session, compare_label),
            self._gallery(session),
            self._status(session, extra),
        )

    # -------------------------------------------------------------- handlers

    def on_upload(
        self, image: Any, session: EditSession | None, compare_label: str
    ) -> tuple[Any, ...]:
        """Start (or restart) a session from an uploaded image."""
        if image is None:
            return (*self._refresh(None, compare_label), None)
        try:
            pil = load_image(image)
        except ImageError as exc:
            gr.Warning(str(exc))
            return (*self._refresh(session, compare_label), session)

        new_session = EditSession(pil)
        note = f"{pil.width}x{pil.height} loaded — describe the edit you want."
        return (*self._refresh(new_session, compare_label, note), new_session)

    def on_select(self, event: gr.SelectData, id_map: list[str]) -> str | None:
        """Map a clicked message back to its step id."""
        try:
            index = event.index if isinstance(event.index, int) else event.index[0]
            return id_map[index]
        except (IndexError, TypeError, ValueError):
            return None

    def on_revert(
        self, selected: str | None, session: EditSession | None, compare_label: str
    ) -> tuple[Any, ...]:
        if session is None or not selected or selected not in session:
            gr.Info("Select a step in the thread first.")
            return self._refresh(session, compare_label)
        session.set_head(selected)
        note = "Reverted — editing from here creates a new branch."
        return self._refresh(session, compare_label, note)

    def on_undo(self, session: EditSession | None, compare_label: str) -> tuple[Any, ...]:
        if session is None:
            return self._refresh(session, compare_label)
        session.undo()
        return self._refresh(session, compare_label, "Undone.")

    def on_pin(
        self, selected: str | None, session: EditSession | None, compare_label: str
    ) -> tuple[Any, ...]:
        if session is None:
            return self._refresh(session, compare_label)
        target = selected if selected and selected in session else session.head_id
        pinned = session.toggle_pin(target)
        note = "Pinned for comparison." if pinned else "Unpinned."
        return self._refresh(session, compare_label, note)

    def on_reuse_prompt(self, selected: str | None, session: EditSession | None) -> str:
        if session is None or not selected or selected not in session:
            return gr.update()
        return session.get(selected).instruction

    def on_branch_select(
        self, event: gr.SelectData, session: EditSession | None, compare_label: str
    ) -> tuple[Any, ...]:
        """Jump to an abandoned branch from the gallery."""
        if session is None:
            return self._refresh(session, compare_label)
        leaves = session.other_branch_leaves()
        try:
            session.set_head(leaves[event.index].id)
        except (IndexError, TypeError):
            return self._refresh(session, compare_label)
        return self._refresh(session, compare_label, "Switched branch.")

    def on_compare_change(
        self, session: EditSession | None, compare_label: str
    ) -> tuple[Any, ...]:
        return self._compare(session, compare_label), self._status(session)

    def on_download(
        self, selected: str | None, session: EditSession | None
    ) -> Any:
        if session is None:
            gr.Info("Nothing to download yet.")
            return None
        step_id = selected if selected and selected in session else session.head_id
        try:
            path = session.export_step(step_id, self.config.output.dir, self.config.output.format)
        except (ImageError, OSError) as exc:
            gr.Warning(f"Could not save: {exc}")
            return None
        gr.Info(f"Saved to {path.name}")
        return str(path)

    def on_stop(self) -> str:
        self._cancel.set()
        return "Stopping after the current step…"

    def on_model_change(self, label: str) -> tuple[Any, Any, Any, Any]:
        """Switch models from the picker.

        Returns (status, steps, guidance, negative-prompt visibility) — the
        generation defaults differ per family, so they move with the model
        rather than silently staying wrong.
        """
        entry = self._entry_for_label(label)
        if entry is None:
            return gr.update(), gr.update(), gr.update(), gr.update()

        if self.editor.is_busy:
            gr.Warning("An edit is running. Wait for it to finish before switching models.")
            return gr.update(), gr.update(), gr.update(), gr.update()

        family = get_family(entry.family)
        mode = self.config.memory.resolve(
            family.working_set_gb, self.device.usable_memory_gb
        )

        try:
            self.model.switch_to(
                entry.repo_id, family, label=entry.label, memory_mode=mode
            )
        except Exception as exc:  # noqa: BLE001 - keep the UI alive
            gr.Warning(f"Could not switch model: {exc}")
            return gr.update(), gr.update(), gr.update(), gr.update()

        self.config.model.active = entry.id
        cached = is_model_cached(entry.repo_id, self.config.model.cache_dir, family)
        note = (
            f"{entry.label} selected · {family.working_set_gb:.0f} GB · memory mode {mode}"
        )
        if not cached:
            note += f" · downloads ~{family.working_set_gb:.0f} GB on the next edit"
        gr.Info(note)

        return (
            note,
            gr.update(value=family.default_steps),
            gr.update(
                value=family.default_guidance,
                minimum=family.guidance_range[0],
                maximum=family.guidance_range[1],
            ),
            gr.update(visible=family.supports_negative_prompt),
        )

    def _entry_for_label(self, label: str):
        for entry in self.config.model.catalog:
            if entry.describe(self.device.usable_memory_gb) == label:
                return entry
        return None

    def on_random_seed(self) -> int:
        return random_seed()

    def on_preset(self, preset_label: str) -> str:
        for preset in self.config.ui.style_presets:
            if preset.label == preset_label:
                return preset.prompt
        return gr.update()

    def on_resolution_change(self, label: str) -> tuple[Any, Any]:
        """Fixed presets fill the width/height boxes; auto presets clear them."""
        for preset in self.config.generation.resolution_presets:
            if preset.label == label:
                return preset.width, preset.height
        return gr.update(), gr.update()

    def _resolve_size(
        self, label: str, width: int | None, height: int | None, image: Image.Image
    ) -> tuple[int | None, int | None]:
        """Decide the output size for one edit.

        Explicit width/height in the boxes always wins. Otherwise an auto
        preset sizes to its megapixel budget at the source aspect ratio, so a
        landscape photo stays landscape.
        """
        if width and height:
            return int(width), int(height)

        for preset in self.config.generation.resolution_presets:
            if preset.label != label:
                continue
            if preset.width and preset.height:
                return preset.width, preset.height
            if preset.megapixels:
                return dimensions_for_megapixels(image, preset.megapixels)
            break

        return None, None  # let the model derive it from the input

    # ------------------------------------------------------------ generation

    def on_submit(
        self,
        instruction: str,
        session: EditSession | None,
        compare_label: str,
        steps: int,
        guidance: float,
        seed_value: Any,
        randomise: bool,
        width: int | None,
        height: int | None,
        negative_prompt: str,
        resolution_label: str = "",
    ) -> Iterator[tuple[Any, ...]]:
        """Run one edit, streaming progress into the thread.

        Yields ``(messages, id_map, compare, gallery, status, prompt_box,
        send_btn, stop_btn, seed_box)``.
        """
        send_idle = gr.update(interactive=True, value="Edit  ⌘↵")
        send_busy = gr.update(interactive=False, value="Editing…")
        stop_on = gr.update(visible=True, interactive=True)
        stop_off = gr.update(visible=False)

        if session is None:
            gr.Warning("Upload an image first.")
            yield (*self._refresh(session, compare_label), instruction, send_idle, stop_off, seed_value)
            return

        if not instruction.strip():
            gr.Warning("Describe the edit you want.")
            yield (*self._refresh(session, compare_label), instruction, send_idle, stop_off, seed_value)
            return

        # Bind the parent NOW. The user may revert the head while this runs;
        # the result must still attach to the image they launched from.
        parent_id = session.head_id
        parent_image = session.get(parent_id).image

        seed = random_seed() if randomise else coerce_seed(seed_value)
        out_width, out_height = self._resolve_size(
            resolution_label, width, height, parent_image
        )

        try:
            request = EditRequest(
                images=[parent_image],
                instruction=instruction.strip(),
                seed=seed,
                steps=int(steps),
                guidance=float(guidance),
                width=out_width,
                height=out_height,
                negative_prompt=negative_prompt or "",
                scheduler=self.config.generation.scheduler,
            )
        except InferenceError as exc:
            gr.Warning(str(exc))
            yield (*self._refresh(session, compare_label), instruction, send_idle, stop_off, seed_value)
            return

        self._cancel = threading.Event()
        cancel_event = self._cancel

        # Progress arrives from the generation thread; the generator below
        # drains this slot to stream updates without blocking the GPU.
        latest: dict[str, EditProgress | None] = {"progress": None}
        outcome: dict[str, Any] = {}

        def on_progress(progress: EditProgress) -> None:
            latest["progress"] = progress

        def run() -> None:
            try:
                outcome["result"] = self.editor.edit(
                    request, on_progress=on_progress, cancel_event=cancel_event
                )
            except InferenceError as exc:
                outcome["error"] = str(exc)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected failure during edit")
                outcome["error"] = f"Unexpected error: {exc}"

        worker = threading.Thread(target=run, name="qwen-edit", daemon=True)
        worker.start()

        base_messages, base_ids = self._thread_messages(session)
        base_messages.append({"role": "user", "content": instruction.strip()})
        base_ids.append(parent_id)

        def pending_view(text: str, preview: Image.Image | None) -> tuple[list[dict], list[str]]:
            messages = list(base_messages)
            ids = list(base_ids)
            if preview is not None:
                messages.append(
                    {"role": "assistant", "content": gr.Image(preview, label="preview")}
                )
                ids.append(parent_id)
            messages.append({"role": "assistant", "content": f"⏳ {text}"})
            ids.append(parent_id)
            return messages, ids

        gallery = self._gallery(session)
        compare = self._compare(session, compare_label)
        last_preview: Image.Image | None = None
        pending_text = "Preparing…"

        yield (
            *pending_view(pending_text, last_preview),
            compare,
            gallery,
            self._status(session, "Generating…"),
            "",  # clear the composer so the next instruction can be drafted
            send_busy,
            stop_on,
            seed,
        )

        while worker.is_alive():
            worker.join(timeout=0.4)
            progress = latest["progress"]
            if progress is None:
                continue
            if progress.preview is not None:
                last_preview = progress.preview
            text = progress.describe()
            if text == pending_text and progress.preview is None:
                continue
            pending_text = text
            yield (
                *pending_view(pending_text, last_preview),
                compare,
                gallery,
                self._status(session, "Generating…"),
                gr.update(),
                send_busy,
                stop_on,
                seed,
            )

        worker.join()

        if "error" in outcome:
            gr.Warning(outcome["error"])
            messages, ids = self._thread_messages(session)
            messages.append({"role": "user", "content": instruction.strip()})
            ids.append(parent_id)
            messages.append({"role": "assistant", "content": f"⚠️ {outcome['error']}"})
            ids.append(parent_id)
            yield (
                messages,
                ids,
                compare,
                gallery,
                self._status(session, "Failed."),
                instruction,
                send_idle,
                stop_off,
                seed,
            )
            return

        result = outcome["result"]
        # advance_head only if the user hasn't navigated away mid-run.
        follow = session.head_id == parent_id
        session.add_step(
            result.image,
            result.instruction,
            parent_id=parent_id,
            advance_head=follow,
            seed=result.seed,
            steps=result.steps,
            guidance=result.guidance,
            width=result.width,
            height=result.height,
            negative_prompt=result.negative_prompt,
            duration=result.duration,
            interrupted=result.interrupted,
        )

        if self.config.output.save_sessions:
            try:
                session.save(self.config.output.sessions_dir, self.config.output.format)
            except OSError as exc:
                logger.warning("Could not persist session: %s", exc)

        note = (
            f"Stopped at step {result.completed_steps}/{result.steps}"
            if result.interrupted
            else f"Done in {format_duration(result.duration)}"
        )
        if not follow:
            note += " — landed on a branch you navigated away from"

        yield (
            *self._refresh(session, compare_label, note),
            "",
            send_idle,
            stop_off,
            seed,
        )

    # ----------------------------------------------------------------- build

    def build(self) -> gr.Blocks:
        cfg = self.config
        gen = cfg.generation
        rec = gen.m5_recommended
        entry = cfg.model.active_entry
        family = get_family(entry.family)
        cached = is_model_cached(entry.repo_id, cfg.model.cache_dir, family)

        with gr.Blocks(title=cfg.ui.title) as demo:
            session_state = gr.State(None)
            id_map_state = gr.State([])
            selected_state = gr.State(None)

            gr.Markdown(
                f"## {cfg.ui.title}\n"
                f"<span class='qe-hint'>{self.device.chip} · "
                f"{self.device.backend_label} · {self.model.label} · "
                f"memory mode: {cfg.memory.mode}</span>",
            )
            if not cached:
                gr.Markdown(
                    "<span class='qe-hint'>⬇️ First edit downloads ~32 GB of weights "
                    "into <code>models/</code>. Later runs load from cache.</span>"
                )

            with gr.Row():
                # ---------------------------------------------- left column
                with gr.Column(scale=4):
                    source = gr.Image(
                        label="Source image",
                        type="pil",
                        height=240,
                        sources=["upload", "clipboard"],
                    )

                    model_choices = [
                        e.describe(self.device.usable_memory_gb)
                        for e in cfg.model.catalog
                    ]
                    model_picker = gr.Dropdown(
                        choices=model_choices,
                        value=cfg.model.active_entry.describe(
                            self.device.usable_memory_gb
                        ),
                        label="Model",
                        info="Switching unloads the current model. Weights "
                             "download on first use.",
                    )

                    with gr.Accordion("Settings", open=False):
                        resolution = gr.Dropdown(
                            choices=[p.label for p in gen.resolution_presets],
                            value=rec.get("preset") or gen.resolution_presets[0].label,
                            label="Resolution",
                        )
                        with gr.Row():
                            width = gr.Number(label="Width", value=None, precision=0)
                            height = gr.Number(label="Height", value=None, precision=0)
                        steps = gr.Slider(
                            4, 60, value=int(rec.get("steps", family.default_steps)), step=1,
                            label="Steps",
                            info="More steps = more detail, linearly more time.",
                        )
                        guidance = gr.Slider(
                            family.guidance_range[0], family.guidance_range[1],
                            value=float(rec.get("guidance", family.default_guidance)),
                            step=0.1, label="Guidance",
                            info="How strictly to follow the instruction. "
                                 "FLUX.2 wants ~1.0; Qwen ~4.0.",
                        )
                        with gr.Row():
                            seed_box = gr.Number(
                                label="Seed", value=random_seed(), precision=0, scale=3
                            )
                            random_btn = gr.Button("🎲", scale=1, min_width=48)
                        randomise = gr.Checkbox(
                            label="New random seed each edit", value=True
                        )
                        negative = gr.Textbox(
                            label="Negative prompt",
                            value=gen.negative_prompt,
                            placeholder="blurry, low quality, distorted…",
                            lines=2,
                            # FLUX.2 rejects a negative prompt outright, so the
                            # field is hidden rather than silently ignored.
                            visible=family.supports_negative_prompt,
                        )
                        gr.Markdown(
                            f"<span class='qe-hint'>Recommended for "
                            f"{self.device.chip}: {rec.get('preset', '768x768')}, "
                            f"{rec.get('steps', 25)} steps, guidance "
                            f"{rec.get('guidance', 4.0)}.</span>"
                        )

                    gr.Markdown("<span class='qe-hint'>Other branches</span>")
                    branches = gr.Gallery(
                        label=None, columns=3, height=140, object_fit="cover",
                        show_label=False, preview=False,
                    )

                # --------------------------------------------- centre column
                with gr.Column(scale=7):
                    thread = gr.Chatbot(
                        label=None,
                        height=520,
                        elem_classes=["qe-thread"],
                        show_label=False,
                        avatar_images=None,
                        group_consecutive_messages=False,
                        placeholder=(
                            "<div style='text-align:center;opacity:.6'>"
                            "<h3>Edit images by describing the change</h3>"
                            "<p>Upload an image, then say what to change.<br>"
                            "Each edit builds on the last one.</p></div>"
                        ),
                    )

                    with gr.Row():
                        prompt = gr.Textbox(
                            placeholder="Replace the background with a futuristic city…",
                            label=None, show_label=False, lines=2, scale=8,
                            autofocus=True,
                        )
                        with gr.Column(scale=2, min_width=120):
                            send = gr.Button("Edit  ⌘↵", variant="primary")
                            stop = gr.Button("Stop", variant="stop", visible=False)

                    with gr.Row():
                        for preset in cfg.ui.style_presets:
                            gr.Button(preset.label, size="sm").click(
                                fn=lambda label=preset.label: self.on_preset(label),
                                outputs=prompt,
                            )

                    status = gr.Markdown(
                        "Upload an image to begin.", elem_classes=["qe-status"]
                    )

                    with gr.Row():
                        revert_btn = gr.Button("↩ Revert here", size="sm")
                        pin_btn = gr.Button("📌 Pin", size="sm")
                        reuse_btn = gr.Button("↻ Reuse prompt", size="sm")
                        undo_btn = gr.Button("⌫ Undo", size="sm")
                        download_btn = gr.Button("⬇ Save", size="sm")

                # ---------------------------------------------- right column
                with gr.Column(scale=5):
                    compare_mode = gr.Radio(
                        COMPARE_CHOICES, value=COMPARE_CHOICES[0],
                        label="Compare", show_label=True,
                    )
                    comparison = gr.ImageSlider(
                        label=None, show_label=False, height=420, interactive=False,
                    )
                    download_file = gr.File(label="Saved file", visible=True)
                    gr.Markdown(
                        "<span class='qe-hint'>Click any message in the thread to "
                        "select that step, then use the buttons under the composer. "
                        "Editing from an older step branches instead of overwriting.</span>"
                    )

            # ------------------------------------------------------- wiring
            refresh_outputs = [thread, id_map_state, comparison, branches, status]

            source.change(
                self.on_upload,
                inputs=[source, session_state, compare_mode],
                outputs=[*refresh_outputs, session_state],
            )

            resolution.change(
                self.on_resolution_change, inputs=resolution, outputs=[width, height]
            )
            model_picker.change(
                self.on_model_change,
                inputs=model_picker,
                outputs=[status, steps, guidance, negative],
            )
            random_btn.click(self.on_random_seed, outputs=seed_box)

            thread.select(self.on_select, inputs=id_map_state, outputs=selected_state)

            submit_inputs = [
                prompt, session_state, compare_mode, steps, guidance,
                seed_box, randomise, width, height, negative, resolution,
            ]
            submit_outputs = [
                thread, id_map_state, comparison, branches, status,
                prompt, send, stop, seed_box,
            ]

            # concurrency_id pins generation to a single lane so the GPU is
            # never asked to run two edits at once, while selection, compare,
            # and gallery events stay responsive on their own lanes.
            send.click(
                self.on_submit,
                inputs=submit_inputs,
                outputs=submit_outputs,
                concurrency_id="gpu",
                concurrency_limit=1,
            )
            prompt.submit(
                self.on_submit,
                inputs=submit_inputs,
                outputs=submit_outputs,
                concurrency_id="gpu",
                concurrency_limit=1,
            )
            stop.click(self.on_stop, outputs=status)

            revert_btn.click(
                self.on_revert,
                inputs=[selected_state, session_state, compare_mode],
                outputs=refresh_outputs,
            )
            undo_btn.click(
                self.on_undo,
                inputs=[session_state, compare_mode],
                outputs=refresh_outputs,
            )
            pin_btn.click(
                self.on_pin,
                inputs=[selected_state, session_state, compare_mode],
                outputs=refresh_outputs,
            )
            reuse_btn.click(
                self.on_reuse_prompt,
                inputs=[selected_state, session_state],
                outputs=prompt,
            )
            download_btn.click(
                self.on_download,
                inputs=[selected_state, session_state],
                outputs=download_file,
            )
            branches.select(
                self.on_branch_select,
                inputs=[session_state, compare_mode],
                outputs=refresh_outputs,
            )
            compare_mode.change(
                self.on_compare_change,
                inputs=[session_state, compare_mode],
                outputs=[comparison, status],
            )

        return demo


def build_ui(config: AppConfig, device: DeviceInfo, model: EditModel) -> gr.Blocks:
    return EditorUI(config, device, model).build()
