"""UI-layer tests that need no GPU and no model weights.

These exercise the parts most likely to break silently: the chat message
format Gradio actually accepts, the index->step-id mapping that makes message
selection work, and the submit generator's contract with its output list.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import gradio as gr
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.inference import EditProgress, EditResult  # noqa: E402
from src.session import EditSession  # noqa: E402
from src.ui import EditorUI  # noqa: E402


def img(colour: str = "red", size: tuple[int, int] = (64, 64)) -> Image.Image:
    return Image.new("RGB", size, colour)


class StubEditor:
    """Stands in for ImageEditor: emits a few progress ticks, then a result."""

    def __init__(self, *, fail: str | None = None, delay: float = 0.0) -> None:
        self.fail = fail
        self.delay = delay
        self.calls: list = []
        self.is_busy = False

    def edit(self, request, *, on_progress=None, cancel_event=None):
        from src.inference import InferenceError

        self.calls.append(request)
        if on_progress:
            on_progress(EditProgress(1, request.steps, 1.0, eta="~2s"))
            on_progress(EditProgress(2, request.steps, 2.0, preview=img("yellow")))
        if self.fail:
            raise InferenceError(self.fail)
        return EditResult(
            image=img("blue"),
            seed=request.seed,
            steps=request.steps,
            guidance=request.guidance,
            width=64,
            height=64,
            duration=1.5,
            instruction=request.instruction,
        )


@pytest.fixture()
def ui() -> EditorUI:
    from src.device import detect_device
    from src.model_loader import QwenEditModel

    cfg = load_config()
    model = QwenEditModel(repo_id=cfg.model.repo_id, cache_dir=cfg.model.cache_dir)
    instance = EditorUI(cfg, detect_device(), model)
    instance.editor = StubEditor()
    return instance


def drain(generator) -> list[tuple]:
    return list(generator)


# ------------------------------------------------------- message format


def test_messages_are_accepted_by_gradio_chatbot(ui: EditorUI) -> None:
    """The embedded gr.Image content must survive Chatbot.postprocess.

    Gradio 6 changed the Chatbot API (``type`` was removed, messages became the
    only format). If the message shape regresses, the thread renders empty
    rather than raising, so assert the component accepts it.
    """
    session = EditSession(img())
    session.add_step(img("green"), "make it green", seed=1, steps=25, guidance=4.0, duration=42.0)

    messages, id_map = ui._thread_messages(session)
    assert len(messages) == len(id_map), "every message needs a step id for .select"

    chatbot = gr.Chatbot()
    payload = chatbot.postprocess(messages)
    assert payload is not None
    assert len(payload.root) == len(messages)


def test_id_map_lets_selection_resolve_to_a_step(ui: EditorUI) -> None:
    session = EditSession(img())
    first = session.add_step(img("green"), "green")
    second = session.add_step(img("blue"), "blue")

    messages, id_map = ui._thread_messages(session)
    assert id_map[0] == session.root_id
    assert second.id in id_map and first.id in id_map

    class Evt:
        index = len(id_map) - 1

    assert ui.on_select(Evt(), id_map) == second.id
    # Out-of-range clicks must not raise.
    class Bad:
        index = 999

    assert ui.on_select(Bad(), id_map) is None


def test_root_only_session_renders_just_the_original(ui: EditorUI) -> None:
    messages, id_map = ui._thread_messages(EditSession(img()))
    assert len(messages) == 1 and len(id_map) == 1


# ------------------------------------------------------------- handlers


def test_upload_starts_a_session(ui: EditorUI) -> None:
    *_, session = ui.on_upload(img("red", (128, 96)), None, "vs previous")
    assert isinstance(session, EditSession)
    assert session.head.is_root
    assert session.root.image.size == (128, 96)


def test_upload_of_nothing_clears_state(ui: EditorUI) -> None:
    *_, session = ui.on_upload(None, None, "vs previous")
    assert session is None


def test_submit_without_image_warns_and_yields_full_output(ui: EditorUI) -> None:
    frames = drain(
        ui.on_submit("do a thing", None, "vs previous", 25, 4.0, 1, False, None, None, "")
    )
    assert len(frames) == 1
    assert len(frames[0]) == 9, "output tuple must match submit_outputs length"


def test_submit_with_blank_instruction_is_rejected(ui: EditorUI) -> None:
    session = EditSession(img())
    frames = drain(
        ui.on_submit("   ", session, "vs previous", 25, 4.0, 1, False, None, None, "")
    )
    assert len(frames) == 1
    assert session.edit_count == 0


def test_successful_submit_appends_a_step(ui: EditorUI) -> None:
    session = EditSession(img())
    frames = drain(
        ui.on_submit("make it blue", session, "vs previous", 12, 4.0, 7, False, 64, 64, "")
    )

    assert len(frames) >= 2, "expect at least a pending frame and a final frame"
    assert all(len(f) == 9 for f in frames), "every yield must match submit_outputs"

    assert session.edit_count == 1
    step = session.head
    assert step.instruction == "make it blue"
    assert step.seed == 7
    assert step.duration == 1.5
    assert ui.editor.calls[0].steps == 12


def test_failed_submit_leaves_the_tree_untouched(ui: EditorUI) -> None:
    ui.editor = StubEditor(fail="model exploded")
    session = EditSession(img())
    frames = drain(
        ui.on_submit("make it blue", session, "vs previous", 12, 4.0, 7, False, None, None, "")
    )

    assert session.edit_count == 0, "a failed edit must not create a step"
    assert all(len(f) == 9 for f in frames)
    # The error is surfaced in the thread rather than silently dropped.
    assert any("exploded" in str(f[0]) for f in frames)


def test_result_attaches_to_its_submit_time_parent(ui: EditorUI) -> None:
    """The user reverts mid-run; the result must not follow the moved head."""
    session = EditSession(img())
    first = session.add_step(img("green"), "green")
    assert session.head_id == first.id

    generator = ui.on_submit(
        "second edit", session, "vs previous", 12, 4.0, 1, False, None, None, ""
    )
    next(generator)                    # start the job; parent is captured here
    session.set_head(session.root_id)  # user navigates away mid-generation
    drain(generator)

    new_step = next(s for s in session.iter_steps() if s.instruction == "second edit")
    assert new_step.parent_id == first.id, "result grafted onto the wrong parent"
    assert session.head_id == session.root_id, "a late result must not steal the head"


def test_randomise_seed_overrides_the_seed_box(ui: EditorUI) -> None:
    session = EditSession(img())
    drain(ui.on_submit("x", session, "vs previous", 12, 4.0, 99, True, None, None, ""))
    assert ui.editor.calls[0].seed != 99


def test_revert_and_pin_require_a_selection(ui: EditorUI) -> None:
    session = EditSession(img())
    step = session.add_step(img("green"), "green")
    session.add_step(img("blue"), "blue")

    ui.on_revert(step.id, session, "vs previous")
    assert session.head_id == step.id

    ui.on_pin(step.id, session, "vs previous")
    assert session.pinned.id == step.id
    ui.on_pin(step.id, session, "vs previous")
    assert session.pinned is None


def test_reuse_prompt_returns_the_instruction(ui: EditorUI) -> None:
    session = EditSession(img())
    step = session.add_step(img("green"), "make it green")
    assert ui.on_reuse_prompt(step.id, session) == "make it green"


def test_stop_sets_the_cancel_flag(ui: EditorUI) -> None:
    ui._cancel = threading.Event()
    ui.on_stop()
    assert ui._cancel.is_set()


def test_branch_gallery_selection_switches_head(ui: EditorUI) -> None:
    session = EditSession(img())
    green = session.add_step(img("green"), "green")
    blue = session.add_step(img("blue"), "blue")
    session.set_head(green.id)
    session.add_step(img("yellow"), "yellow")

    class Evt:
        index = 0

    ui.on_branch_select(Evt(), session, "vs previous")
    assert session.head_id == blue.id


def test_resolution_preset_updates_dimensions(ui: EditorUI) -> None:
    """Fixed presets fill the boxes; auto presets clear them for later sizing."""
    width, height = ui.on_resolution_change("Square — 768 x 768")
    assert (width, height) == (768, 768)
    width, height = ui.on_resolution_change("Balanced — match input, 0.6 MP")
    assert width is None and height is None


# ------------------------------------------------- aspect-preserving sizing


def test_auto_preset_preserves_aspect_ratio(ui: EditorUI) -> None:
    """A landscape photo must not come back squashed into a square."""
    landscape = img("red", (1600, 900))
    width, height = ui._resolve_size("Balanced — match input, 0.6 MP", None, None, landscape)

    assert width % 16 == 0 and height % 16 == 0
    assert width > height, "landscape input should stay landscape"
    assert abs((width / height) - (1600 / 900)) < 0.05
    assert 0.4 < (width * height) / 1_000_000 < 0.8


def test_auto_preset_handles_portrait(ui: EditorUI) -> None:
    width, height = ui._resolve_size(
        "Quality — match input, 1.0 MP", None, None, img("red", (900, 1600))
    )
    assert height > width
    assert 0.8 < (width * height) / 1_000_000 < 1.25


def test_explicit_dimensions_win_over_the_preset(ui: EditorUI) -> None:
    width, height = ui._resolve_size(
        "Balanced — match input, 0.6 MP", 512, 512, img("red", (1600, 900))
    )
    assert (width, height) == (512, 512)


def test_fixed_preset_reframes_deliberately(ui: EditorUI) -> None:
    width, height = ui._resolve_size("Square — 768 x 768", None, None, img("red", (1600, 900)))
    assert (width, height) == (768, 768)


def test_submit_passes_resolved_size_to_the_editor(ui: EditorUI) -> None:
    session = EditSession(img("red", (1600, 900)))
    drain(
        ui.on_submit(
            "x", session, "vs previous", 12, 4.0, 1, False, None, None, "",
            "Balanced — match input, 0.6 MP",
        )
    )
    request = ui.editor.calls[0]
    assert request.width and request.height
    assert request.width > request.height
