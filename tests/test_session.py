"""Tests for the edit-tree session state.

These need no model and no GPU, so they run in under a second:

    ./venv/bin/python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.session import EditSession, SessionError  # noqa: E402
from src.utils import coerce_seed, format_duration, snap_dimension  # noqa: E402


def img(colour: str, size: int = 64) -> Image.Image:
    return Image.new("RGB", (size, size), colour)


@pytest.fixture()
def session() -> EditSession:
    return EditSession(img("red"))


def test_root_is_the_starting_head(session: EditSession) -> None:
    assert session.head.id == session.root_id
    assert session.head.is_root
    assert session.edit_count == 0
    assert session.compare_pair("previous") is None


def test_linear_edits_form_a_chat_thread(session: EditSession) -> None:
    session.add_step(img("green"), "make it green", seed=1)
    session.add_step(img("blue"), "now blue", seed=2)

    assert [s.instruction for s in session.active_path()] == ["", "make it green", "now blue"]
    assert session.edit_count == 2
    assert session.depth_of(session.head_id) == 2


def test_branching_preserves_the_abandoned_work(session: EditSession) -> None:
    green = session.add_step(img("green"), "green", seed=1)
    blue = session.add_step(img("blue"), "blue", seed=2)

    session.set_head(green.id)
    yellow = session.add_step(img("yellow"), "yellow instead", seed=3)

    assert session.head.id == yellow.id
    assert blue.id in session, "reverting must not destroy the old branch"
    assert [s.instruction for s in session.active_path()] == ["", "green", "yellow instead"]
    assert [leaf.id for leaf in session.other_branch_leaves()] == [blue.id]
    assert len(session.branch_gallery()) == 1


def test_redo_picks_the_newest_child_not_a_clock_tie(session: EditSession) -> None:
    """Two children created in the same second must still order deterministically."""
    green = session.add_step(img("green"), "green")
    session.add_step(img("blue"), "blue")          # first child of green
    session.set_head(green.id)
    yellow = session.add_step(img("yellow"), "yellow")  # second child of green

    session.set_head(green.id)
    assert session.redo().id == yellow.id


def test_late_result_attaches_to_its_submit_time_parent(session: EditSession) -> None:
    """A slow edit must land on the node that was head when it was submitted.

    Guards the mid-generation revert: the user reverts while the GPU is busy,
    and the result must not graft itself onto whatever head became.
    """
    green = session.add_step(img("green"), "green")
    captured_parent = green.id

    session.set_head(session.root_id)  # user reverts mid-run
    late = session.add_step(
        img("pink"), "late result", parent_id=captured_parent, advance_head=False
    )

    assert late.parent_id == captured_parent
    assert session.head.id == session.root_id, "a late result must not steal the head"


def test_undo_stops_at_the_root(session: EditSession) -> None:
    green = session.add_step(img("green"), "green")
    assert session.undo().id == session.root_id
    assert session.undo().id == session.root_id  # no-op, not an error
    assert green.id in session


def test_pin_toggles_and_drives_comparison(session: EditSession) -> None:
    green = session.add_step(img("green"), "green")
    blue = session.add_step(img("blue"), "blue")

    assert session.toggle_pin(green.id).id == green.id
    assert session.compare_pair("pinned")[0] is green.image
    assert session.toggle_pin(green.id) is None
    assert session.compare_pair("previous")[0] is green.image
    assert session.compare_pair("original")[0] is session.root.image
    assert session.compare_pair("previous")[1] is blue.image


def test_unknown_ids_raise(session: EditSession) -> None:
    with pytest.raises(SessionError):
        session.set_head("nope")
    with pytest.raises(SessionError):
        session.get("nope")
    with pytest.raises(SessionError):
        session.add_step(img("green"), "x", parent_id="nope")


def test_save_and_load_round_trip(session: EditSession, tmp_path: Path) -> None:
    green = session.add_step(img("green"), "green", seed=7, steps=25, guidance=4.0)
    session.add_step(img("blue"), "blue", seed=8)
    session.set_head(green.id)
    yellow = session.add_step(img("yellow"), "yellow", seed=9)
    session.toggle_pin(green.id)

    manifest = session.save(tmp_path)
    restored = EditSession.load(manifest.parent)

    assert restored.id == session.id
    assert len(restored) == len(session)
    assert restored.head_id == yellow.id
    assert restored.pinned.id == green.id
    assert [s.instruction for s in restored.active_path()] == ["", "green", "yellow"]
    assert restored.get(green.id).seed == 7
    assert restored.get(green.id).steps == 25

    # New edits on the restored session must still sort after the old ones.
    fresh = restored.add_step(img("purple"), "purple")
    assert fresh.sequence > max(s.sequence for s in session.iter_steps())


def test_reset_starts_over(session: EditSession) -> None:
    session.add_step(img("green"), "green")
    session.reset(img("white"))
    assert len(session) == 1
    assert session.head.is_root
    assert session.edit_count == 0


# --------------------------------------------------------------- utils tests


def test_snap_dimension_respects_the_latent_grid() -> None:
    assert snap_dimension(1000) == 992      # 16-aligned
    assert snap_dimension(1024) == 1024
    assert snap_dimension(100) == 256       # clamped to the floor
    assert snap_dimension(None) is None


def test_coerce_seed_handles_ui_input() -> None:
    assert coerce_seed(42) == 42
    assert coerce_seed("42") == 42
    assert 0 <= coerce_seed("") <= 2**32 - 1
    assert 0 <= coerce_seed(None) <= 2**32 - 1
    assert 0 <= coerce_seed("banana") <= 2**32 - 1
    assert 0 <= coerce_seed(-5) <= 2**32 - 1


def test_format_duration_reads_naturally() -> None:
    assert format_duration(0.25) == "250 ms"
    assert format_duration(45.2) == "45.2s"
    assert format_duration(92) == "1m 32s"
