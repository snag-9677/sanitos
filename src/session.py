"""Edit-session state: a tree of edit steps with a movable head.

Why a tree and not a list. Every node costs 30-90 seconds of GPU time. With a
linear history, "go back to step 3 and try something else" would destroy steps
4-7 — several minutes of unrepeatable work. A tree makes reverting free: the
old branch stays alive and reachable.

The tree is deliberately *not* drawn as a graph. The UI renders the path from
the root to the current head, which reads as an ordinary chat thread; other
branches surface as a small gallery of leaf thumbnails.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

from .utils import ImageError, load_image, save_image, thumbnail

logger = logging.getLogger(__name__)


class SessionError(RuntimeError):
    """Raised for invalid operations on an edit session."""


@dataclass(slots=True)
class EditStep:
    """One node in the edit tree.

    The root step has ``parent_id is None`` and an empty instruction; it holds
    the image the user uploaded.
    """

    id: str
    parent_id: str | None
    image: Image.Image
    instruction: str = ""
    seed: int | None = None
    steps: int | None = None
    guidance: float | None = None
    width: int | None = None
    height: int | None = None
    negative_prompt: str = ""
    duration: float | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    # Monotonic insertion order within the session. Wall-clock timestamps tie
    # when two steps land in the same second, which makes "newest child" — and
    # therefore redo() and the branch gallery — pick arbitrarily.
    sequence: int = 0
    saved_path: Path | None = None
    interrupted: bool = False
    error: str | None = None

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    def settings_caption(self) -> str:
        """One-line summary of how this step was produced."""
        if self.is_root:
            return f"{self.image.width}x{self.image.height} · original"
        bits = [f"{self.image.width}x{self.image.height}"]
        if self.steps is not None:
            bits.append(f"{self.steps} steps")
        if self.guidance is not None:
            bits.append(f"guidance {self.guidance:g}")
        if self.seed is not None:
            bits.append(f"seed {self.seed}")
        if self.interrupted:
            bits.append("stopped early")
        return " · ".join(bits)

    def to_dict(self) -> dict[str, Any]:
        """Serialise everything except the image itself."""
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "instruction": self.instruction,
            "seed": self.seed,
            "steps": self.steps,
            "guidance": self.guidance,
            "width": self.width,
            "height": self.height,
            "negative_prompt": self.negative_prompt,
            "duration": self.duration,
            "created_at": self.created_at,
            "sequence": self.sequence,
            "saved_path": str(self.saved_path) if self.saved_path else None,
            "interrupted": self.interrupted,
            "error": self.error,
        }


class EditSession:
    """A tree of :class:`EditStep` nodes with a movable head pointer."""

    def __init__(self, root_image: Image.Image, session_id: str | None = None) -> None:
        self.id = session_id or uuid.uuid4().hex[:12]
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self._steps: dict[str, EditStep] = {}
        self._pinned_id: str | None = None
        self._counter = 0

        root = EditStep(id=uuid.uuid4().hex[:12], parent_id=None, image=root_image)
        self._steps[root.id] = root
        self.root_id = root.id
        self.head_id = root.id

    def _next_sequence(self) -> int:
        self._counter += 1
        return self._counter

    # ---------------------------------------------------------------- access

    def __len__(self) -> int:
        return len(self._steps)

    def __contains__(self, step_id: object) -> bool:
        return step_id in self._steps

    @property
    def root(self) -> EditStep:
        return self._steps[self.root_id]

    @property
    def head(self) -> EditStep:
        return self._steps[self.head_id]

    @property
    def edit_count(self) -> int:
        """Number of actual edits (every node except the root)."""
        return len(self._steps) - 1

    def get(self, step_id: str) -> EditStep:
        try:
            return self._steps[step_id]
        except KeyError as exc:
            raise SessionError(f"No such step: {step_id}") from exc

    def children_of(self, step_id: str) -> list[EditStep]:
        return [s for s in self._steps.values() if s.parent_id == step_id]

    @property
    def pinned(self) -> EditStep | None:
        return self._steps.get(self._pinned_id) if self._pinned_id else None

    # ------------------------------------------------------------- mutation

    def add_step(
        self,
        image: Image.Image,
        instruction: str,
        *,
        parent_id: str | None = None,
        advance_head: bool = True,
        **params: Any,
    ) -> EditStep:
        """Append an edit as a child of ``parent_id`` (default: current head).

        ``parent_id`` is passed explicitly by the generation job, which captures
        it at submit time. That keeps a long-running edit correctly attached
        even if the user reverts the head while it is still running.
        """
        parent = parent_id or self.head_id
        if parent not in self._steps:
            raise SessionError(f"Cannot attach step to unknown parent: {parent}")

        step = EditStep(
            id=uuid.uuid4().hex[:12],
            parent_id=parent,
            image=image,
            instruction=instruction,
            sequence=self._next_sequence(),
            **params,
        )
        self._steps[step.id] = step
        if advance_head:
            self.head_id = step.id
        return step

    def set_head(self, step_id: str) -> EditStep:
        """Move the head. Editing from here creates a branch."""
        if step_id not in self._steps:
            raise SessionError(f"Cannot move head to unknown step: {step_id}")
        self.head_id = step_id
        return self.head

    def undo(self) -> EditStep:
        """Move the head to its parent. No-op at the root."""
        if self.head.parent_id is None:
            return self.head
        return self.set_head(self.head.parent_id)

    def redo(self) -> EditStep:
        """Move to the most recent child of the head, if any."""
        children = self.children_of(self.head_id)
        if not children:
            return self.head
        return self.set_head(max(children, key=lambda s: s.sequence).id)

    def toggle_pin(self, step_id: str | None) -> EditStep | None:
        """Pin a step for comparison, or unpin if it is already pinned."""
        if step_id is None or step_id not in self._steps:
            self._pinned_id = None
        elif self._pinned_id == step_id:
            self._pinned_id = None
        else:
            self._pinned_id = step_id
        return self.pinned

    def reset(self, root_image: Image.Image) -> None:
        """Discard everything and start over from a new source image."""
        self._steps.clear()
        self._pinned_id = None
        self._counter = 0
        root = EditStep(id=uuid.uuid4().hex[:12], parent_id=None, image=root_image)
        self._steps[root.id] = root
        self.root_id = root.id
        self.head_id = root.id

    # ------------------------------------------------------------ traversal

    def path_to(self, step_id: str) -> list[EditStep]:
        """Root-to-node path. This is what the chat thread renders."""
        chain: list[EditStep] = []
        current: str | None = step_id
        seen: set[str] = set()
        while current is not None:
            if current in seen:  # defensive: a cycle would hang the UI
                logger.error("Cycle detected in edit tree at %s", current)
                break
            seen.add(current)
            step = self._steps.get(current)
            if step is None:
                break
            chain.append(step)
            current = step.parent_id
        return list(reversed(chain))

    def active_path(self) -> list[EditStep]:
        return self.path_to(self.head_id)

    def depth_of(self, step_id: str) -> int:
        return len(self.path_to(step_id)) - 1

    def leaves(self) -> list[EditStep]:
        """Terminal nodes — one per branch."""
        parents = {s.parent_id for s in self._steps.values() if s.parent_id}
        return [s for s in self._steps.values() if s.id not in parents]

    def other_branch_leaves(self) -> list[EditStep]:
        """Leaves that are not on the current path, newest first.

        These populate the "other branches" gallery so abandoned work stays one
        click away instead of being invisible.
        """
        on_path = {s.id for s in self.active_path()}
        return sorted(
            (leaf for leaf in self.leaves() if leaf.id not in on_path),
            key=lambda s: s.sequence,
            reverse=True,
        )

    def compare_pair(self, mode: str) -> tuple[Image.Image, Image.Image] | None:
        """Return ``(before, after)`` for the comparison slider.

        ``mode`` is one of ``previous``, ``original``, ``pinned``. Returns None
        when there is nothing meaningful to compare yet.
        """
        head = self.head
        if mode == "original":
            before = self.root
        elif mode == "pinned":
            before = self.pinned or self.root
        else:
            before = self._steps.get(head.parent_id) if head.parent_id else None

        if before is None or before.id == head.id:
            return None
        return before.image, head.image

    def iter_steps(self) -> Iterator[EditStep]:
        return iter(self._steps.values())

    # ----------------------------------------------------------- persistence

    def save(self, directory: Path, image_format: str = "png") -> Path:
        """Write the whole tree (images + manifest) to ``directory/<id>``."""
        session_dir = directory / self.id
        session_dir.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {
            "id": self.id,
            "created_at": self.created_at,
            "root_id": self.root_id,
            "head_id": self.head_id,
            "pinned_id": self._pinned_id,
            "steps": [],
        }

        suffix = "jpg" if image_format in {"jpg", "jpeg"} else "png"
        for step in self._steps.values():
            image_path = session_dir / f"{step.id}.{suffix}"
            if not image_path.exists():
                try:
                    step.image.save(image_path)
                except (OSError, ValueError) as exc:
                    logger.warning("Could not save step image %s: %s", step.id, exc)
                    continue
            entry = step.to_dict()
            entry["image_file"] = image_path.name
            manifest["steps"].append(entry)

        manifest_path = session_dir / "session.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    @classmethod
    def load(cls, session_dir: Path) -> "EditSession":
        """Restore a session previously written by :meth:`save`."""
        manifest_path = Path(session_dir) / "session.json"
        if not manifest_path.is_file():
            raise SessionError(f"No session.json in {session_dir}")

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SessionError(f"Could not read {manifest_path}: {exc}") from exc

        entries = {e["id"]: e for e in manifest.get("steps", [])}
        root_id = manifest.get("root_id")
        if root_id not in entries:
            raise SessionError(f"Session manifest has no root step: {manifest_path}")

        def image_for(entry: dict[str, Any]) -> Image.Image:
            return load_image(Path(session_dir) / entry["image_file"])

        session = cls(image_for(entries[root_id]), session_id=manifest.get("id"))
        # Replace the auto-created root so ids in the manifest stay valid.
        root_step = session.root
        session._steps.pop(root_step.id)
        root_step.id = root_id
        session._steps[root_id] = root_step
        session.root_id = root_id
        session.head_id = root_id

        # Insert the rest parent-before-child so every parent exists on arrival.
        remaining = [e for eid, e in entries.items() if eid != root_id]
        inserted = {root_id}
        progress = True
        while remaining and progress:
            progress = False
            for entry in list(remaining):
                if entry.get("parent_id") not in inserted:
                    continue
                try:
                    step = EditStep(
                        id=entry["id"],
                        parent_id=entry["parent_id"],
                        image=image_for(entry),
                        instruction=entry.get("instruction", ""),
                        seed=entry.get("seed"),
                        steps=entry.get("steps"),
                        guidance=entry.get("guidance"),
                        width=entry.get("width"),
                        height=entry.get("height"),
                        negative_prompt=entry.get("negative_prompt", ""),
                        duration=entry.get("duration"),
                        created_at=entry.get("created_at", session.created_at),
                        sequence=int(entry.get("sequence", 0)),
                        interrupted=entry.get("interrupted", False),
                        error=entry.get("error"),
                    )
                except (ImageError, KeyError) as exc:
                    logger.warning("Skipping unreadable step %s: %s", entry.get("id"), exc)
                    remaining.remove(entry)
                    progress = True
                    continue
                session._steps[step.id] = step
                inserted.add(step.id)
                remaining.remove(entry)
                progress = True

        if remaining:
            logger.warning("%d orphaned steps skipped in %s", len(remaining), session_dir)

        head_id = manifest.get("head_id")
        session.head_id = head_id if head_id in session._steps else root_id
        pinned = manifest.get("pinned_id")
        session._pinned_id = pinned if pinned in session._steps else None
        # Resume the counter above every restored step so new edits keep sorting last.
        session._counter = max((s.sequence for s in session._steps.values()), default=0)
        return session

    def export_step(
        self, step_id: str, directory: Path, image_format: str = "png"
    ) -> Path:
        """Save one step to the outputs directory with a metadata sidecar."""
        step = self.get(step_id)
        metadata = step.to_dict() | {
            "session_id": self.id,
            "model": "Qwen-Image-Edit-2509 (6-bit, MLX)",
            "lineage": [s.instruction for s in self.path_to(step_id) if s.instruction],
        }
        path = save_image(
            step.image,
            directory,
            step.instruction or "original",
            fmt=image_format,
            metadata=metadata,
        )
        step.saved_path = path
        return path

    def branch_gallery(self, size: int = 192) -> list[tuple[Image.Image, str]]:
        """(thumbnail, caption) pairs for the off-path branches gallery."""
        items: list[tuple[Image.Image, str]] = []
        for leaf in self.other_branch_leaves():
            label = leaf.instruction.strip() or "original"
            if len(label) > 40:
                label = label[:39] + "…"
            items.append((thumbnail(leaf.image, size), f"step {self.depth_of(leaf.id)} · {label}"))
        return items
