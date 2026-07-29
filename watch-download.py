#!/usr/bin/env python3
"""Watch the model download until it finishes.

The ~32 GB download happens inside the app, so there is no progress bar in your
shell. Run this from a second terminal to see live progress, throughput, and an
ETA:

    ./watch-download.py                  # refresh every 5s until done
    ./watch-download.py --interval 30    # gentler refresh
    ./watch-download.py --once           # print once and exit

Exit codes: 0 complete, 1 still downloading (with --once) or stalled,
130 interrupted. That makes it scriptable:

    ./watch-download.py && python app.py     # launch as soon as weights land

Watching is read-only — stopping this script never affects the download.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))


def _reexec_in_venv() -> None:
    """Re-run under the project virtualenv if invoked with a bare `python3`.

    ``./watch-download.py`` should work without activating anything, so if the
    dependencies aren't importable but the project venv has them, hand over.
    """
    import importlib.util

    venv_python = PROJECT_DIR / "venv" / "bin" / "python"
    if not venv_python.is_file():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    if importlib.util.find_spec("yaml") is not None:
        return
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])


_reexec_in_venv()

from src.config import ConfigError, load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="watch-download.py",
        description="Live progress for the Qwen Image Edit model download.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml.")
    parser.add_argument(
        "--interval", type=float, default=5.0, help="Seconds between refreshes (default 5)."
    )
    parser.add_argument(
        "--once", action="store_true", help="Print the current status once and exit."
    )
    parser.add_argument(
        "--stall-after", type=float, default=120.0,
        help="Warn after this many seconds without progress (default 120).",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
        return 2

    from src.model_loader import (
        download_status,
        format_download_status,
        watch_download,
    )

    repo_id = config.model.repo_id
    cache_dir = config.model.cache_dir

    print(f"\n  Model   {repo_id}")
    print(f"  Cache   {cache_dir}\n")

    if args.once:
        status = download_status(repo_id, cache_dir)
        print(format_download_status(status))
        print()
        return 0 if status["complete"] else 1

    return watch_download(
        repo_id, cache_dir, interval=args.interval, stall_after=args.stall_after
    )


if __name__ == "__main__":
    raise SystemExit(main())
