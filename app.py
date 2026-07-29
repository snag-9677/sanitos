#!/usr/bin/env python3
"""Qwen Image Edit — local text-guided image editing on Apple Silicon.

    python app.py                     # launch the web UI
    python app.py --preload           # download/load weights, then launch
    python app.py --check             # environment report only, no model
    python app.py --config other.yaml # use a different configuration

Runs entirely offline after the first download. No cloud APIs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# The project root must be importable before anything under src/ is touched.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import ConfigError, configure_logging, load_config  # noqa: E402
from src.device import detect_device, startup_banner  # noqa: E402

logger = logging.getLogger("qwen_image_edit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="Local text-guided image editing with Qwen Image Edit 6-bit on Apple Silicon.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml.")
    parser.add_argument(
        "--model", type=str, default=None, metavar="ID",
        help="Which catalog model to start with (see --list-models).",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List selectable models with sizes, then exit.",
    )
    parser.add_argument("--host", type=str, default=None, help="Override the bind address.")
    parser.add_argument("--port", type=int, default=None, help="Override the port.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    parser.add_argument(
        "--preload", action="store_true",
        help="Download and load the model before serving instead of on first edit.",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open a browser on launch."
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Print the environment report and exit without loading the model.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Check the cached weight files for truncation, then exit.",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show per-component model download progress, then exit.",
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="With --status, refresh live with throughput and ETA until complete.",
    )

    fetch = parser.add_argument_group("model fetching")
    fetch.add_argument(
        "--download", action="store_true",
        help="Download and verify the weights with retries, then exit.",
    )
    fetch.add_argument(
        "--import-from", type=Path, default=None, metavar="DIR",
        help="Adopt a weight directory copied from another machine, then exit.",
    )
    fetch.add_argument(
        "--mirror", type=str, default=None, metavar="URL",
        help="Alternative HuggingFace endpoint for restricted networks.",
    )
    fetch.add_argument(
        "--jobs", type=int, default=4, help="Parallel download workers (default 4)."
    )
    fetch.add_argument(
        "--no-xet", action="store_true",
        help="Disable HuggingFace's Xet transport; use plain HTTP.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
        return 2

    if args.model:
        try:
            config.model.active = config.model.entry(args.model).id
        except ConfigError as exc:
            print(f"{exc}", file=sys.stderr)
            return 2

    configure_logging(config.logging)
    device = detect_device()

    if args.list_models:
        from src.families import get_family as _family
        from src.model_loader import is_model_cached as _cached

        print(f"\n  Selectable models  (machine has {device.usable_memory_gb:.0f} GB)\n")
        for candidate in config.model.catalog:
            fits = candidate.size_gb + 2 <= device.usable_memory_gb
            mark = "*" if candidate.id == config.model.active else " "
            on_disk = _cached(candidate.repo_id, config.model.cache_dir, _family(candidate.family))
            print(
                f"  {mark} {candidate.id:16s} {candidate.size_gb:5.1f} GB  "
                f"{'fits' if fits else 'TOO BIG':8s} "
                f"{'cached' if on_disk else 'not downloaded':15s} {candidate.label}"
            )
            if candidate.notes:
                print(f"      {candidate.notes}")
        print("\n  * = active.  Select with --model <id>, or in the UI.\n")
        return 0

    # Route HuggingFace at the project model directory *before* anything
    # imports huggingface_hub, which snapshots these into constants on import.
    from src.families import get_family
    from src.model_loader import EditModel, is_model_cached

    entry = config.model.active_entry
    family = get_family(entry.family)
    # `auto` picks per model, so it must be resolved after the model is known.
    memory_mode = config.memory.resolve(family.working_set_gb, device.usable_memory_gb)
    # Only bound memory where there is no swap to fall back on.
    memory_budget = device.gpu_memory_gb if device.backend == "mlx-cuda" else None

    print(startup_banner(device, family.label, memory_mode, family.working_set_gb))

    cached = is_model_cached(entry.repo_id, config.model.cache_dir, family)
    size_note = f"not downloaded yet (~{family.working_set_gb:.0f} GB on first edit)"
    print(f"  Weights       {'cached' if cached else size_note}")
    print(f"  Repo          {entry.repo_id}")
    print(f"  Model dir     {config.model.cache_dir}")
    print(f"  Outputs       {config.output.dir}")
    print()

    if args.import_from is not None:
        from src.model_loader import ModelLoadError, import_model

        try:
            path = import_model(args.import_from, config.model.cache_dir, entry.repo_id)
        except ModelLoadError as exc:
            print(f"\n  ❌ {exc}\n", file=sys.stderr)
            return 1
        print(f"\n  ✅ Weights imported to {path}\n")
        return 0

    if args.download:
        from src.model_loader import ModelLoadError, download_model

        try:
            download_model(
                entry.repo_id,
                config.model.cache_dir,
                family=family,
                workers=args.jobs,
                disable_xet=args.no_xet or None,
                endpoint=args.mirror,
            )
        except KeyboardInterrupt:
            print("\n  Interrupted. Re-run to resume from where it stopped.\n")
            return 130
        except ModelLoadError as exc:
            print(f"\n  ❌ {exc}\n", file=sys.stderr)
            return 1
        print("\n  ✅ Weights downloaded and verified.\n")
        return 0

    if args.status or args.watch:
        from src.model_loader import (
            download_status,
            format_download_status,
            watch_download,
        )

        if args.watch:
            return watch_download(entry.repo_id, config.model.cache_dir)

        status = download_status(entry.repo_id, config.model.cache_dir)
        print(format_download_status(status))
        print()
        return 0 if status["complete"] else 1

    if args.verify:
        from src.model_loader import verify_weights

        if not cached:
            print("  Weights are not downloaded yet — nothing to verify.")
            return 1

        from mflux.models.common.resolution.path_resolution import PathResolution

        root = PathResolution.resolve(
            path=entry.repo_id,
            patterns=family.load_weight_definition().get_download_patterns(),
        )
        problems = verify_weights(Path(root)) if root else ["Could not resolve the model directory."]
        if problems:
            print("  ❌ Weight problems found:")
            for problem in problems:
                print(f"     - {problem}")
            print("\n     Delete the snapshot under model.cache_dir and re-run to re-download.")
            return 1
        print("  ✅ All weight shards are complete.")
        return 0

    if args.check:
        if device.backend == "cpu":
            print("  ⚠️  Metal unavailable — see the troubleshooting section in README.md.")
            return 1
        print("  ✅ Ready to run.")
        return 0

    model = EditModel(
        repo_id=entry.repo_id,
        cache_dir=config.model.cache_dir,
        family=family,
        label=entry.label,
        memory_mode=memory_mode,
        memory_budget_gb=memory_budget,
        cache_limit_bytes=config.memory.cache_limit_bytes,
        vae_tiling=config.memory.vae_tiling,
        quantize=config.model.quantize,
        lora_paths=config.model.lora_paths,
        lora_scales=config.model.lora_scales,
    )

    if args.preload or config.model.preload:
        print("  Preloading weights (this takes a while on first run)…\n")
        try:
            model.ensure_loaded(
                progress=lambda p: print(f"    {p.stage}{f' — {p.detail}' if p.detail else ''}")
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Preload failed: %s", exc)
            print(f"\n  ❌ {exc}\n", file=sys.stderr)
            return 1
        print()

    from src.ui import build_ui, launch_style

    demo = build_ui(config, device, model)

    host = args.host or config.ui.server_name
    port = args.port or config.ui.server_port
    print(f"  Starting UI on http://{host}:{port}\n")

    try:
        demo.queue(default_concurrency_limit=4).launch(
            server_name=host,
            server_port=port,
            share=args.share or config.ui.share,
            inbrowser=not args.no_browser and config.ui.inbrowser,
            show_error=True,
            quiet=True,
            # Return control so the REST routes can be attached to the live
            # FastAPI app; launch() builds it, so mounting earlier is discarded.
            prevent_thread_lock=True,
            **launch_style(),
        )

        if config.api_enabled:
            from src.api import mount_api

            mount_api(demo, config, model)
            print(f"  REST API   http://{host}:{port}/api/edit\n")

        demo.block_thread()
    except KeyboardInterrupt:
        print("\n  Shutting down.")
    except OSError as exc:
        logger.error("Could not start the server: %s", exc)
        print(
            f"\n  ❌ Could not bind {host}:{port} — {exc}\n"
            f"     Another instance may be running. Try --port {port + 1}.\n",
            file=sys.stderr,
        )
        return 1
    finally:
        model.unload()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
