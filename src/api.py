"""Local REST API mounted alongside the Gradio app.

Useful for scripting batch edits or driving the model from another tool on the
same machine. Bound to localhost by default, like the UI.

    POST /api/edit      multipart or JSON (base64) -> edited PNG
    GET  /api/status    model + device state
    GET  /api/health    liveness

Example:

    curl -X POST http://127.0.0.1:7860/api/edit \\
      -F image=@photo.jpg \\
      -F 'instruction=change the shirt colour to blue' \\
      -F steps=25 -o edited.png
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

from PIL import Image

from .config import AppConfig
from .device import active_memory_gb, detect_device, peak_memory_gb
from .inference import EditRequest, ImageEditor, InferenceError
from .model_loader import QwenEditModel
from .utils import ImageError, coerce_seed, load_image

logger = logging.getLogger(__name__)


def _decode_image(payload: str | bytes) -> Image.Image:
    """Accept raw bytes or a base64 string (with or without a data: prefix)."""
    if isinstance(payload, str):
        if "," in payload and payload.strip().startswith("data:"):
            payload = payload.split(",", 1)[1]
        try:
            payload = base64.b64decode(payload, validate=True)
        except (ValueError, TypeError) as exc:
            raise ImageError(f"Invalid base64 image data: {exc}") from exc
    try:
        return load_image(Image.open(io.BytesIO(payload)))
    except (OSError, ValueError) as exc:
        raise ImageError(f"Could not decode image: {exc}") from exc


def mount_api(demo: Any, config: AppConfig, model: QwenEditModel) -> None:
    """Attach REST routes to the FastAPI app behind the Gradio Blocks.

    Failures here are logged and swallowed: a broken optional API must never
    stop the UI from starting.
    """
    try:
        from fastapi import File, Form, HTTPException, Request, UploadFile
        from fastapi.responses import JSONResponse, Response
    except ImportError:
        logger.info("FastAPI unavailable; REST API disabled.")
        return

    app = getattr(demo, "app", None)
    if app is None:
        logger.warning("Gradio app not exposed; REST API disabled.")
        return

    editor = ImageEditor(model, enable_previews=False)
    gen = config.generation

    @app.get("/api/health")
    async def health() -> JSONResponse:  # noqa: D401
        return JSONResponse({"status": "ok"})

    @app.get("/api/status")
    async def status() -> JSONResponse:
        device = detect_device()
        return JSONResponse(
            {
                "model": model.label,
                "repo_id": model.repo_id,
                "loaded": model.is_loaded,
                "load_seconds": model.load_seconds,
                "busy": editor.is_busy,
                "device": {
                    "chip": device.chip,
                    "backend": device.backend,
                    "metal": device.metal_available,
                    "memory_gb": round(device.total_memory_gb, 1),
                },
                "memory": {
                    "mode": model.memory_mode,
                    "active_gb": round(active_memory_gb(), 2),
                    "peak_gb": round(peak_memory_gb(), 2),
                },
                "defaults": {
                    "steps": gen.steps,
                    "guidance": gen.guidance,
                    "scheduler": gen.scheduler,
                },
            }
        )

    async def _run(
        image: Image.Image,
        instruction: str,
        steps: int,
        guidance: float,
        seed: Any,
        width: int | None,
        height: int | None,
        negative_prompt: str,
        as_json: bool,
    ) -> Response:
        try:
            request = EditRequest(
                images=[image],
                instruction=instruction,
                seed=coerce_seed(seed),
                steps=steps,
                guidance=guidance,
                width=width,
                height=height,
                negative_prompt=negative_prompt or "",
                scheduler=gen.scheduler,
            )
            result = editor.edit(request)
        except InferenceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("API edit failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        buffer = io.BytesIO()
        result.image.save(buffer, format="PNG")
        payload = buffer.getvalue()

        if as_json:
            return JSONResponse(
                {
                    "image_base64": base64.b64encode(payload).decode("ascii"),
                    "seed": result.seed,
                    "steps": result.steps,
                    "guidance": result.guidance,
                    "width": result.width,
                    "height": result.height,
                    "duration_seconds": round(result.duration, 2),
                    "peak_memory_gb": round(result.peak_memory_gb, 2),
                }
            )

        return Response(
            content=payload,
            media_type="image/png",
            headers={
                "X-Seed": str(result.seed),
                "X-Duration-Seconds": f"{result.duration:.2f}",
                "X-Resolution": f"{result.width}x{result.height}",
            },
        )

    @app.post("/api/edit")
    async def edit_multipart(
        request: Request,
        image: UploadFile | None = File(default=None),
        instruction: str = Form(default=""),
        steps: int = Form(default=gen.steps),
        guidance: float = Form(default=gen.guidance),
        seed: str = Form(default=""),
        width: int | None = Form(default=None),
        height: int | None = Form(default=None),
        negative_prompt: str = Form(default=""),
        response_format: str = Form(default="png"),
    ) -> Response:
        # Multipart when a file is attached, otherwise fall back to a JSON body.
        if image is not None:
            try:
                pil = _decode_image(await image.read())
            except ImageError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return await _run(
                pil, instruction, steps, guidance, seed or None,
                width, height, negative_prompt, response_format == "json",
            )

        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=400,
                detail="Provide either a multipart 'image' file or a JSON body "
                "with an 'image' base64 field.",
            ) from exc

        if not body.get("image"):
            raise HTTPException(status_code=400, detail="Missing 'image'.")

        try:
            pil = _decode_image(body["image"])
        except ImageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return await _run(
            pil,
            body.get("instruction", ""),
            int(body.get("steps", gen.steps)),
            float(body.get("guidance", gen.guidance)),
            body.get("seed"),
            body.get("width"),
            body.get("height"),
            body.get("negative_prompt", ""),
            body.get("response_format", "json") == "json",
        )

    logger.info("REST API mounted at /api/edit, /api/status, /api/health")
