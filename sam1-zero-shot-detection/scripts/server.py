"""SAM3 Object Detection API Server.

Receives image frames, returns detected objects with text prompts.
Requires NGC PyTorch 26.03+ container for DGX Spark compatibility.
"""

import base64
import io
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

# --- flash_attn_interface shim for flash-attn 2.x (SAM 3.1 expects top-level module) ---
try:
    import flash_attn_interface  # noqa: F401
except ImportError:
    try:
        from flash_attn import flash_attn_interface
        sys.modules["flash_attn_interface"] = flash_attn_interface
        logging.getLogger(__name__).info("Shimmed flash_attn_interface from flash_attn.flash_attn_interface")
    except ImportError:
        pass  # flash-attn not installed, image-only mode still works
# --- End flash_attn shim ---

# --- DGX Spark dtype compatibility patches ---
_orig_linear = F.linear
def _safe_linear(input, weight, bias=None):
    return _orig_linear(input.to(weight.dtype), weight, bias)
F.linear = _safe_linear

_orig_sdpa = F.scaled_dot_product_attention
def _safe_sdpa(q, k, v, *a, **kw):
    return _orig_sdpa(q, k, v, *a, **kw).to(q.dtype)
F.scaled_dot_product_attention = _safe_sdpa
# --- End patches ---

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default prompts for video scene analysis
DEFAULT_PROMPTS = os.environ.get(
    "SAM3_PROMPTS",
    "person,presentation slide,code on screen,whiteboard,diagram,chart,logo",
).split(",")

SCORE_THRESHOLD = float(os.environ.get("SAM3_SCORE_THRESHOLD", "0.5"))

app = FastAPI(title="SAM3 Object Detection API", version="0.1.0")

# Global model (loaded once)
_model = None
_processor = None


def get_processor():
    global _model, _processor
    if _processor is None:
        logger.info("Loading SAM3 model...")
        _model = build_sam3_image_model()
        _processor = Sam3Processor(_model)
        logger.info("SAM3 model loaded (%.1f GiB VRAM)", torch.cuda.max_memory_allocated() / 1024**3)
    return _processor


class DetectRequest(BaseModel):
    image_base64: str
    prompts: list[str] | None = None
    score_threshold: float | None = None


class Detection(BaseModel):
    prompt: str
    count: int
    max_score: float
    boxes: list[list[float]]


class DetectResponse(BaseModel):
    detections: list[Detection]
    tags: list[str]
    elapsed_ms: int


class SegmentRequest(BaseModel):
    image_base64: str
    prompts: list[str] | None = None
    score_threshold: float | None = None


class SegmentDetection(BaseModel):
    prompt: str
    count: int
    scores: list[float]
    boxes: list[list[float]]
    masks_rle: list[dict]  # {counts: str, size: [h, w]}


class SegmentResponse(BaseModel):
    detections: list[SegmentDetection]
    tags: list[str]
    image_size: list[int]  # [h, w]
    elapsed_ms: int


@app.on_event("startup")
async def startup():
    get_processor()


@app.get("/health")
async def health():
    return {"status": "ok", "model": "sam3", "vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 1)}


@app.post("/detect")
async def detect(req: DetectRequest) -> DetectResponse:
    """Detect objects in an image using text prompts."""
    t0 = time.time()
    processor = get_processor()
    prompts = req.prompts or DEFAULT_PROMPTS
    threshold = req.score_threshold or SCORE_THRESHOLD

    # Decode image
    img_bytes = base64.b64decode(req.image_base64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((1024, int(1024 * img.height / img.width)))

    # Run detection
    state = processor.set_image(img)

    detections = []
    tags = set()

    for prompt in prompts:
        prompt = prompt.strip()
        if not prompt:
            continue
        output = processor.set_text_prompt(state=state, prompt=prompt)
        masks = output["masks"]
        scores = output["scores"]
        boxes = output.get("boxes", torch.zeros(0, 4))

        # Filter by threshold
        keep = scores >= threshold
        n = keep.sum().item()

        if n > 0:
            kept_scores = scores[keep]
            kept_boxes = boxes[keep] if boxes.numel() > 0 else torch.zeros(0, 4)
            detections.append(Detection(
                prompt=prompt,
                count=n,
                max_score=round(kept_scores.max().item(), 3),
                boxes=kept_boxes.tolist(),
            ))
            tags.add(prompt)

    elapsed_ms = int((time.time() - t0) * 1000)
    return DetectResponse(
        detections=detections,
        tags=sorted(tags),
        elapsed_ms=elapsed_ms,
    )


def _mask_to_rle(mask: np.ndarray) -> dict:
    """Encode binary mask to RLE (run-length encoding)."""
    h, w = mask.shape
    flat = mask.flatten()
    changes = np.diff(np.concatenate([[0], flat, [0]]))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    counts = []
    prev = 0
    for s, e in zip(starts, ends):
        counts.append(s - prev)  # zeros before run
        counts.append(e - s)     # ones in run
        prev = e
    if prev < len(flat):
        counts.append(len(flat) - prev)
    return {"counts": ",".join(str(c) for c in counts), "size": [h, w]}


@app.post("/segment")
async def segment(req: SegmentRequest) -> SegmentResponse:
    """Detect and segment objects, returning masks as RLE."""
    t0 = time.time()
    processor = get_processor()
    prompts = req.prompts or DEFAULT_PROMPTS
    threshold = req.score_threshold or SCORE_THRESHOLD

    # Decode image
    img_bytes = base64.b64decode(req.image_base64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((1024, int(1024 * img.height / img.width)))

    # Run detection
    state = processor.set_image(img)

    detections = []
    tags = set()

    for prompt in prompts:
        prompt = prompt.strip()
        if not prompt:
            continue
        output = processor.set_text_prompt(state=state, prompt=prompt)
        masks = output["masks"]
        scores = output["scores"]
        boxes = output.get("boxes", torch.zeros(0, 4))

        # Filter by threshold
        keep = scores >= threshold
        n = keep.sum().item()

        if n > 0:
            kept_scores = scores[keep]
            kept_boxes = boxes[keep] if boxes.numel() > 0 else torch.zeros(0, 4)
            kept_masks = masks[keep]

            # Encode masks as RLE
            masks_rle = []
            for m in kept_masks:
                binary = (m.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
                masks_rle.append(_mask_to_rle(binary))

            detections.append(SegmentDetection(
                prompt=prompt,
                count=n,
                scores=[round(s.item(), 3) for s in kept_scores],
                boxes=kept_boxes.tolist(),
                masks_rle=masks_rle,
            ))
            tags.add(prompt)

    elapsed_ms = int((time.time() - t0) * 1000)
    return SegmentResponse(
        detections=detections,
        tags=sorted(tags),
        image_size=[img.height, img.width],
        elapsed_ms=elapsed_ms,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("SAM3_PORT", "8105"))
    uvicorn.run(app, host="0.0.0.0", port=port)
