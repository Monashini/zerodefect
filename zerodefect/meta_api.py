"""Thin wrappers around Meta Model API (OpenAI-compatible, https://api.meta.ai/v1).

* SAM 3.1   — finds every part in a frame from a noun phrase, and turns a rough defect box
              into a pixel-accurate defect mask (box prompt).
* Muse Spark — vision-language inspector and shift-report agent (see inspector.py / report.py).
"""
from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .config import settings

log = logging.getLogger("zerodefect.meta")

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:  # pragma: no cover
    OpenAI = AsyncOpenAI = None

try:
    import meta_sam_parser as msp
except ImportError:  # pragma: no cover
    msp = None


def to_data_url(img_rgb: np.ndarray, quality: int = 88, max_side: int = 1024) -> str:
    h, w = img_rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img_rgb = cv2.resize(img_rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode image")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


_sync_client = None
_async_client = None


def client():
    global _sync_client
    if _sync_client is None:
        if OpenAI is None:
            raise RuntimeError("pip install openai")
        _sync_client = OpenAI(
            base_url=settings.base_url, api_key=settings.api_key, timeout=settings.request_timeout, max_retries=2
        )
    return _sync_client


def async_client():
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(
            base_url=settings.base_url, api_key=settings.api_key, timeout=settings.request_timeout, max_retries=2
        )
    return _async_client


# ----------------------------------------------------------------------------------------
# SAM 3.1
# ----------------------------------------------------------------------------------------

@dataclass
class SegObject:
    object_id: str
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 (half-open) in source pixels
    mask: np.ndarray  # uint8 0/1, full frame size
    confidence: float | None = None

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def _records_to_objects(records, frame_hw: tuple[int, int]) -> list[SegObject]:
    H, W = frame_hw
    boxes, masks = {}, {}
    for rec in records:
        if rec.kind == "box":
            boxes[rec.object_id] = rec
        elif rec.kind == "mask":
            masks[rec.object_id] = rec
    objects = []
    for oid, b in boxes.items():
        x1, y1 = max(0, int(round(b.left))), max(0, int(round(b.top)))
        x2, y2 = min(W, int(round(b.right))), min(H, int(round(b.bottom)))
        full = np.zeros((H, W), np.uint8)
        m = masks.get(oid)
        if m is not None and x2 > x1 and y2 > y1:
            raster = np.frombuffer(msp.decode_mask_to_raster(m.mask), np.uint8).reshape(m.mask.height, m.mask.width)
            # The mask raster is box-local; resample it onto its source-media bounds.
            bx1, by1 = max(0, int(round(m.bounds.left))), max(0, int(round(m.bounds.top)))
            bx2, by2 = min(W, int(round(m.bounds.right))), min(H, int(round(m.bounds.bottom)))
            if bx2 > bx1 and by2 > by1:
                full[by1:by2, bx1:bx2] = cv2.resize(raster, (bx2 - bx1, by2 - by1), interpolation=cv2.INTER_NEAREST)
        else:
            full[y1:y2, x1:x2] = 1
        objects.append(SegObject(str(oid), (x1, y1, x2, y2), full, b.confidence))
    return objects


async def _parse_events(events, frame_hw):
    parsed = msp.parse_responses_stream(events, msp.image_segmentation_format())
    result = await parsed.final_result()
    return _records_to_objects(result.records, frame_hw)


def parse_sam_text(text: str, frame_hw: tuple[int, int]) -> list[SegObject]:
    """Parse a complete SAM output_text lane (used by tests and for logged responses)."""

    async def gen():
        lane = {"item_id": "msg_local", "output_index": 0, "content_index": 0}
        if text:
            yield {"type": "response.output_text.delta", "delta": text, **lane}
        yield {"type": "response.output_text.done", "text": text, **lane}
        yield {"type": "response.completed", "response": {"status": "completed"}}

    if not text:
        return []
    return asyncio.run(_parse_events(gen(), frame_hw))


async def sam_segment_async(img_rgb: np.ndarray, prompt: str) -> list[SegObject]:
    """Concept prompt: return every instance of `prompt` (e.g. "metal gear") in the image."""
    events = await async_client().responses.create(
        model=settings.sam_model,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": to_data_url(img_rgb, max_side=4096, quality=92)},
                ],
            }
        ],
        stream=True,
        metadata={"mask_encoding": "one_bit", "include_confidence": "true"},
    )
    try:
        return await _parse_events(events, img_rgb.shape[:2])
    except Exception as exc:  # zero matches or empty lane
        log.info("SAM returned no objects for %r (%s)", prompt, exc)
        return []


async def sam_box_async(img_rgb: np.ndarray, box: tuple[int, int, int, int]) -> SegObject | None:
    """Box prompt: pixel-accurate mask of whatever lies inside `box` (used for defects)."""
    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    text = msp.build_object_prompt(
        size=(W, H),
        objects=[msp.PromptObject(id=1, prompts=[msp.FramePrompt(box=msp.PromptBox(x1, y1, x2, y2))])],
    )
    events = await async_client().responses.create(
        model=settings.sam_model,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text},
                    {"type": "input_image", "image_url": to_data_url(img_rgb, max_side=4096, quality=92)},
                ],
            }
        ],
        stream=True,
        metadata={"mask_encoding": "one_bit"},
    )
    try:
        objs = await _parse_events(events, (H, W))
    except Exception as exc:
        log.info("SAM box prompt returned nothing (%s)", exc)
        return None
    return objs[0] if objs else None


# ----------------------------------------------------------------------------------------
# Local fallbacks (no API key / offline / API error)
# ----------------------------------------------------------------------------------------

def local_segment_parts(img_rgb: np.ndarray, min_area_frac: float = 0.01) -> list[SegObject]:
    """Classical foreground segmentation: parts differ from the (roughly uniform) belt."""
    H, W = img_rgb.shape[:2]
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    border = np.concatenate([lab[:6].reshape(-1, 3), lab[-6:].reshape(-1, 3), lab[:, :6].reshape(-1, 3), lab[:, -6:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    dist = cv2.GaussianBlur(dist, (0, 0), 1.5)
    thr = max(12.0, cv2.threshold(np.clip(dist * 3, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 3)
    fg = (dist > thr).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    objs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area_frac * H * W:
            continue
        m = (labels == i).astype(np.uint8)
        # fill internal holes so a washer's centre counts as part of the part
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(m)
        cv2.drawContours(filled, cnts, -1, 1, -1)
        objs.append(SegObject(str(len(objs)), (int(x), int(y), int(x + w), int(y + h)), filled, None))
    objs.sort(key=lambda o: (o.box[0], o.box[1]))
    return objs


def local_refine_box(anomaly_map: np.ndarray, box: tuple[int, int, int, int], level: float = 1.0) -> np.ndarray:
    """Defect mask from the anomaly map itself when SAM is unavailable."""
    x1, y1, x2, y2 = box
    m = np.zeros(anomaly_map.shape, np.uint8)
    roi = anomaly_map[y1:y2, x1:x2]
    m[y1:y2, x1:x2] = (roi >= max(level, roi.max() * 0.78)).astype(np.uint8)
    return m
