"""Procedural factory parts with ground-truth defects.

Used for tests, the offline demo and the ZeroDefect-Synth sanity benchmark. Every image
gets random pose, lighting and sensor noise so "good" parts are never pixel-identical.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

PARTS = ("gear", "washer", "bracket")
DEFECTS = ("scratch", "crack", "missing_hole", "stain", "chipped_edge")


def _background(rng: np.random.Generator, size: int) -> np.ndarray:
    base = rng.integers(55, 75)
    bg = np.full((size, size, 3), base, np.float32)
    bg[..., 2] += 6  # slightly bluish conveyor belt
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    gx, gy = rng.uniform(-18, 18, 2)
    bg += (xx * gx + yy * gy)[..., None]
    bg += rng.normal(0, 3, bg.shape)
    return bg


def _draw_part(kind: str, size: int, rng: np.random.Generator):
    """Returns (rgb float image with transparent bg = -1, holes list, outline mask)."""
    canvas = np.zeros((size, size, 3), np.float32)
    mask = np.zeros((size, size), np.uint8)
    c = size // 2
    metal = np.array([175, 178, 185], np.float32) + rng.normal(0, 4, 3)
    holes: list[tuple[int, int, int]] = []
    if kind == "gear":
        r_out, r_in, teeth = int(size * 0.36), int(size * 0.30), 14
        pts = []
        for i in range(teeth * 4):
            a = 2 * math.pi * i / (teeth * 4)
            r = r_out if (i % 4) in (1, 2) else r_in
            pts.append((c + r * math.cos(a), c + r * math.sin(a)))
        cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
        holes.append((c, c, int(size * 0.08)))
        for k in range(4):
            a = math.pi / 2 * k + math.pi / 4
            holes.append((int(c + size * 0.17 * math.cos(a)), int(c + size * 0.17 * math.sin(a)), int(size * 0.035)))
    elif kind == "washer":
        cv2.circle(mask, (c, c), int(size * 0.34), 255, -1)
        holes.append((c, c, int(size * 0.14)))
        metal = np.array([196, 170, 120], np.float32) + rng.normal(0, 4, 3)  # brass
    else:  # bracket / PCB-like plate
        w, h = int(size * 0.66), int(size * 0.44)
        cv2.rectangle(mask, (c - w // 2, c - h // 2), (c + w // 2, c + h // 2), 255, -1)
        metal = np.array([40, 120, 70], np.float32) + rng.normal(0, 4, 3)  # green PCB
        for dx in (-0.25, 0.25):
            for dy in (-0.12, 0.12):
                holes.append((int(c + dx * size), int(c + dy * size), int(size * 0.03)))
        # copper traces
        for k in range(4):
            y = int(c - h * 0.3 + k * h * 0.2)
            cv2.line(canvas, (c - w // 3, y), (c + w // 3, y), (200, 140, 60), 3)
    body = mask > 0
    shade = np.zeros_like(canvas)
    shade[body] = metal
    canvas = np.where(canvas.sum(-1, keepdims=True) > 0, canvas, shade)
    for (hx, hy, hr) in holes:
        cv2.circle(mask, (hx, hy), hr, 0, -1)
    canvas[mask == 0] = 0
    # brushed-metal texture
    tex = cv2.GaussianBlur(rng.normal(0, 6, (size, size)).astype(np.float32), (0, 0), 1.2)
    canvas[mask > 0] += tex[mask > 0][:, None]
    return canvas, holes, mask


def _apply_defect(kind: str, canvas, mask, holes, size, rng):
    gt = np.zeros((size, size), np.uint8)
    ys, xs = np.nonzero(mask)
    pick = rng.integers(len(xs))
    px, py = int(xs[pick]), int(ys[pick])
    if kind == "scratch":
        a = rng.uniform(0, math.pi)
        L = rng.uniform(0.12, 0.22) * size
        p1 = (int(px - L / 2 * math.cos(a)), int(py - L / 2 * math.sin(a)))
        p2 = (int(px + L / 2 * math.cos(a)), int(py + L / 2 * math.sin(a)))
        cv2.line(gt, p1, p2, 255, 2)
        gt &= mask
        canvas[gt > 0] = canvas[gt > 0] * 0.45 + 140
    elif kind == "crack":
        pts = [(px, py)]
        a = rng.uniform(0, 2 * math.pi)
        for _ in range(9):
            a += rng.uniform(-0.7, 0.7)
            pts.append((int(pts[-1][0] + 7 * math.cos(a)), int(pts[-1][1] + 7 * math.sin(a))))
        cv2.polylines(gt, [np.array(pts, np.int32)], False, 255, 2)
        gt &= mask
        canvas[gt > 0] = 25
    elif kind == "missing_hole" and holes:
        hx, hy, hr = holes[int(rng.integers(len(holes)))]
        cv2.circle(gt, (hx, hy), hr + 1, 255, -1)
        fill = canvas[mask > 0].mean(0)
        canvas[gt > 0] = fill
        mask[gt > 0] = 255
    elif kind == "stain":
        r = int(rng.uniform(0.04, 0.07) * size)
        cv2.ellipse(gt, (px, py), (r, int(r * 0.7)), rng.uniform(0, 180), 0, 360, 255, -1)
        gt &= mask
        tint = np.array([120, 70, 30], np.float32)  # rust
        canvas[gt > 0] = canvas[gt > 0] * 0.4 + tint * 0.6
    else:  # chipped_edge (also fallback when no holes)
        edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        ey, ex = np.nonzero(edge)
        k = rng.integers(len(ex))
        r = int(rng.uniform(0.035, 0.06) * size)
        cv2.circle(gt, (int(ex[k]), int(ey[k])), r, 255, -1)
        gt &= mask
        mask[gt > 0] = 0
        canvas[gt > 0] = 0
    return gt


def _render(kind, defect, size, rng, pose_jitter=True):
    canvas, holes, mask = _draw_part(kind, size, rng)
    gt = np.zeros((size, size), np.uint8)
    if defect:
        gt = _apply_defect(defect, canvas, mask, holes, size, rng)
    if pose_jitter:
        angle = rng.uniform(-3, 3)
        dx, dy = rng.uniform(-0.015, 0.015, 2) * size
        m = cv2.getRotationMatrix2D((size / 2, size / 2), angle, rng.uniform(0.98, 1.02))
        m[:, 2] += (dx, dy)
        canvas = cv2.warpAffine(canvas, m, (size, size), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, m, (size, size), flags=cv2.INTER_NEAREST)
        gt = cv2.warpAffine(gt, m, (size, size), flags=cv2.INTER_NEAREST)
    return canvas, mask, gt


def _composite(bg: np.ndarray, canvas: np.ndarray, mask: np.ndarray, x0: int, y0: int) -> None:
    s = mask.shape[0]
    region = bg[y0 : y0 + s, x0 : x0 + s]
    shadow = cv2.GaussianBlur(np.roll(mask, (6, 6), (0, 1)).astype(np.float32) / 255, (0, 0), 6)
    region *= (1 - 0.35 * shadow)[..., None]
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255, (3, 3), 0)[..., None]
    bg[y0 : y0 + s, x0 : x0 + s] = canvas * alpha + region * (1 - alpha)


def _finish(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = img * rng.uniform(0.92, 1.08) + rng.normal(0, 2.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_part(
    kind: str = "gear",
    defect: str | None = None,
    size: int = 320,
    seed: int | None = None,
    pose_jitter: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb uint8 image, ground-truth defect mask uint8 0/255)."""
    rng = np.random.default_rng(seed)
    canvas, mask, gt = _render(kind, defect, size, rng, pose_jitter)
    bg = _background(rng, size)
    _composite(bg, canvas, mask, 0, 0)
    return _finish(bg, rng), gt


def make_conveyor_frame(
    kind: str = "gear",
    defects: list[str | None] | None = None,
    part_size: int = 240,
    seed: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[str | None]]:
    """Several parts on one belt; returns (frame, part boxes, defects)."""
    defects = defects if defects is not None else [None, "scratch", None]
    rng = np.random.default_rng(seed)
    gap = 30
    w = len(defects) * (part_size + gap) + gap
    h = part_size + 2 * gap
    side = max(w, h)
    frame = _background(rng, side)[:h, :w].copy()
    boxes = []
    for i, d in enumerate(defects):
        canvas, mask, _ = _render(kind, d, part_size, np.random.default_rng(int(rng.integers(1 << 30))))
        x0 = gap + i * (part_size + gap) + int(rng.integers(-8, 9))
        y0 = gap + int(rng.integers(-8, 9))
        x0, y0 = int(np.clip(x0, 0, w - part_size)), int(np.clip(y0, 0, h - part_size))
        _composite(frame, canvas, mask, x0, y0)
        boxes.append((x0, y0, x0 + part_size, y0 + part_size))
    return _finish(frame, rng), boxes, defects
