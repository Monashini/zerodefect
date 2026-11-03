"""End-to-end inspection: frame -> parts (SAM 3.1) -> anomaly map -> Muse Spark verdict ->
pixel-accurate defect mask (SAM 3.1 box prompt) -> annotated frame + log entry."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import meta_api
from .config import settings
from .detector import FeatureExtractor, GoldenSampleModel, heatmap_overlay
from .inspector import fuse, local_inspect, spark_inspect
from .store import Store

log = logging.getLogger("zerodefect.pipeline")

GREEN, RED, AMBER = (34, 197, 94), (239, 68, 68), (245, 158, 11)


@dataclass
class Profile:
    name: str = "Part"
    part_prompt: str = "metal part"  # SAM 3.1 noun phrase
    rotation_invariant: bool = False
    notes: str = ""
    sensitivity: float = 1.0
    created: float = field(default_factory=time.time)

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-") or "part"


def jpeg_data_url(img_rgb: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def crop_part(frame: np.ndarray, box, pad: float = 0.08) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Square crop around a part box (keeps aspect so the golden and test crops align)."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * pad)
    nx1, ny1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    nx2, ny2 = int(round(cx + side / 2)), int(round(cy + side / 2))
    # pad with edge pixels if the square leaves the frame
    padl, padt = max(0, -nx1), max(0, -ny1)
    padr, padb = max(0, nx2 - W), max(0, ny2 - H)
    src = cv2.copyMakeBorder(frame, padt, padb, padl, padr, cv2.BORDER_REPLICATE) if (padl or padt or padr or padb) else frame
    crop = src[ny1 + padt : ny2 + padt, nx1 + padl : nx2 + padl]
    return crop.copy(), (nx1, ny1, nx2, ny2)


class Engine:
    """Holds the active product profile + golden model, runs inspections."""

    def __init__(self, data_dir: Path | None = None, extractor: FeatureExtractor | None = None):
        self.data_dir = Path(data_dir or settings.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "captures").mkdir(exist_ok=True)
        self.extractor = extractor or FeatureExtractor(settings.backbone_path)
        self.store = Store(self.data_dir / "inspections.db")
        self.profile: Profile | None = None
        self.model: GoldenSampleModel | None = None
        self.golden_crop: np.ndarray | None = None
        self.golden_frame: np.ndarray | None = None
        self.last_parts_source = "none"
        self._load_active()

    # -- status ------------------------------------------------------------------------------
    def status(self) -> dict:
        return {
            "meta_api": settings.meta_enabled,
            "spark_model": settings.spark_model if settings.meta_enabled else None,
            "sam_model": settings.sam_model if settings.meta_enabled else None,
            "profile": asdict(self.profile) if self.profile else None,
            "ready": self.model is not None,
            "golden": jpeg_data_url(self.golden_crop) if self.golden_crop is not None else None,
            "threshold": self.model.threshold if self.model else None,
            "calibrated_with": self.model.calibrated_with if self.model else 0,
            "golden_count": len(self.model.golden) if self.model else 0,
        }

    # -- persistence -------------------------------------------------------------------------
    def _profile_dir(self) -> Path:
        return self.data_dir / "profiles" / self.profile.slug

    def _save(self) -> None:
        d = self._profile_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / "profile.json").write_text(json.dumps(asdict(self.profile), indent=2))
        np.savez_compressed(d / "model.npz", **self.model.state())
        cv2.imwrite(str(d / "golden.jpg"), self.golden_crop[:, :, ::-1])
        (self.data_dir / "active.txt").write_text(self.profile.slug)

    def _load_active(self) -> None:
        active = self.data_dir / "active.txt"
        if not active.exists():
            return
        d = self.data_dir / "profiles" / active.read_text().strip()
        try:
            self.profile = Profile(**json.loads((d / "profile.json").read_text()))
            self.model = GoldenSampleModel(self.extractor).load_state(np.load(d / "model.npz"))
            self.golden_crop = cv2.imread(str(d / "golden.jpg"))[:, :, ::-1].copy()
        except Exception as exc:  # corrupt or old profile: start clean
            log.warning("could not load profile %s: %s", d, exc)
            self.profile = self.model = self.golden_crop = None

    # -- part finding ------------------------------------------------------------------------
    async def find_parts(self, frame: np.ndarray, prompt: str) -> list[meta_api.SegObject]:
        if settings.meta_enabled:
            try:
                objs = await meta_api.sam_segment_async(frame, prompt)
                # drop tiny fragments; SAM finds every instance, we only want whole parts
                big = [o for o in objs if o.area > 0.004 * frame.shape[0] * frame.shape[1]]
                if big:
                    self.last_parts_source = settings.sam_model
                    return sorted(big, key=lambda o: (o.box[0], o.box[1]))
            except Exception as exc:
                log.warning("SAM 3.1 unavailable, using local segmentation: %s", exc)
        self.last_parts_source = "local"
        return meta_api.local_segment_parts(frame)

    # -- golden sample -----------------------------------------------------------------------
    async def register_golden(self, frames: list[np.ndarray], profile: Profile) -> dict:
        crops = []
        for f in frames:
            parts = await self.find_parts(f, profile.part_prompt)
            if parts:
                biggest = max(parts, key=lambda o: o.area)
                crops.append(crop_part(f, biggest.box)[0])
            else:
                crops.append(f)
        model = GoldenSampleModel(self.extractor, rotation_invariant=profile.rotation_invariant,
                                  sensitivity=profile.sensitivity)
        await asyncio.to_thread(model.fit, crops)
        self.profile, self.model, self.golden_crop = profile, model, cv2.resize(crops[0], (256, 256))
        self._save()
        return self.status()

    async def calibrate(self, frames: list[np.ndarray]) -> dict:
        crops = []
        for f in frames:
            for p in await self.find_parts(f, self.profile.part_prompt):
                crops.append(crop_part(f, p.box)[0])
        await asyncio.to_thread(self.model.calibrate, crops)
        self._save()
        return {**self.status(), "calibration_parts": len(crops)}

    def set_sensitivity(self, value: float) -> dict:
        self.model.set_sensitivity(value)
        self.profile.sensitivity = self.model.sensitivity
        self._save()
        return self.status()

    async def teach_ok(self, rec_id: int) -> dict:
        """Operator says 'this was actually OK' -> add it to the golden set and refit."""
        rec = self.store.get(rec_id)
        if not rec or not rec.get("thumb"):
            raise KeyError(rec_id)
        crop = cv2.imread(str(self.data_dir / "captures" / rec["thumb"]))[:, :, ::-1].copy()
        goldens = list(self.model.golden) + [crop]
        model = GoldenSampleModel(self.extractor, rotation_invariant=self.profile.rotation_invariant,
                                  sensitivity=self.profile.sensitivity)
        await asyncio.to_thread(model.fit, goldens)
        old_thr = self.model.threshold * self.model.sensitivity
        model.threshold = max(model.threshold, old_thr / model.sensitivity) if self.model.calibrated_with else model.threshold
        model.calibrated_with = self.model.calibrated_with
        self.model = model
        self.store.update_verdict(rec_id, "PASS_OVERRIDE")
        self._save()
        return self.status()

    # -- inspection --------------------------------------------------------------------------
    async def inspect(self, frame: np.ndarray, use_spark: bool = True) -> dict:
        if self.model is None:
            raise RuntimeError("Register a golden sample first")
        t0 = time.time()
        frame_id = uuid.uuid4().hex[:10]
        parts = await self.find_parts(frame, self.profile.part_prompt)
        if not parts:
            H, W = frame.shape[:2]
            parts = [meta_api.SegObject("0", (0, 0, W, H), np.ones((H, W), np.uint8))]
        golden = self.model.golden[0]

        async def one(idx: int, part: meta_api.SegObject) -> dict:
            crop, cbox = crop_part(frame, part.box)
            scored = await asyncio.to_thread(self.model.score, crop)
            heat = heatmap_overlay(crop, scored.anomaly_map)
            verdict = None
            if use_spark and settings.meta_enabled:
                try:
                    verdict = await spark_inspect(golden, crop, heat, scored, self.profile.name, self.profile.notes)
                except Exception as exc:
                    log.warning("Muse Spark unavailable, using local explanation: %s", exc)
            if verdict is None:
                verdict = local_inspect(golden, crop, scored)
            final = fuse(scored, verdict)

            # Defect localisation in frame pixels
            defect_mask_frame = None
            defect_source = None
            if final != "PASS":
                b = verdict.get("bbox") or {}
                ch, cw = crop.shape[:2]
                if b and (b.get("x_max", 0) > b.get("x_min", 0)):
                    dx1, dy1 = b["x_min"] * cw / 1000, b["y_min"] * ch / 1000
                    dx2, dy2 = b["x_max"] * cw / 1000, b["y_max"] * ch / 1000
                elif scored.regions:
                    dx1, dy1, dx2, dy2 = scored.regions[0].box
                else:
                    dx1 = dy1 = dx2 = dy2 = 0
                if dx2 > dx1 and dy2 > dy1:
                    fb = (
                        int(max(0, cbox[0] + dx1)), int(max(0, cbox[1] + dy1)),
                        int(min(frame.shape[1], cbox[0] + dx2)), int(min(frame.shape[0], cbox[1] + dy2)),
                    )
                    if settings.meta_enabled:
                        try:
                            seg = await meta_api.sam_box_async(frame, fb)
                            if seg is not None and seg.area > 0:
                                # keep the defect inside the part
                                defect_mask_frame = seg.mask & (part.mask if part.mask.any() else 1)
                                defect_source = settings.sam_model
                        except Exception as exc:
                            log.warning("SAM box prompt failed: %s", exc)
                    if defect_mask_frame is None or not defect_mask_frame.any():
                        amap = np.zeros(frame.shape[:2], np.float32)
                        H, W = frame.shape[:2]
                        sx1, sy1 = max(0, cbox[0]), max(0, cbox[1])
                        sx2, sy2 = min(W, cbox[2]), min(H, cbox[3])
                        resized = cv2.resize(scored.anomaly_map, (cbox[2] - cbox[0], cbox[3] - cbox[1]))
                        amap[sy1:sy2, sx1:sx2] = resized[sy1 - cbox[1] : sy2 - cbox[1], sx1 - cbox[0] : sx2 - cbox[0]]
                        defect_mask_frame = meta_api.local_refine_box(amap, fb, level=1.0)
                        defect_source = "anomaly-map"
            area_pct = None
            if defect_mask_frame is not None and part.mask.any():
                area_pct = round(100.0 * float(defect_mask_frame.sum()) / float(part.mask.sum()), 2)

            thumb = f"{frame_id}_{idx}.jpg"
            cv2.imwrite(str(self.data_dir / "captures" / thumb), crop[:, :, ::-1])
            rec_id = self.store.add({
                "product": self.profile.name, "frame_id": frame_id, "part_index": idx, "verdict": final,
                "score": round(scored.score, 3), "defect_type": verdict.get("defect_type"),
                "severity": verdict.get("severity"), "location": verdict.get("location"),
                "explanation": verdict.get("explanation"), "likely_cause": verdict.get("likely_cause"),
                "defect_area_pct": area_pct, "source": verdict.get("source"), "thumb": thumb,
                "extra": {"spark_verdict": verdict.get("verdict"), "confidence": verdict.get("confidence")},
            })
            return {
                "id": rec_id,
                "index": idx,
                "box": list(part.box),
                "verdict": final,
                "score": round(scored.score, 3),
                "detector_verdict": "FAIL" if scored.is_defect else "PASS",
                "inspector": verdict,
                "defect_area_pct": area_pct,
                "defect_mask_source": defect_source,
                "heatmap": jpeg_data_url(heat),
                "crop": jpeg_data_url(crop),
                "_mask": defect_mask_frame,
                "_part_mask": part.mask,
            }

        results = await asyncio.gather(*(one(i, p) for i, p in enumerate(parts)))
        annotated = annotate(frame, results)
        for r in results:
            r.pop("_mask", None)
            r.pop("_part_mask", None)
        n_fail = sum(r["verdict"] == "FAIL" for r in results)
        n_review = sum(r["verdict"] == "REVIEW" for r in results)
        callouts = [r["inspector"].get("callout") for r in results if r["verdict"] != "PASS"]
        if n_fail or n_review:
            speech = " ".join(f"Part {r['index'] + 1}: {r['inspector'].get('callout') or 'check'}"
                              for r in results if r["verdict"] != "PASS") if len(results) > 1 else (callouts[0] or "Reject.")
        else:
            speech = "All parts pass." if len(results) > 1 else "Pass."
        return {
            "frame_id": frame_id,
            "verdict": "FAIL" if n_fail else ("REVIEW" if n_review else "PASS"),
            "parts": results,
            "annotated": jpeg_data_url(annotated),
            "speech": speech,
            "parts_source": self.last_parts_source,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }


def annotate(frame: np.ndarray, results: list[dict]) -> np.ndarray:
    out = frame.copy()
    H, W = frame.shape[:2]
    lw = max(2, int(round(min(H, W) / 250)))
    overlay = out.copy()
    for r in results:
        color = {"PASS": GREEN, "FAIL": RED}.get(r["verdict"], AMBER)
        m = r.get("_mask")
        if m is not None and m.any():
            overlay[m > 0] = (255, 40, 40)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    for r in results:
        color = {"PASS": GREEN, "FAIL": RED}.get(r["verdict"], AMBER)
        pm = r.get("_part_mask")
        if pm is not None and pm.any() and pm.sum() < pm.size:
            cnts, _ = cv2.findContours(pm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, color, lw)
        else:
            x1, y1, x2, y2 = r["box"]
            cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), color, lw)
        m = r.get("_mask")
        if m is not None and m.any():
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, (255, 255, 0), max(1, lw - 1))
        x1, y1 = r["box"][0], r["box"][1]
        label = f"#{r['index'] + 1} {r['verdict']}"
        if r["verdict"] != "PASS":
            dt = r['inspector'].get('defect_type') or ''
            label += f" {'defect' if dt in ('', 'other') else dt.replace('_', ' ')}"
        fs = max(0.5, min(H, W) / 700)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        ty = max(th + 6, y1 - 4)
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), color, -1)
        cv2.putText(out, label, (x1 + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
    return out
