"""Muse Spark as the QC inspector: compares the part with its golden sample, reads the
detector heatmap, and returns a structured, plain-language verdict."""
from __future__ import annotations

import json
import logging

import numpy as np

from .config import settings
from .detector import Region, ScoreResult
from .meta_api import async_client, to_data_url

log = logging.getLogger("zerodefect.inspector")

DEFECT_TYPES = [
    "none",
    "scratch",
    "crack",
    "dent",
    "missing_feature",
    "extra_material",
    "stain_or_contamination",
    "discoloration",
    "chipped_or_broken_edge",
    "bent_or_deformed",
    "misprint_or_label_error",
    "other",
]

VERDICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
        "defect_type": {"type": "string", "enum": DEFECT_TYPES},
        "location": {"type": "string", "description": "Plain words, e.g. 'outer edge at 2 o'clock'"},
        "bbox": {
            "type": "object",
            "additionalProperties": False,
            "description": "Defect box on IMAGE 2, normalised 0-1000 grid. All zeros if PASS.",
            "properties": {
                "x_min": {"type": "integer"},
                "y_min": {"type": "integer"},
                "x_max": {"type": "integer"},
                "y_max": {"type": "integer"},
            },
            "required": ["x_min", "y_min", "x_max", "y_max"],
        },
        "severity": {"type": "integer", "description": "0 = none, 1 = cosmetic ... 5 = safety-critical"},
        "confidence": {"type": "number"},
        "explanation": {"type": "string", "description": "One sentence for the operator"},
        "callout": {"type": "string", "description": "<= 8 words, spoken aloud, e.g. 'Crack near top-left edge, reject.'"},
        "likely_cause": {"type": "string", "description": "Short guess at the process cause, or empty"},
    },
    "required": [
        "verdict",
        "defect_type",
        "location",
        "bbox",
        "severity",
        "confidence",
        "explanation",
        "callout",
        "likely_cause",
    ],
}

SYSTEM_PROMPT = (
    "You are a senior quality-control inspector on a small factory line. You judge parts only "
    "against the golden sample the owner provided. Differences in lighting, shadow, background, "
    "small shifts or rotation, and camera noise are NOT defects. Real defects are physical: "
    "scratches, cracks, dents, chips, missing or extra features, stains, wrong colour, bending, "
    "misprints. Be decisive and concrete. Speak like a helpful supervisor, in simple words."
)


def _region_text(regions: list[Region], hw: tuple[int, int]) -> str:
    h, w = hw
    if not regions:
        return "none above threshold"
    out = []
    for r in regions[:3]:
        x1, y1, x2, y2 = r.box
        out.append(
            f"[x_min={int(x1 * 1000 / w)}, y_min={int(y1 * 1000 / h)}, x_max={int(x2 * 1000 / w)}, "
            f"y_max={int(y2 * 1000 / h)}] strength {r.peak:.2f}"
        )
    return "; ".join(out)


async def spark_inspect(
    golden: np.ndarray,
    part: np.ndarray,
    heat_overlay: np.ndarray,
    scored: ScoreResult,
    part_name: str,
    notes: str = "",
) -> dict:
    """Ask Muse Spark for a verdict. Raises on API errors (caller falls back)."""
    user_text = (
        f"Product: {part_name}. {notes}\n"
        "IMAGE 1 = golden sample (a perfect part). IMAGE 2 = part under inspection. "
        "IMAGE 3 = IMAGE 2 with an anomaly heatmap from a zero-shot vision detector "
        "(red/yellow = looks different from the golden sample).\n"
        f"Detector anomaly score: {scored.score:.2f} (1.00 = fail threshold). "
        f"Detector hot-spots on IMAGE 2: {_region_text(scored.regions, part.shape[:2])}.\n"
        "The detector can raise false alarms on glare, shadows or pose; it can also miss subtle "
        "defects. Look carefully at IMAGE 2 versus IMAGE 1 and decide PASS or FAIL. If FAIL, "
        "give the defect box on IMAGE 2 on a 0-1000 grid (0,0 = top-left)."
    )
    resp = await async_client().chat.completions.create(
        model=settings.spark_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": to_data_url(golden, max_side=768)}},
                    {"type": "image_url", "image_url": {"url": to_data_url(part, max_side=768)}},
                    {"type": "image_url", "image_url": {"url": to_data_url(heat_overlay, max_side=768)}},
                ],
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "InspectionVerdict", "schema": VERDICT_SCHEMA, "strict": True},
        },
        reasoning_effort="low",
    )
    data = json.loads(resp.choices[0].message.content)
    data["source"] = settings.spark_model
    return data


# ----------------------------------------------------------------------------------------
# Offline explanation (no API key): still plain-language, derived from the heatmap + pixels
# ----------------------------------------------------------------------------------------

def _where(cx: float, cy: float, w: int, h: int) -> str:
    fx, fy = cx / w, cy / h
    vert = "top" if fy < 0.36 else "bottom" if fy > 0.64 else ""
    horz = "left" if fx < 0.36 else "right" if fx > 0.64 else ""
    name = "-".join(p for p in (vert, horz) if p) or "centre"
    r = np.hypot(fx - 0.5, fy - 0.5)
    zone = "edge" if r > 0.3 else "face"
    return f"{name} {zone}" if name != "centre" else "centre"


def local_inspect(golden: np.ndarray, part: np.ndarray, scored: ScoreResult) -> dict:
    h, w = part.shape[:2]
    if not scored.is_defect or not scored.regions:
        return {
            "verdict": "PASS",
            "defect_type": "none",
            "location": "",
            "bbox": {"x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0},
            "severity": 0,
            "confidence": float(np.clip(1.0 - scored.score / 2, 0.5, 0.99)),
            "explanation": "Matches the golden sample.",
            "callout": "Pass.",
            "likely_cause": "",
            "source": "local",
        }
    r = scored.regions[0]
    x1, y1, x2, y2 = r.box
    # The local fallback only localises; naming the defect type is Muse Spark's job.
    kind = "other"
    where = _where((x1 + x2) / 2, (y1 + y2) / 2, w, h)
    sev = int(np.clip(round(1 + (r.peak - 1) * 3 + (r.area_px / (h * w)) * 40), 1, 5))
    return {
        "verdict": "FAIL",
        "defect_type": kind,
        "location": where,
        "bbox": {
            "x_min": int(x1 * 1000 / w),
            "y_min": int(y1 * 1000 / h),
            "x_max": int(x2 * 1000 / w),
            "y_max": int(y2 * 1000 / h),
        },
        "severity": sev,
        "confidence": float(np.clip(0.5 + (r.peak - 1) * 0.5, 0.5, 0.95)),
        "explanation": f"Does not match the golden sample at the {where}.",
        "callout": f"Defect at the {where}. Reject.",
        "likely_cause": "",
        "source": "local",
    }


def fuse(detector: ScoreResult, verdict: dict) -> str:
    """Combine the two opinions. Disagreements go to a human instead of being guessed."""
    spark_fail = verdict.get("verdict") == "FAIL"
    conf = float(verdict.get("confidence", 0.5))
    if verdict.get("source") == "local":
        return "FAIL" if detector.is_defect else "PASS"
    if spark_fail and (detector.score > 0.8 or conf >= 0.7):
        return "FAIL"
    if not spark_fail and detector.score <= 1.0:
        return "PASS"
    if detector.score > 1.6 and not spark_fail:
        return "REVIEW"
    if spark_fail:
        return "REVIEW"
    return "PASS" if conf >= 0.75 else "REVIEW"
