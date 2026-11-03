"""FastAPI app: REST API + the single-page operator UI in /web."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import ROOT, settings
from .pipeline import Engine, Profile
from .report import build_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = FastAPI(title="ZeroDefect", version="1.0")
engine: Engine | None = None
MAX_SIDE = 1600


def get_engine() -> Engine:
    global engine
    if engine is None:
        engine = Engine()
    return engine


async def read_image(file: UploadFile) -> np.ndarray:
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"Could not read image {file.filename!r}")
    h, w = img.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img[:, :, ::-1].copy()  # RGB


@app.get("/api/status")
def status():
    return get_engine().status()


@app.post("/api/golden")
async def golden(
    files: list[UploadFile] = File(...),
    name: str = Form("Part"),
    part_prompt: str = Form("metal part"),
    rotation_invariant: bool = Form(False),
    notes: str = Form(""),
):
    frames = [await read_image(f) for f in files]
    profile = Profile(name=name.strip() or "Part", part_prompt=part_prompt.strip() or "part",
                      rotation_invariant=rotation_invariant, notes=notes.strip())
    return await get_engine().register_golden(frames, profile)


@app.post("/api/calibrate")
async def calibrate(files: list[UploadFile] = File(...)):
    e = get_engine()
    if e.model is None:
        raise HTTPException(400, "Register a golden sample first")
    return await e.calibrate([await read_image(f) for f in files])


@app.post("/api/sensitivity")
async def sensitivity(value: float = Form(...)):
    e = get_engine()
    if e.model is None:
        raise HTTPException(400, "Register a golden sample first")
    return e.set_sensitivity(value)


@app.post("/api/inspect")
async def inspect(file: UploadFile = File(...), use_spark: bool = Form(True)):
    e = get_engine()
    if e.model is None:
        raise HTTPException(400, "Register a golden sample first")
    return await e.inspect(await read_image(file), use_spark=use_spark)


@app.post("/api/feedback/{rec_id}")
async def feedback(rec_id: int, label: str = Form(...)):
    """label = 'ok' (false alarm -> learn it) or 'defect' (confirm)."""
    e = get_engine()
    if label == "ok":
        try:
            return await e.teach_ok(rec_id)
        except KeyError:
            raise HTTPException(404, "unknown inspection")
    e.store.update_verdict(rec_id, "FAIL")
    return {"ok": True}


@app.get("/api/history")
def history(limit: int = 50):
    rows = get_engine().store.rows(limit=limit)
    for r in rows:
        r["thumb_url"] = f"/captures/{r['thumb']}" if r.get("thumb") else None
    return rows


@app.get("/api/stats")
def stats(hours: float = 12):
    e = get_engine()
    since = time.time() - hours * 3600
    product = e.profile.name if e.profile else None
    return {
        "summary": e.store.summary(since, product=product),
        "by_hour": e.store.by_hour(since, product=product),
        "breakdown": e.store.defect_breakdown(since, product=product),
    }


@app.post("/api/report")
async def report(hours: float = Form(12)):
    e = get_engine()
    return await build_report(e.store, time.time() - hours * 3600, e.profile.name if e.profile else None)


@app.post("/api/reset_log")
def reset_log():
    get_engine().store.clear()
    return {"ok": True}


@app.get("/api/demo/frame")
def demo_frame(kind: str = "gear", defects: str = "random", seed: int | None = None, golden: bool = False):
    """Synthetic conveyor frame for trying ZeroDefect without a camera (JPEG)."""
    from fastapi.responses import Response

    from .synth import DEFECTS, PARTS, make_conveyor_frame

    if kind not in PARTS:
        raise HTTPException(400, f"kind must be one of {PARTS}")
    rng = np.random.default_rng(seed)
    if golden:
        plan = [None]
    elif defects == "random":
        plan = [None if rng.random() < 0.6 else str(rng.choice(DEFECTS)) for _ in range(4)]
    else:
        plan = [None if d in ("", "ok", "none") else d for d in defects.split(",")]
    frame, _, _ = make_conveyor_frame(kind, plan, seed=int(rng.integers(1 << 30)))
    ok, buf = cv2.imencode(".jpg", frame[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"X-Defects": ",".join(d or "ok" for d in plan)})


@app.get("/captures/{name}")
def capture(name: str):
    p = (get_engine().data_dir / "captures" / Path(name).name)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.exception_handler(RuntimeError)
async def runtime_error(_, exc: RuntimeError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
