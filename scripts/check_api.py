"""Check that your MODEL_API_KEY works with Muse Spark and SAM 3.1 before a demo.

    python scripts/check_api.py
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zerodefect import meta_api  # noqa: E402
from zerodefect.config import settings  # noqa: E402
from zerodefect.synth import make_conveyor_frame  # noqa: E402


async def main():
    if not settings.api_key:
        sys.exit("MODEL_API_KEY is not set (put it in .env or export it).")
    c = meta_api.client()
    t = time.time()
    r = c.chat.completions.create(model=settings.spark_model, messages=[{"role": "user", "content": "Reply with: ready"}])
    print(f"Muse Spark ({settings.spark_model}): {r.choices[0].message.content!r}  {time.time() - t:.1f}s")

    frame, boxes, _ = make_conveyor_frame("gear", [None, "crack", None], seed=3)
    t = time.time()
    objs = await meta_api.sam_segment_async(frame, "metal gear")
    print(f"SAM 3.1 ({settings.sam_model}): found {len(objs)} gear(s) (expected 3)  {time.time() - t:.1f}s")
    for o in objs:
        print("   box", o.box, "mask px", o.area, "conf", o.confidence)
    if objs:
        t = time.time()
        seg = await meta_api.sam_box_async(frame, objs[0].box)
        print(f"SAM 3.1 box prompt: mask px {seg.area if seg else 0}  {time.time() - t:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
