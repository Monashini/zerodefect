"""Meta Model API integration tested against a faithful mock of the wire format."""
import asyncio
import json
import types

import cv2
import numpy as np
from meta_sam_parser._mask_codec import _encode_raster

from zerodefect import meta_api
from zerodefect.synth import make_conveyor_frame


def sam_line(objs, W, H):
    """objs: list of (id, (x1,y1,x2,y2), mask HxW over the box)"""
    recs = []
    for oid, (x1, y1, x2, y2), m in objs:
        h, w = m.shape
        recs.append(
            f"{oid}<|box;x1={x1};y1={y1};x2={x2 - 1};y2={y2 - 1};w={W};h={H};c=0.95|>"
            f"<|mask;x=0;y=0;c=0.95;data={h},{w},{_encode_raster(m.flatten().tolist(), w, h)}|>"
        )
    return "<0f>" + ",".join(recs)


class FakeStream:
    def __init__(self, text):
        lane = {"item_id": "msg_1", "output_index": 0, "content_index": 0}
        self.events = []
        if text:
            half = len(text) // 2  # split across deltas like a real stream
            self.events += [types.SimpleNamespace(type="response.output_text.delta", delta=text[:half], **lane),
                            types.SimpleNamespace(type="response.output_text.delta", delta=text[half:], **lane)]
        self.events += [types.SimpleNamespace(type="response.output_text.done", text=text, **lane),
                        types.SimpleNamespace(type="response.completed", response={"status": "completed"})]

    def __aiter__(self):
        async def gen():
            for e in self.events:
                yield e
        return gen()


class FakeClient:
    """Mimics openai.AsyncOpenAI for responses (SAM) and chat.completions (Muse Spark)."""

    def __init__(self, frame_boxes, spark_reply):
        self.frame_boxes = frame_boxes
        self.spark_reply = spark_reply
        self.calls = []
        self.responses = types.SimpleNamespace(create=self._responses)
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._chat))

    async def _responses(self, **kw):
        self.calls.append(("sam", kw))
        content = kw["input"][0]["content"]
        prompt = content[0]["text"]
        url = content[1]["image_url"]
        assert kw["model"] == "sam-3.1" and kw["stream"] is True and url.startswith("data:image/jpeg;base64,")
        import base64
        img = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        if prompt.startswith("<0f>"):  # box prompt -> return a small blob inside the box
            import re
            x1, y1, x2, y2 = [int(v) for v in re.findall(r"x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+)", prompt)[0]]
            x2, y2 = x2 + 1, y2 + 1
            m = np.zeros((y2 - y1, x2 - x1), np.uint8)
            cv2.ellipse(m, ((x2 - x1) // 2, (y2 - y1) // 2), (max(1, (x2 - x1) // 3), max(1, (y2 - y1) // 3)), 0, 0, 360, 1, -1)
            return FakeStream(sam_line([(1, (x1, y1, x2, y2), m)], W, H))
        objs = []
        for i, (x1, y1, x2, y2) in enumerate(self.frame_boxes):
            m = np.zeros((y2 - y1, x2 - x1), np.uint8)
            cv2.circle(m, ((x2 - x1) // 2, (y2 - y1) // 2), int((x2 - x1) * 0.36), 1, -1)
            objs.append((i, (x1, y1, x2, y2), m))
        return FakeStream(sam_line(objs, W, H))

    async def _chat(self, **kw):
        self.calls.append(("spark", kw))
        assert kw["model"].startswith("muse-spark")
        reply = self.spark_reply(kw)
        if isinstance(reply, types.SimpleNamespace):
            msg = reply
        else:
            msg = types.SimpleNamespace(content=json.dumps(reply), tool_calls=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def test_sam_concept_prompt_roundtrip(monkeypatch):
    frame, boxes, _ = make_conveyor_frame("gear", [None, None, None], seed=4)
    fake = FakeClient(boxes, lambda kw: {})
    monkeypatch.setattr(meta_api, "async_client", lambda: fake)
    objs = asyncio.run(meta_api.sam_segment_async(frame, "metal gear"))
    assert len(objs) == 3
    for o, b in zip(objs, boxes):
        assert o.box == tuple(b)
        assert 0.3 < o.area / ((b[2] - b[0]) * (b[3] - b[1])) < 0.5
        assert o.confidence == 0.95


def test_sam_box_prompt(monkeypatch):
    frame, boxes, _ = make_conveyor_frame("gear", [None], seed=4)
    fake = FakeClient(boxes, lambda kw: {})
    monkeypatch.setattr(meta_api, "async_client", lambda: fake)
    seg = asyncio.run(meta_api.sam_box_async(frame, (40, 50, 100, 90)))
    ys, xs = np.nonzero(seg.mask)
    assert xs.min() >= 40 and xs.max() < 100 and ys.min() >= 50 and ys.max() < 90
    assert "<0f>1<|box;x1=40;y1=50;x2=99;y2=89" in fake.calls[0][1]["input"][0]["content"][0]["text"]


def test_parse_empty_and_text():
    assert meta_api.parse_sam_text("", (10, 10)) == []
