"""End-to-end: golden -> inspect with (mocked) SAM 3.1 + Muse Spark, and fully offline."""
import asyncio
import types

import pytest

from zerodefect import meta_api, pipeline, report
from zerodefect.config import settings
from zerodefect.synth import make_conveyor_frame

from test_meta_api import FakeClient


def spark_judge(kw):
    if kw.get("tools"):  # report agent
        msgs = kw["messages"]
        if not any(m.get("role") == "tool" for m in msgs if isinstance(m, dict)):
            calls = [types.SimpleNamespace(id=f"c{i}", function=types.SimpleNamespace(name=n, arguments="{}"))
                     for i, n in enumerate(["get_summary", "get_defect_breakdown", "get_defects_by_hour"])]
            return types.SimpleNamespace(content="", tool_calls=calls)
        return types.SimpleNamespace(content="**Shift summary**\n- 3 parts", tool_calls=None)
    text = kw["messages"][1]["content"][0]["text"]
    score = float(text.split("Detector anomaly score: ")[1].split(" ")[0])
    assert len(kw["messages"][1]["content"]) == 4  # prompt + golden + part + heatmap
    assert kw["response_format"]["type"] == "json_schema"
    if score > 1.0:
        return {"verdict": "FAIL", "defect_type": "crack", "location": "lower face",
                "bbox": {"x_min": 300, "y_min": 500, "x_max": 600, "y_max": 800}, "severity": 4,
                "confidence": 0.9, "explanation": "Hairline crack running from the bore.",
                "callout": "Crack near the bore, reject.", "likely_cause": "Over-pressing at the punch station."}
    return {"verdict": "PASS", "defect_type": "none", "location": "", "bbox": {"x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0},
            "severity": 0, "confidence": 0.95, "explanation": "Matches the golden sample.", "callout": "Pass.", "likely_cause": ""}


@pytest.fixture
def engine(tmp_path, extractor):
    return pipeline.Engine(data_dir=tmp_path, extractor=extractor)


def test_offline_pipeline(engine, monkeypatch):
    monkeypatch.setattr(settings, "api_key", None)
    golden, _, _ = make_conveyor_frame("gear", [None], seed=1)
    asyncio.run(engine.register_golden([golden], pipeline.Profile(name="Gear", part_prompt="metal gear")))
    frame, boxes, plan = make_conveyor_frame("gear", [None, "crack", "stain"], seed=8)
    r = asyncio.run(engine.inspect(frame))
    assert len(r["parts"]) == 3 and r["parts_source"] == "local"
    assert r["parts"][0]["verdict"] == "PASS"
    assert r["parts"][1]["verdict"] == "FAIL" and r["parts"][2]["verdict"] == "FAIL"
    assert r["verdict"] == "FAIL" and r["annotated"].startswith("data:image/jpeg")
    # teach: mark a failure as OK -> golden set grows
    asyncio.run(engine.teach_ok(r["parts"][2]["id"]))
    assert len(engine.model.golden) == 2
    rep = asyncio.run(report.build_report(engine.store))
    assert rep["source"] == "local" and "parts inspected" in rep["markdown"]
    # engine reload from disk
    again = pipeline.Engine(data_dir=engine.data_dir, extractor=engine.extractor)
    assert again.model is not None and again.profile.name == "Gear"


def test_meta_pipeline_with_mock(engine, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "test-key")
    monkeypatch.setattr(settings, "offline", False)
    golden, gboxes, _ = make_conveyor_frame("gear", [None], seed=1)
    frame, boxes, plan = make_conveyor_frame("gear", [None, "crack", None], seed=8)
    fake = FakeClient(gboxes, spark_judge)
    monkeypatch.setattr(meta_api, "async_client", lambda: fake)
    monkeypatch.setattr(report, "async_client", lambda: fake)
    import zerodefect.inspector as insp
    monkeypatch.setattr(insp, "async_client", lambda: fake)
    asyncio.run(engine.register_golden([golden], pipeline.Profile(name="Gear", part_prompt="metal gear")))
    fake.frame_boxes = boxes
    r = asyncio.run(engine.inspect(frame))
    assert r["parts_source"] == "sam-3.1"
    verdicts = [p["verdict"] for p in r["parts"]]
    assert verdicts[1] == "FAIL" and verdicts[0] == "PASS"
    bad = r["parts"][1]
    assert bad["inspector"]["source"] == "muse-spark-1.3" and bad["defect_mask_source"] == "sam-3.1"
    assert bad["defect_area_pct"] and bad["defect_area_pct"] > 0
    assert "Crack near the bore" in r["speech"]
    rep = asyncio.run(report.build_report(engine.store))
    assert rep["source"] == "muse-spark-1.3" and [t["tool"] for t in rep["trace"]] == [
        "get_summary", "get_defect_breakdown", "get_defects_by_hour"]


def test_meta_failure_falls_back(engine, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "test-key")
    monkeypatch.setattr(settings, "offline", False)

    async def boom(*a, **k):
        raise RuntimeError("network down")
    broken = types.SimpleNamespace(responses=types.SimpleNamespace(create=boom),
                                   chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=boom)))
    monkeypatch.setattr(meta_api, "async_client", lambda: broken)
    import zerodefect.inspector as insp
    monkeypatch.setattr(insp, "async_client", lambda: broken)
    golden, _, _ = make_conveyor_frame("washer", [None], seed=1)
    asyncio.run(engine.register_golden([golden], pipeline.Profile(name="Washer", part_prompt="brass washer")))
    frame, _, _ = make_conveyor_frame("washer", [None, "chipped_edge"], seed=3)
    r = asyncio.run(engine.inspect(frame))
    assert [p["verdict"] for p in r["parts"]] == ["PASS", "FAIL"]
    assert all(p["inspector"]["source"] == "local" for p in r["parts"])
