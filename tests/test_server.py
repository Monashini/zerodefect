import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch, extractor):
    from zerodefect import server
    from zerodefect.config import settings
    from zerodefect.pipeline import Engine

    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(server, "engine", Engine(data_dir=tmp_path, extractor=extractor))
    return TestClient(server.app)


def test_http_flow(client):
    assert client.get("/api/status").json()["ready"] is False
    assert client.post("/api/inspect", files={"file": ("a.jpg", b"xx", "image/jpeg")}).status_code == 400
    g = client.get("/api/demo/frame?kind=gear&golden=true&seed=1").content
    r = client.post("/api/golden", files=[("files", ("g.jpg", g, "image/jpeg"))], data={"name": "Gear", "part_prompt": "metal gear"})
    assert r.status_code == 200 and r.json()["ready"]
    f = client.get("/api/demo/frame?kind=gear&defects=ok,crack&seed=5")
    assert f.headers["x-defects"] == "ok,crack"
    r = client.post("/api/inspect", files={"file": ("f.jpg", f.content, "image/jpeg")}).json()
    assert [p["verdict"] for p in r["parts"]] == ["PASS", "FAIL"]
    assert client.post("/api/sensitivity", data={"value": "1.2"}).json()["profile"]["sensitivity"] == pytest.approx(1.2)
    assert client.post(f"/api/feedback/{r['parts'][1]['id']}", data={"label": "defect"}).json()["ok"]
    stats = client.get("/api/stats").json()
    assert stats["summary"]["total_parts"] == 2 and stats["summary"]["fail"] == 1
    hist = client.get("/api/history").json()
    assert client.get(hist[0]["thumb_url"]).status_code == 200
    assert "markdown" in client.post("/api/report", data={"hours": 12}).json()
    assert client.get("/").status_code == 200
