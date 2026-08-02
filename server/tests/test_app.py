import json

import pytest
from fastapi.testclient import TestClient

from server import app as app_module
from server import config


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(config, "LIVE_DIR", tmp_path / "live")
    monkeypatch.setenv("MOCAP_TOKEN", "sekret")
    app_module._last_live_write.clear()
    with TestClient(app_module.app) as c:
        yield c


def _legacy_frame(session="s1", role="camera", idx=1):
    lm = [{"x": 0.0, "y": -0.1, "z": 0.05, "visibility": 0.9}] * 33
    return {
        "sessionId": session, "role": role, "frameIndex": idx,
        "timeMs": 100.0 * idx, "unixMs": 1753980041233,
        "landmarks": lm, "worldLandmarks": lm,
    }


def _v2_frame(session="s1", role="camera", seq=1):
    return {
        "v": 2, "type": "pose", "sessionId": session, "role": role,
        "seq": seq, "tMs": 100.0 * seq, "unixMs": 1753980041233,
        "coord": "mp-camera",
        "world": [[0.0, -0.1, 0.05, 0.9]] * 33,
    }


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_ws_rejects_bad_token(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/ws?token=wrong") as ws:
            ws.send_text(json.dumps({"type": "hello", "role": "camera",
                                     "sessionId": "s1"}))
            ws.receive_text()


def test_legacy_and_v2_frames_become_canonical(client):
    with client.websocket_connect("/ws?token=sekret") as ws:
        ws.send_text(json.dumps({"type": "hello", "role": "camera",
                                 "sessionId": "s1"}))
        ws.receive_text()  # presence
        ws.send_text(json.dumps(_legacy_frame(idx=1)))
        ws.send_text(json.dumps(_v2_frame(seq=2)))
        # messages are processed in order: a hello after the frames guarantees
        # both were handled once its presence reply arrives
        ws.send_text(json.dumps({"type": "hello", "role": "camera",
                                 "sessionId": "s1"}))
        ws.receive_text()

    live = json.loads((config.LIVE_DIR / "latest_pose.json").read_text())
    assert live["coord"] == "zup-xfwd"
    assert len(live["world"]) == 33 and len(live["world"][0]) == 4
    # y_mp=-0.1 -> Z=+0.1 (upright convention applied)
    assert live["world"][0][2] == pytest.approx(0.1)

    jsonl = config.SESSIONS_DIR / "s1" / "pose_camera.jsonl"
    assert jsonl.exists()
    lines = [json.loads(l) for l in jsonl.read_text().splitlines()]
    assert [f["seq"] for f in lines] == [1, 2]
    assert all(f["coord"] == "zup-xfwd" for f in lines)


def test_countdown_is_session_scoped(client):
    with client.websocket_connect("/ws?token=sekret") as a, \
         client.websocket_connect("/ws?token=sekret") as b:
        a.send_text(json.dumps({"type": "hello", "role": "camera",
                                "sessionId": "sessA"}))
        a.receive_text()  # presence
        b.send_text(json.dumps({"type": "hello", "role": "camera",
                                "sessionId": "sessB"}))
        b.receive_text()  # presence

        a.send_text(json.dumps({"type": "go", "sessionId": "sessA",
                                "seconds": 3}))
        msg = json.loads(a.receive_text())
        assert msg["type"] == "countdown" and msg["sessionId"] == "sessA"

        # sessB must NOT receive the countdown; next thing b sees should be
        # its own echo test message ack path -- simplest: send a fresh hello
        # and confirm the next message is presence, not countdown.
        b.send_text(json.dumps({"type": "hello", "role": "camera2",
                                "sessionId": "sessB"}))
        msg_b = json.loads(b.receive_text())
        assert msg_b["type"] == "presence"


def test_chunked_upload_roundtrip(client):
    data1, data2 = b"a" * 1000, b"b" * 500
    for i, chunk in enumerate([data1, data2]):
        r = client.post(
            f"/upload/s1/camera/take1.webm?index={i}&total=2&token=sekret",
            content=chunk)
        assert r.status_code == 200, r.text
    r = client.post(
        f"/upload/s1/camera/take1.webm/complete?size={len(data1) + len(data2)}"
        f"&token=sekret")
    assert r.status_code == 200, r.text
    final = config.SESSIONS_DIR / "s1" / "videos" / "take1.webm"
    assert final.exists() and final.stat().st_size == 1500
    assert not final.with_suffix(".webm.part").exists()


def test_upload_size_mismatch_rejected(client):
    client.post("/upload/s1/camera/x.webm?index=0&total=1&token=sekret",
                content=b"abc")
    r = client.post("/upload/s1/camera/x.webm/complete?size=999&token=sekret")
    assert r.status_code == 409


def test_upload_requires_token(client):
    r = client.post("/upload/s1/camera/x.webm?index=0&total=1&token=nope",
                    content=b"abc")
    assert r.status_code == 401


def test_upload_preflight_allowed_cross_origin(client):
    """The capture page is always a different origin from this server, and
    the octet-stream chunk body triggers a CORS preflight."""
    r = client.options(
        "/upload/s1/camera/x.webm?index=0&total=1&token=sekret",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 200, r.text
    assert r.headers.get("access-control-allow-origin") in ("*", "http://localhost:3000")
