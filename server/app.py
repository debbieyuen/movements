"""Pose server v2: one FastAPI process behind one cloudflared tunnel.

    uvicorn server.app:app --host 0.0.0.0 --port 8765

Endpoints:
    WS   /ws?token=...        pose frames + presence + countdown
    GET  /health              liveness probe
    POST /upload/...          chunked video upload (see routes below)

The server validates, timestamps, converts MediaPipe axes to the canonical
z-up frame exactly once (server/protocol.py), and persists:

    sessions/<sid>/pose_<role>.jsonl   buffered append log per camera
    sessions/<sid>/videos/<file>       uploaded recordings
    live/latest_pose.json              latest canonical frame (30 Hz, atomic)
    live/latest_pose_<role>.json       per-role variant

It does NOT retarget. Retargeting belongs to consumers
(server/live_viewer_h1.py for the live demo, mocap/ for offline clips).
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .protocol import (
    COORD_ZUP_XFWD,
    DEFAULT_DEPTH_SCALE,
    PROTOCOL_VERSION,
    build_live_frame,
    mp_world_to_zup,
)
from .sessions import SessionStore, role_slug

TOKEN: Optional[str] = None  # set on startup so tests can override env first
STORE = SessionStore()

# Per-role throttle stamps for the live files.
_last_live_write: Dict[str, float] = {}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global TOKEN
    TOKEN = config.resolve_token()
    config.LIVE_DIR.mkdir(parents=True, exist_ok=True)
    config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="movements pose server", version="2.0", lifespan=_lifespan)

# The capture app is always a different origin from this server (localhost:3000
# vs :8765 in dev; a Vercel/LAN page vs a Cloudflare tunnel in the field), and
# chunk uploads send Content-Type: application/octet-stream, which triggers a
# CORS preflight. Without this the browser's OPTIONS gets a 405 and every
# upload silently fails. Auth is the ?token= query param, not the origin, and
# no cookies are used -- so allow any origin but no credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _token_ok(token: Optional[str]) -> bool:
    return TOKEN is None or token == TOKEN


async def _atomic_write_json(path: Path, obj: Any) -> None:
    """Atomic temp+replace write, off the event loop, with a bounded retry for
    Windows-style transient PermissionError while a reader holds the file."""

    def _write() -> None:
        tmp = path.parent / (path.name + ".tmp")
        tmp.write_text(json.dumps(obj), encoding="utf-8")
        last_err: Optional[Exception] = None
        for _ in range(40):
            try:
                tmp.replace(path)
                return
            except PermissionError as e:  # pragma: no cover - Windows only
                last_err = e
                time.sleep(0.005)
        print(f"[server] atomic write to {path.name} kept failing: {last_err!r}")

    await asyncio.to_thread(_write)


def _canonical_frame(data: dict, server_unix_ms: int) -> Optional[dict]:
    """Convert an inbound pose message (v1 legacy or v2 wire) to the canonical
    on-disk frame. Returns None if the message has no usable landmarks."""
    if data.get("v") == PROTOCOL_VERSION and data.get("type") == "pose":
        world = data.get("world")
        if not isinstance(world, list) or len(world) != 33:
            return None
        return {
            "v": PROTOCOL_VERSION,
            "sessionId": str(data.get("sessionId", "unknown")),
            "role": str(data.get("role", "camera")),
            "seq": int(data.get("seq", 0) or 0),
            "tMs": float(data.get("tMs", 0.0) or 0.0),
            "unixMs": int(data.get("unixMs", 0) or 0),
            "serverUnixMs": server_unix_ms,
            "coord": COORD_ZUP_XFWD,
            "meta": {
                "depthScale": DEFAULT_DEPTH_SCALE,
                "source": str(data.get("source", "mediapipe-tasks-vision")),
            },
            "world": mp_world_to_zup(world),
        }
    # Legacy v1 message from the pre-rework client.
    return build_live_frame(data, server_unix_ms=server_unix_ms)


async def _persist_frame(state, frame: dict) -> None:
    slug = role_slug(frame["role"])
    if state.writer(frame["role"]).append(frame):
        await state.writer(frame["role"]).flush()

    now = time.monotonic()
    if now - _last_live_write.get(slug, 0.0) >= 1.0 / config.LIVE_WRITE_HZ:
        _last_live_write[slug] = now
        await _atomic_write_json(config.LIVE_DIR / "latest_pose.json", frame)
        await _atomic_write_json(
            config.LIVE_DIR / f"latest_pose_{slug}.json", frame)


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "protocol": PROTOCOL_VERSION}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, token: Optional[str] = Query(default=None)):
    if not _token_ok(token):
        await ws.close(code=4401, reason="bad token")
        return
    await ws.accept()
    state = None
    n_frames = 0
    last_stats = time.monotonic()

    try:
        while True:
            try:
                data = json.loads(await ws.receive_text())
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"ok": False, "error": "invalid json"}))
                continue

            msg_type = data.get("type")

            if msg_type in ("hello", "heartbeat"):
                state = STORE.get_or_create(data.get("sessionId", "unknown"))
                changed = state.clients.get(ws) != data.get("role")
                state.clients[ws] = data.get("role")
                if msg_type == "hello" or changed:
                    await state.broadcast(
                        {"v": 2, "type": "presence",
                         "sessionId": state.session_id, "roles": state.roles()})
                continue

            if msg_type == "go":
                state = STORE.get_or_create(data.get("sessionId", "unknown"))
                await state.broadcast(
                    {"v": 2, "type": "countdown",
                     "sessionId": state.session_id,
                     "seconds": int(data.get("seconds", 3))})
                continue

            # Everything else is a pose frame. One bad frame must never tear
            # down the socket -- the robot would freeze on the last pose.
            try:
                server_unix_ms = int(time.time() * 1000)
                frame = _canonical_frame(data, server_unix_ms)
                if frame is not None:
                    state = STORE.get_or_create(frame["sessionId"])
                    if ws not in state.clients:
                        state.clients[ws] = frame["role"]
                    await _persist_frame(state, frame)
                    n_frames += 1
                    if n_frames % config.ACK_EVERY == 0:
                        await ws.send_text(json.dumps(
                            {"v": 2, "type": "ack", "seq": frame["seq"]}))
            except Exception as e:  # noqa: BLE001 - keep the stream alive
                print(f"[server] frame processing error, skipped: {e!r}")

            now = time.monotonic()
            if now - last_stats >= 1.0:
                last_stats = now
                print(f"[server] {n_frames} frames this connection "
                      f"({data.get('role', '?')})")

    except WebSocketDisconnect:
        pass
    finally:
        await STORE.drop(ws)


# --------------------------------------------------------------------------
# Chunked video upload.
#
# The browser slices a recording into chunks (Cloudflare quick tunnels cap
# request bodies around 100 MB) and POSTs them in order; "complete" verifies
# the byte count and atomically renames the .part file into place.
# --------------------------------------------------------------------------
def _upload_path(session_id: str, role: str, filename: str) -> Path:
    safe_name = Path(filename).name  # strip any path components
    if not safe_name or safe_name.startswith("."):
        raise HTTPException(status_code=400, detail="bad filename")
    d = config.SESSIONS_DIR / role_slug(session_id) / "videos"
    d.mkdir(parents=True, exist_ok=True)
    return d / safe_name


@app.post("/upload/{session_id}/{role}/{filename}")
async def upload_chunk(
    session_id: str,
    role: str,
    filename: str,
    request: Request,
    index: int = Query(ge=0),
    total: int = Query(ge=1),
    token: Optional[str] = Query(default=None),
) -> dict:
    if not _token_ok(token):
        raise HTTPException(status_code=401, detail="bad token")
    body = await request.body()
    if len(body) > config.MAX_UPLOAD_CHUNK_BYTES:
        raise HTTPException(status_code=413, detail="chunk too large")
    part = _upload_path(session_id, role, filename).with_suffix(
        _upload_path(session_id, role, filename).suffix + ".part")

    def _append() -> None:
        mode = "wb" if index == 0 else "ab"
        with part.open(mode) as f:
            f.write(body)

    await asyncio.to_thread(_append)
    return {"ok": True, "index": index, "received": len(body)}


@app.post("/upload/{session_id}/{role}/{filename}/complete")
async def upload_complete(
    session_id: str,
    role: str,
    filename: str,
    size: int = Query(ge=0),
    token: Optional[str] = Query(default=None),
) -> dict:
    if not _token_ok(token):
        raise HTTPException(status_code=401, detail="bad token")
    final = _upload_path(session_id, role, filename)
    part = final.with_suffix(final.suffix + ".part")
    if not part.exists():
        raise HTTPException(status_code=404, detail="no upload in progress")
    actual = part.stat().st_size
    if actual != size:
        raise HTTPException(
            status_code=409,
            detail=f"size mismatch: expected {size}, got {actual}")
    part.replace(final)
    return {"ok": True, "path": str(final), "bytes": actual}
