"""Per-session state: connected clients, presence, and buffered frame logs."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import WebSocket

from . import config


def role_slug(role: Any) -> str:
    s = str(role if role not in (None, "?", "") else "unknown").strip().lower()
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in s) or "unknown"


class JsonlBuffer:
    """Buffered append-only jsonl writer.

    Frames are appended to an in-memory list and flushed to disk off the event
    loop (asyncio.to_thread) every JSONL_FLUSH_FRAMES frames or
    JSONL_FLUSH_SECONDS seconds, and on session close. This replaces the old
    server's six synchronous file writes per frame.
    """

    def __init__(self, path: Path):
        self.path = path
        self._buf: list[str] = []
        self._last_flush = time.monotonic()
        self._lock = asyncio.Lock()

    def append(self, frame: dict) -> bool:
        """Queue a frame; returns True if a flush is due."""
        self._buf.append(json.dumps(frame))
        return (
            len(self._buf) >= config.JSONL_FLUSH_FRAMES
            or (time.monotonic() - self._last_flush) >= config.JSONL_FLUSH_SECONDS
        )

    async def flush(self) -> None:
        async with self._lock:
            if not self._buf:
                return
            lines, self._buf = self._buf, []
            self._last_flush = time.monotonic()
            payload = "\n".join(lines) + "\n"

            def _write() -> None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(payload)

            await asyncio.to_thread(_write)


class SessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.dir = config.SESSIONS_DIR / session_id
        self.clients: Dict[WebSocket, Optional[str]] = {}
        self._writers: Dict[str, JsonlBuffer] = {}

    def writer(self, role: str) -> JsonlBuffer:
        slug = role_slug(role)
        if slug not in self._writers:
            self._writers[slug] = JsonlBuffer(self.dir / f"pose_{slug}.jsonl")
        return self._writers[slug]

    def roles(self) -> list[str]:
        return sorted({r for r in self.clients.values() if r})

    async def broadcast(self, msg: dict) -> None:
        text = json.dumps(msg)
        for ws in list(self.clients):
            try:
                await ws.send_text(text)
            except Exception:  # noqa: BLE001 - a dead socket must not block the rest
                pass

    async def close_writer_buffers(self) -> None:
        for w in self._writers.values():
            await w.flush()


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}

    def get_or_create(self, session_id: str) -> SessionState:
        sid = str(session_id or "unknown")
        if sid not in self._sessions:
            self._sessions[sid] = SessionState(sid)
        return self._sessions[sid]

    def find(self, ws: WebSocket) -> Optional[SessionState]:
        for state in self._sessions.values():
            if ws in state.clients:
                return state
        return None

    async def drop(self, ws: WebSocket) -> None:
        state = self.find(ws)
        if state is None:
            return
        state.clients.pop(ws, None)
        await state.broadcast(
            {"v": 2, "type": "presence", "sessionId": state.session_id,
             "roles": state.roles()})
        if not state.clients:
            await state.close_writer_buffers()
