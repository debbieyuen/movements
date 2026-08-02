"""Server configuration. Paths are anchored to the repo root so the server
behaves the same no matter which directory it is launched from."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SESSIONS_DIR = REPO_ROOT / "sessions"
LIVE_DIR = REPO_ROOT / "live"

# How often the live/latest_pose*.json files are refreshed at most.
LIVE_WRITE_HZ = 30.0

# Flush buffered jsonl frames every N frames or T seconds, whichever first.
JSONL_FLUSH_FRAMES = 60
JSONL_FLUSH_SECONDS = 2.0

# Ack every Nth pose frame (per connection) instead of every frame.
ACK_EVERY = 30

# Upload chunks may not exceed Cloudflare quick-tunnel's ~100 MB request cap.
MAX_UPLOAD_CHUNK_BYTES = 32 * 1024 * 1024


def resolve_token() -> str | None:
    """Shared-secret auth for the public tunnel.

    MOCAP_TOKEN env var:
      unset        -> generate a random token and print it at startup (enforced)
      "disabled"   -> auth off (local-only development)
      anything else-> that value is the token (enforced)
    """
    raw = os.environ.get("MOCAP_TOKEN")
    if raw is None:
        token = secrets.token_urlsafe(16)
        print(f"[server] MOCAP_TOKEN not set; generated token for this run: {token}")
        print("[server] pass it to clients as ?token=... (set MOCAP_TOKEN to pin it, "
              "or MOCAP_TOKEN=disabled to turn auth off)")
        return token
    if raw.strip().lower() == "disabled" or raw.strip() == "":
        print("[server] WARNING: auth disabled (MOCAP_TOKEN=disabled). "
              "Do not expose this server through a tunnel.")
        return None
    return raw
