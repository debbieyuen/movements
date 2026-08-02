"""Pose protocol v2: the single place axis conventions are defined.

Canonical world frame ("zup-xfwd"): right-handed, +X forward (toward the
camera), +Y left, +Z up, gravity -Z, meters, quaternions wxyz.

MediaPipe world landmarks ("mp-camera") arrive hip-centered in meters with
+x image-right, +y image-DOWN, +z away from the camera. They are converted
to the canonical frame exactly once, here, before anything is persisted:

    X = -s * z_mp        (toward the camera = forward)
    Y =      x_mp        (image-right = subject-left = world-left)
    Z =     -y_mp        (image-up = world-up)

The rotation part has det = +1 (a proper rotation), so handedness and all
cross-product logic downstream are preserved. `s` is the depth compression
factor: MediaPipe's monocular depth is far noisier than its image-plane
coordinates, so we shrink the depth axis at this boundary and record the
factor in frame meta so any consumer can divide it back out.

The authoritative message shapes live in protocol/pose_protocol.schema.json;
the pydantic models here must stay in sync (enforced by server/tests).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from pydantic import BaseModel, Field

PROTOCOL_VERSION = 2

COORD_MP_CAMERA = "mp-camera"
COORD_ZUP_XFWD = "zup-xfwd"

# 1.0 = raw MediaPipe depth, 0.0 = ignore depth entirely. ~0.35 keeps gross
# reaching toward/away from the camera while killing straight-arm-looks-bent
# jitter. Recorded in every persisted frame's meta.depthScale.
DEFAULT_DEPTH_SCALE = 0.35

Landmark4 = List[float]  # [x, y, z, visibility]


class FrameMeta(BaseModel):
    depthScale: float = Field(gt=0)
    source: str


class PoseFrameLive(BaseModel):
    """Server-persisted frame: live/latest_pose.json and session jsonl."""

    v: int = PROTOCOL_VERSION
    sessionId: str
    role: str
    seq: int = Field(ge=0)
    tMs: float
    unixMs: int
    serverUnixMs: int
    coord: str = COORD_ZUP_XFWD
    meta: FrameMeta
    world: List[Landmark4] = Field(min_length=33, max_length=33)


def _landmark_xyzv(p: Any) -> Optional[Landmark4]:
    """Accept either the legacy dict form {x,y,z,visibility} or [x,y,z,vis]."""
    try:
        if isinstance(p, dict):
            x, y, z = float(p["x"]), float(p["y"]), float(p["z"])
            vis = p.get("visibility")
        else:
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            vis = p[3] if len(p) > 3 else None
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    v = float(vis) if isinstance(vis, (int, float)) else 1.0
    return [x, y, z, v]


def mp_world_to_zup(
    world: Sequence[Any], depth_scale: float = DEFAULT_DEPTH_SCALE
) -> List[Landmark4]:
    """Convert MediaPipe world landmarks to the canonical z-up frame."""
    out: List[Landmark4] = []
    for p in world:
        lm = _landmark_xyzv(p)
        if lm is None:
            out.append([0.0, 0.0, 0.0, 0.0])  # unusable point: visibility 0
            continue
        x, y, z, vis = lm
        out.append([-depth_scale * z, x, -y, vis])
    return out


def build_live_frame(
    legacy_frame: dict,
    *,
    server_unix_ms: int,
    depth_scale: float = DEFAULT_DEPTH_SCALE,
    source: str = "mediapipe-pose-legacy",
) -> Optional[dict]:
    """Build a canonical v2 frame from a legacy (v1) client pose message.

    Legacy messages carry {sessionId, role, frameIndex, timeMs, unixMs,
    landmarks, worldLandmarks} with dict-shaped landmarks. Returns None if the
    frame has no usable world landmarks (the caller should just skip it).
    """
    world_raw = legacy_frame.get("worldLandmarks")
    if not isinstance(world_raw, list) or len(world_raw) != 33:
        return None
    frame = PoseFrameLive(
        sessionId=str(legacy_frame.get("sessionId", "unknown")),
        role=str(legacy_frame.get("role", "camera")),
        seq=int(legacy_frame.get("frameIndex", 0) or 0),
        tMs=float(legacy_frame.get("timeMs", 0.0) or 0.0),
        unixMs=int(legacy_frame.get("unixMs", 0) or 0),
        serverUnixMs=int(server_unix_ms),
        meta=FrameMeta(depthScale=depth_scale, source=source),
        world=mp_world_to_zup(world_raw, depth_scale),
    )
    return frame.model_dump()
