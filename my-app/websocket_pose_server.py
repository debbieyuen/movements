import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import websockets

from live_h1_remapper import pose_to_h1

HOST = "0.0.0.0"
PORT = 8765

OUT_DIR = Path("received_frames")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LATEST_FILE = OUT_DIR / "latest_frame.json"
JSONL_FILE = OUT_DIR / "frames.jsonl"
LATEST_H1_FILE = OUT_DIR / "latest_h1_frame.json"
LATEST_MUJOCO_FILE = OUT_DIR / "latest_mujoco_frame.json"


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON via a temp file + atomic rename.

    Consumers (the playback scripts) poll these files at high frequency. A plain
    write_text truncates then fills the file, so a reader can catch it empty and
    hit "Expecting value: line 1 column 1". os.replace (Path.replace) swaps the
    finished file in atomically, so readers only ever see a complete document.
    """
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    # On Windows, os.replace raises PermissionError ("Access is denied") if a
    # reader (the playback scripts polling these files) happens to have the
    # destination open at that instant. It is transient, so retry briefly
    # before giving up. Never let it propagate: a single failed swap must not
    # tear down the client connection.
    last_err: Optional[Exception] = None
    for _ in range(40):
        try:
            tmp.replace(path)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.005)
    print(f"  (atomic write to {path.name} kept failing: {last_err!r}; "
          f"skipped this frame, next one will refresh it)")

# MuJoCo Gymnasium Humanoid-v5 actuated-joint order (qpos[7:24], 17 DOFs).
# This MUST match the model's joint order, because play_latest_mujoco.py writes
# these values positionally into qpos[7:24]. The Humanoid has NO ankle joints
# and TWO DOFs per shoulder (shoulder1 + shoulder2).
MUJOCO_QPOS_NAMES = [
    "abdomen_y",
    "abdomen_z",
    "abdomen_x",
    "right_hip_x",
    "right_hip_z",
    "right_hip_y",
    "right_knee",
    "left_hip_x",
    "left_hip_z",
    "left_hip_y",
    "left_knee",
    "right_shoulder1",
    "right_shoulder2",
    "right_elbow",
    "left_shoulder1",
    "left_shoulder2",
    "left_elbow",
]

MUJOCO_QVEL_NAMES = MUJOCO_QPOS_NAMES[:]

# --- Retargeting tuning knobs ------------------------------------------------
# The Humanoid's shoulder1/shoulder2 joints are on DIAGONAL axes and the knee/
# elbow flex toward NEGATIVE angles, so a naive mapping can drive a joint the
# wrong way (e.g. "lift arm -> arm goes backward"). Watch the MuJoCo window and
# flip any joint that moves opposite your body by changing its sign (1.0 -> -1.0).
MJ_SIGN = {
    "shoulder1": -1.0,  # arm raise (front/back). Flip if arms swing backward.
    "shoulder2": 1.0,   # arm spread (in/out). Flip if arms cross the wrong way.
    "elbow": -1.0,      # elbow bend. Flip if the forearm bends the wrong way.
    "knee": -1.0,       # knee bend (Humanoid knees flex negative).
    "hip_pitch": 1.0,   # leg forward/back.
}

_LAST_MUJOCO_QPOS: Optional[List[float]] = None
_LAST_MUJOCO_TS_MS: Optional[int] = None


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _as_point(p: Any) -> Optional[List[float]]:
    if not isinstance(p, dict):
        return None
    if "x" not in p or "y" not in p or "z" not in p:
        return None
    try:
        return [float(p["x"]), float(p["y"]), float(p["z"])]
    except Exception:
        return None


def _get_landmarks(frame: Dict[str, Any]) -> List[Any]:
    pts = frame.get("worldLandmarks")
    if isinstance(pts, list) and pts:
        return pts
    pts = frame.get("landmarks")
    if isinstance(pts, list) and pts:
        return pts
    return []


def _lm(points: Sequence[Any], idx: int) -> Optional[List[float]]:
    if idx < 0 or idx >= len(points):
        return None
    return _as_point(points[idx])


def _mid(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[List[float]]:
    if a is None or b is None:
        return None
    return [0.5 * (float(a[0]) + float(b[0])), 0.5 * (float(a[1]) + float(b[1])), 0.5 * (float(a[2]) + float(b[2]))]


def _vec(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [float(b[0] - a[0]), float(b[1] - a[1]), float(b[2] - a[2])]


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(float(v[0]) * float(v[0]) + float(v[1]) * float(v[1]) + float(v[2]) * float(v[2]))


def _unit(v: Sequence[float]) -> List[float]:
    n = _norm(v)
    if n < 1e-9:
        return [0.0, 0.0, 0.0]
    return [float(v[0]) / n, float(v[1]) / n, float(v[2]) / n]


def _angle_at(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    ab = _unit(_vec(b, a))
    cb = _unit(_vec(b, c))
    dot = ab[0] * cb[0] + ab[1] * cb[1] + ab[2] * cb[2]
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)


def _safe_axis(v: Sequence[float]) -> List[float]:
    u = _unit(v)
    if u == [0.0, 0.0, 0.0]:
        return [0.0, 1.0, 0.0]
    return u


def pose_to_mujoco(server_frame: Dict[str, Any]) -> Dict[str, Any]:
    points = _get_landmarks(server_frame)

    # MediaPipe Pose landmark indices.
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    ls = _lm(points, LEFT_SHOULDER)
    rs = _lm(points, RIGHT_SHOULDER)
    le = _lm(points, LEFT_ELBOW)
    re = _lm(points, RIGHT_ELBOW)
    lw = _lm(points, LEFT_WRIST)
    rw = _lm(points, RIGHT_WRIST)
    lh = _lm(points, LEFT_HIP)
    rh = _lm(points, RIGHT_HIP)
    lk = _lm(points, LEFT_KNEE)
    rk = _lm(points, RIGHT_KNEE)
    la = _lm(points, LEFT_ANKLE)
    ra = _lm(points, RIGHT_ANKLE)
    nose = _lm(points, NOSE)

    pelvis = _mid(lh, rh)
    shoulders = _mid(ls, rs)

    if pelvis is None:
        pelvis = [0.0, 0.0, 0.0]
    if shoulders is None:
        shoulders = [0.0, 0.35, 0.0]

    # We keep the root upright and stable for MuJoCo playback.
    root_pose = [0.0, 0.0, 1.35, 1.0, 0.0, 0.0, 0.0]

    # Small torso motion from shoulder/hip geometry.
    torso_yaw = 0.0
    torso_pitch = 0.0
    torso_roll = 0.0
    if ls is not None and rs is not None and lh is not None and rh is not None:
        shoulder_line = _vec(ls, rs)
        hip_line = _vec(lh, rh)
        torso_roll = _clamp(0.8 * (shoulder_line[1] + hip_line[1]), -0.45, 0.45)
        torso_yaw = _clamp(0.8 * (shoulder_line[0] + hip_line[0]), -0.35, 0.35)
        if nose is not None:
            torso_pitch = _clamp(0.6 * (shoulders[1] - nose[1]), -0.5, 0.5)

    # Arms: main motion comes from hand height and elbow bend.
    left_raise = 0.0
    right_raise = 0.0
    left_out = 0.0
    right_out = 0.0
    left_elbow_bend = 0.0
    right_elbow_bend = 0.0

    if ls is not None and le is not None and lw is not None:
        # In image/world landmarks, smaller y usually means higher in frame.
        left_raise = _clamp((ls[1] - lw[1]) * 3.2, -1.5, 1.5)
        left_out = _clamp((lw[0] - ls[0]) * 2.5, -1.2, 1.2)
        left_elbow_bend = _clamp(math.pi - _angle_at(ls, le, lw), 0.0, 2.2)
    if rs is not None and re is not None and rw is not None:
        right_raise = _clamp((rs[1] - rw[1]) * 3.2, -1.5, 1.5)
        right_out = _clamp((rw[0] - rs[0]) * 2.5, -1.2, 1.2)
        right_elbow_bend = _clamp(math.pi - _angle_at(rs, re, rw), 0.0, 2.2)

    left_shoulder_pitch = _clamp(left_raise, -1.6, 1.6)
    right_shoulder_pitch = _clamp(right_raise, -1.6, 1.6)
    left_shoulder_roll = _clamp(-0.55 * left_out, -1.2, 1.2)
    right_shoulder_roll = _clamp(0.55 * right_out, -1.2, 1.2)
    left_shoulder_yaw = _clamp(0.2 * left_out, -0.8, 0.8)
    right_shoulder_yaw = _clamp(-0.2 * right_out, -0.8, 0.8)
    left_elbow = left_elbow_bend
    right_elbow = right_elbow_bend

    # Legs: keep mostly neutral with slight bending from the pose.
    left_hip_yaw = 0.0
    left_hip_roll = 0.0
    left_hip_pitch = 0.0
    left_knee = 0.0
    left_ankle_pitch = 0.0
    left_ankle_roll = 0.0
    right_hip_yaw = 0.0
    right_hip_roll = 0.0
    right_hip_pitch = 0.0
    right_knee = 0.0
    right_ankle_pitch = 0.0
    right_ankle_roll = 0.0

    if lh is not None and lk is not None and la is not None:
        left_knee = _clamp(math.pi - _angle_at(lh, lk, la), 0.0, 2.2)
        left_hip_pitch = _clamp(-0.35 * left_knee, -0.8, 0.8)
        left_ankle_pitch = _clamp(-0.25 * left_knee, -0.8, 0.8)
    if rh is not None and rk is not None and ra is not None:
        right_knee = _clamp(math.pi - _angle_at(rh, rk, ra), 0.0, 2.2)
        right_hip_pitch = _clamp(-0.35 * right_knee, -0.8, 0.8)
        right_ankle_pitch = _clamp(-0.25 * right_knee, -0.8, 0.8)

    # Emit in Humanoid-v5 joint order (see MUJOCO_QPOS_NAMES). The MuJoCo humanoid
    # has no ankle joints, so ankle values are dropped; each shoulder takes two DOFs
    # (pitch -> shoulder1, roll -> shoulder2). Per-joint direction is corrected via
    # MJ_SIGN so you can flip any joint that moves the wrong way in the viewer.
    mujoco_qpos = [
        torso_pitch,                               # abdomen_y (forward/back bend)
        torso_yaw,                                 # abdomen_z
        torso_roll,                                # abdomen_x (side lean)
        right_hip_roll,                            # right_hip_x
        right_hip_yaw,                             # right_hip_z
        MJ_SIGN["hip_pitch"] * right_hip_pitch,    # right_hip_y
        MJ_SIGN["knee"] * right_knee,              # right_knee
        left_hip_roll,                             # left_hip_x
        left_hip_yaw,                              # left_hip_z
        MJ_SIGN["hip_pitch"] * left_hip_pitch,     # left_hip_y
        MJ_SIGN["knee"] * left_knee,               # left_knee
        MJ_SIGN["shoulder1"] * right_shoulder_pitch,  # right_shoulder1
        MJ_SIGN["shoulder2"] * right_shoulder_roll,   # right_shoulder2
        MJ_SIGN["elbow"] * right_elbow,               # right_elbow
        MJ_SIGN["shoulder1"] * left_shoulder_pitch,   # left_shoulder1
        MJ_SIGN["shoulder2"] * left_shoulder_roll,    # left_shoulder2
        MJ_SIGN["elbow"] * left_elbow,                # left_elbow
    ]

    global _LAST_MUJOCO_QPOS, _LAST_MUJOCO_TS_MS
    now_ms = int(server_frame.get("serverUnixMs", int(time.time() * 1000)))
    mujoco_qvel = [0.0 for _ in mujoco_qpos]
    if _LAST_MUJOCO_QPOS is not None and _LAST_MUJOCO_TS_MS is not None:
        dt = max((now_ms - _LAST_MUJOCO_TS_MS) / 1000.0, 1.0 / 120.0)
        mujoco_qvel = [float((cur - prev) / dt) for cur, prev in zip(mujoco_qpos, _LAST_MUJOCO_QPOS)]
    _LAST_MUJOCO_QPOS = mujoco_qpos[:]
    _LAST_MUJOCO_TS_MS = now_ms

    return {
        "frameIndex": server_frame.get("frameIndex"),
        "timeMs": server_frame.get("timeMs"),
        "unixMs": server_frame.get("unixMs"),
        "serverUnixMs": server_frame.get("serverUnixMs"),
        "root_pose": root_pose,
        "mujoco_joint_names": MUJOCO_QPOS_NAMES,
        "mujoco_qpos": mujoco_qpos,
        "mujoco_qvel": mujoco_qvel,
        "debug": {
            "pelvis": pelvis,
            "shoulders": shoulders,
            "joint_names": MUJOCO_QPOS_NAMES,
        },
    }


async def handle_client(websocket):
    print("Client connected")

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"ok": False, "error": "invalid json"}))
                continue

            server_frame = {
                "serverUnixMs": int(time.time() * 1000),
                **data,
            }
            frame_index = server_frame.get("frameIndex", "?")

            # Any failure while processing ONE frame (a file lock, a bad
            # landmark, a converter edge case) must never break the socket:
            # if it did, the browser would stop streaming and the robot would
            # freeze on the last frame. Log it and keep the connection alive.
            try:
                # Save browser pose frame (atomic so readers never see a half-write).
                _atomic_write_json(LATEST_FILE, server_frame)

                with JSONL_FILE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(server_frame) + "\n")

                # Convert to H1 live frame.
                h1_frame = pose_to_h1(server_frame)
                _atomic_write_json(LATEST_H1_FILE, h1_frame)

                # Convert directly to a MuJoCo-friendly frame.
                mujoco_frame = pose_to_mujoco(server_frame)
                _atomic_write_json(LATEST_MUJOCO_FILE, mujoco_frame)
            except Exception as e:  # noqa: BLE001 - keep the stream alive no matter what
                print(f"  (frame {frame_index} processing error, skipped: {e!r})")

            role = server_frame.get("role", "?")
            time_ms = server_frame.get("timeMs", "?")
            print(f"Received frame {frame_index} from {role} at {time_ms} ms")

            await websocket.send(
                json.dumps(
                    {
                        "ok": True,
                        "saved": True,
                        "frameIndex": frame_index,
                    }
                )
            )

    except websockets.ConnectionClosed:
        print("Client disconnected")


async def main():
    print(f"WebSocket server listening on ws://{HOST}:{PORT}")
    async with websockets.serve(handle_client, HOST, PORT, max_size=20 * 1024 * 1024):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
