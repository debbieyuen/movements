"""Remap MediaPipe pose JSON frames to a Unitree H1-friendly motion file.

Input shape expected (from CameraRecorder / QuestRecorder):
[
  {
    "frameIndex": 0,
    "t": 30356,
    "timeMs": 3812,
    "unixMs": 1781707688821,
    "landmarks": [ {"x":..., "y":..., "z":..., "visibility":...}, ... ],
    "worldLandmarks": [ {"x":..., "y":..., "z":..., "visibility":...}, ... ]
  },
  ...
]

Output shape:
{
  "sessionId": "...",
  "source": "mediapipe_pose",
  "target": "unitree_h1",
  "frames": [
    {
      "frameIndex": 0,
      "timeMs": 3812,
      "unixMs": 1781707688821,
      "root_pose": [x, y, z, qx, qy, qz, qw],
      "h1_qpos": [...],
      "h1_qvel": [...],
      "debug": {
        "pelvis": [...],
        "chest": [...],
        ...
      }
    }
  ]
}

Notes:
- This is a *starter* remapper. It uses MediaPipe world landmarks for a rough H1 pose.
- It is intended for visualization / initial simulation, not final precise robot control.
- H1 joint ordering in the output is configurable; keep this file and adjust the index order
  to match your ManiSkill / H1 wrapper if needed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# MediaPipe Pose landmark indices
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

# --- H1 joint vector layout -------------------------------------------------
# This file produces a generic H1 qpos vector with the following layout.
# If your ManiSkill / H1 asset uses a different order, only edit H1_JOINT_NAMES.
H1_JOINT_NAMES: List[str] = [
    # floating base (7)
    "base_x",
    "base_y",
    "base_z",
    "base_qx",
    "base_qy",
    "base_qz",
    "base_qw",
    # upper body
    "torso_yaw",
    "torso_pitch",
    "torso_roll",
    "neck_yaw",
    "neck_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    # lower body
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
]


@dataclass
class FrameData:
    frameIndex: int
    timeMs: float
    unixMs: int
    root_pose: List[float]
    h1_qpos: List[float]
    h1_qvel: List[float]
    debug: Dict[str, Any]


def _as_np(v: Sequence[float]) -> np.ndarray:
    return np.asarray(v, dtype=np.float64)


def _norm(v: np.ndarray, eps: float = 1e-9) -> float:
    return float(np.linalg.norm(v) + eps)


def _unit(v: np.ndarray) -> np.ndarray:
    n = _norm(v)
    return v / n


def _safe_get_landmark(points: Sequence[Dict[str, Any]], idx: int) -> Optional[np.ndarray]:
    if idx < 0 or idx >= len(points):
        return None
    p = points[idx]
    if p is None:
        return None
    if any(k not in p for k in ("x", "y", "z")):
        return None
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])], dtype=np.float64)


def _midpoint(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if a is None or b is None:
        return None
    return (a + b) * 0.5


def _quat_from_axes(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> List[float]:
    """Convert orthonormal axes to quaternion [qx, qy, qz, qw]."""
    # Rotation matrix with axes as columns
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    # Robust matrix-to-quaternion conversion
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-12)) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-12)) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-12)) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q = q / np.linalg.norm(q)
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def _quat_to_rotvec(q: Sequence[float]) -> np.ndarray:
    qx, qy, qz, qw = q
    v = np.array([qx, qy, qz], dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-10:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(n, qw)
    axis = v / n
    return axis * angle


class H1PoseRemapper:
    def __init__(self) -> None:
        self.prev_qpos: Optional[np.ndarray] = None
        self.prev_time_ms: Optional[float] = None

    def _build_body_frame(self, lm: Sequence[Dict[str, Any]]) -> Dict[str, Optional[np.ndarray]]:
        ls = _safe_get_landmark(lm, LEFT_SHOULDER)
        rs = _safe_get_landmark(lm, RIGHT_SHOULDER)
        lh = _safe_get_landmark(lm, LEFT_HIP)
        rh = _safe_get_landmark(lm, RIGHT_HIP)
        le = _safe_get_landmark(lm, LEFT_ELBOW)
        re = _safe_get_landmark(lm, RIGHT_ELBOW)
        lw = _safe_get_landmark(lm, LEFT_WRIST)
        rw = _safe_get_landmark(lm, RIGHT_WRIST)
        lk = _safe_get_landmark(lm, LEFT_KNEE)
        rk = _safe_get_landmark(lm, RIGHT_KNEE)
        la = _safe_get_landmark(lm, LEFT_ANKLE)
        ra = _safe_get_landmark(lm, RIGHT_ANKLE)
        nose = _safe_get_landmark(lm, NOSE)

        pelvis = _midpoint(lh, rh)
        chest = _midpoint(ls, rs)
        spine = _midpoint(pelvis, chest)
        head = nose if nose is not None else chest

        return {
            "pelvis": pelvis,
            "chest": chest,
            "spine": spine,
            "head": head,
            "ls": ls,
            "rs": rs,
            "lh": lh,
            "rh": rh,
            "le": le,
            "re": re,
            "lw": lw,
            "rw": rw,
            "lk": lk,
            "rk": rk,
            "la": la,
            "ra": ra,
        }

    def _estimate_root_pose(self, body: Dict[str, Optional[np.ndarray]]) -> Tuple[List[float], np.ndarray]:
        pelvis = body["pelvis"]
        chest = body["chest"]
        ls = body["ls"]
        rs = body["rs"]
        lh = body["lh"]
        rh = body["rh"]

        if pelvis is None:
            pelvis = np.zeros(3, dtype=np.float64)
        if chest is None:
            chest = pelvis + np.array([0.0, 0.25, 0.0], dtype=np.float64)
        if ls is None or rs is None:
            left = np.array([-0.15, 0.0, 0.0], dtype=np.float64)
            right = np.array([0.15, 0.0, 0.0], dtype=np.float64)
            ls = chest + left
            rs = chest + right

        x_axis = _unit(rs - ls)  # left -> right
        y_axis = _unit(chest - pelvis)  # pelvis -> chest
        z_axis = _unit(np.cross(x_axis, y_axis))
        y_axis = _unit(np.cross(z_axis, x_axis))  # re-orthogonalize

        q = _quat_from_axes(x_axis, y_axis, z_axis)
        return q, pelvis

    def _joint_angle_between(self, a: np.ndarray, b: np.ndarray) -> float:
        """Returns a signed-ish bend scalar in radians for a 2-bone chain.

        This is intentionally simple. It is enough to create a live pose that roughly tracks
        the human motion before you build a better IK/optimization layer.
        """
        ua = _unit(a)
        ub = _unit(b)
        dot = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
        return math.acos(dot)

    def remap_frame(self, frame: Dict[str, Any]) -> FrameData:
        lm = frame.get("worldLandmarks") or frame.get("landmarks") or []
        body = self._build_body_frame(lm)

        root_q, root_p = self._estimate_root_pose(body)

        # Build some simple local direction vectors from the world landmarks.
        ls, rs = body["ls"], body["rs"]
        le, re = body["le"], body["re"]
        lw, rw = body["lw"], body["rw"]
        lh, rh = body["lh"], body["rh"]
        lk, rk = body["lk"], body["rk"]
        la, ra = body["la"], body["ra"]
        pelvis = body["pelvis"] if body["pelvis"] is not None else root_p
        chest = body["chest"] if body["chest"] is not None else pelvis + np.array([0.0, 0.25, 0.0])
        head = body["head"] if body["head"] is not None else chest + np.array([0.0, 0.2, 0.0])

        # Default directions if any joints are missing.
        if ls is None:
            ls = chest + np.array([-0.15, 0.0, 0.0])
        if rs is None:
            rs = chest + np.array([0.15, 0.0, 0.0])
        if lh is None:
            lh = pelvis + np.array([-0.1, -0.1, 0.0])
        if rh is None:
            rh = pelvis + np.array([0.1, -0.1, 0.0])
        if le is None:
            le = ls + np.array([-0.12, -0.1, 0.0])
        if re is None:
            re = rs + np.array([0.12, -0.1, 0.0])
        if lw is None:
            lw = le + np.array([-0.10, -0.10, 0.0])
        if rw is None:
            rw = re + np.array([0.10, -0.10, 0.0])
        if lk is None:
            lk = lh + np.array([0.0, -0.25, 0.0])
        if rk is None:
            rk = rh + np.array([0.0, -0.25, 0.0])
        if la is None:
            la = lk + np.array([0.0, -0.28, 0.05])
        if ra is None:
            ra = rk + np.array([0.0, -0.28, -0.05])

        # Compute simple joint bend scalars.
        left_elbow = self._joint_angle_between(ls - le, lw - le)
        right_elbow = self._joint_angle_between(rs - re, rw - re)
        left_knee = self._joint_angle_between(lh - lk, la - lk)
        right_knee = self._joint_angle_between(rh - rk, ra - rk)

        # Upper body orientation heuristics.
        torso_yaw = math.atan2((rs - ls)[2], (rs - ls)[0] + 1e-8)
        torso_pitch = math.atan2((chest - pelvis)[1], abs((chest - pelvis)[0]) + abs((chest - pelvis)[2]) + 1e-8)
        torso_roll = math.atan2((rs - ls)[1], abs((rs - ls)[0]) + 1e-8)

        neck_yaw = 0.0
        neck_pitch = math.atan2((head - chest)[1], abs((head - chest)[0]) + abs((head - chest)[2]) + 1e-8)

        # Shoulder yaw/pitch/roll are rough proxies.
        left_shoulder_vec = le - ls
        right_shoulder_vec = re - rs
        left_shoulder_pitch = math.atan2(-left_shoulder_vec[1], abs(left_shoulder_vec[0]) + abs(left_shoulder_vec[2]) + 1e-8)
        right_shoulder_pitch = math.atan2(-right_shoulder_vec[1], abs(right_shoulder_vec[0]) + abs(right_shoulder_vec[2]) + 1e-8)
        left_shoulder_roll = math.atan2(left_shoulder_vec[2], abs(left_shoulder_vec[0]) + 1e-8)
        right_shoulder_roll = math.atan2(right_shoulder_vec[2], abs(right_shoulder_vec[0]) + 1e-8)
        left_shoulder_yaw = math.atan2(left_shoulder_vec[0], abs(left_shoulder_vec[2]) + 1e-8)
        right_shoulder_yaw = math.atan2(right_shoulder_vec[0], abs(right_shoulder_vec[2]) + 1e-8)

        # Hips.
        left_hip_yaw = 0.0
        right_hip_yaw = 0.0
        left_hip_roll = math.atan2((lk - lh)[2], abs((lk - lh)[1]) + 1e-8)
        right_hip_roll = math.atan2((rk - rh)[2], abs((rk - rh)[1]) + 1e-8)
        left_hip_pitch = math.atan2(-(lk - lh)[1], abs((lk - lh)[0]) + abs((lk - lh)[2]) + 1e-8)
        right_hip_pitch = math.atan2(-(rk - rh)[1], abs((rk - rh)[0]) + abs((rk - rh)[2]) + 1e-8)

        left_ankle_pitch = math.atan2(-(la - lk)[1], abs((la - lk)[0]) + abs((la - lk)[2]) + 1e-8)
        left_ankle_roll = math.atan2((la - lk)[2], abs((la - lk)[0]) + 1e-8)
        right_ankle_pitch = math.atan2(-(ra - rk)[1], abs((ra - rk)[0]) + abs((ra - rk)[2]) + 1e-8)
        right_ankle_roll = math.atan2((ra - rk)[2], abs((ra - rk)[0]) + 1e-8)

        # Compose a qpos vector using the chosen layout.
        qpos = np.zeros(len(H1_JOINT_NAMES), dtype=np.float64)
        qpos[0:3] = root_p
        qpos[3:7] = np.array(root_q, dtype=np.float64)
        qpos[7] = torso_yaw
        qpos[8] = torso_pitch
        qpos[9] = torso_roll
        qpos[10] = neck_yaw
        qpos[11] = neck_pitch
        qpos[12] = left_shoulder_pitch
        qpos[13] = left_shoulder_roll
        qpos[14] = left_shoulder_yaw
        qpos[15] = left_elbow
        qpos[16] = right_shoulder_pitch
        qpos[17] = right_shoulder_roll
        qpos[18] = right_shoulder_yaw
        qpos[19] = right_elbow
        qpos[20] = left_hip_yaw
        qpos[21] = left_hip_roll
        qpos[22] = left_hip_pitch
        qpos[23] = left_knee
        qpos[24] = left_ankle_pitch
        qpos[25] = left_ankle_roll
        qpos[26] = right_hip_yaw
        qpos[27] = right_hip_roll
        qpos[28] = right_hip_pitch
        qpos[29] = right_knee
        qpos[30] = right_ankle_pitch
        qpos[31] = right_ankle_roll

        # Simple qvel from finite difference.
        unix_ms = int(frame.get("unixMs", int(time.time() * 1000)))
        time_ms = float(frame.get("timeMs", frame.get("t", 0.0)))

        if self.prev_qpos is None or self.prev_time_ms is None:
            qvel = np.zeros_like(qpos)
        else:
            dt = max((time_ms - self.prev_time_ms) / 1000.0, 1e-3)
            qvel = (qpos - self.prev_qpos) / dt

        self.prev_qpos = qpos.copy()
        self.prev_time_ms = time_ms

        debug = {
            "pelvis": pelvis.tolist(),
            "chest": chest.tolist(),
            "head": head.tolist(),
            "landmark_count": len(lm),
            "joint_names": H1_JOINT_NAMES,
        }

        return FrameData(
            frameIndex=int(frame.get("frameIndex", 0)),
            timeMs=time_ms,
            unixMs=unix_ms,
            root_pose=[float(x) for x in np.concatenate([root_p, np.array(root_q)])],
            h1_qpos=[float(x) for x in qpos],
            h1_qvel=[float(x) for x in qvel],
            debug=debug,
        )


def remap_json(input_path: Path, output_path: Path, session_id: Optional[str] = None) -> Dict[str, Any]:
    raw = json.loads(input_path.read_text())
    if isinstance(raw, dict):
        frames = raw.get("frames", raw.get("poseFrames", []))
    else:
        frames = raw

    remapper = H1PoseRemapper()
    out_frames: List[Dict[str, Any]] = []

    for frame in frames:
        mapped = remapper.remap_frame(frame)
        out_frames.append(
            {
                "frameIndex": mapped.frameIndex,
                "timeMs": mapped.timeMs,
                "unixMs": mapped.unixMs,
                "root_pose": mapped.root_pose,
                "h1_qpos": mapped.h1_qpos,
                "h1_qvel": mapped.h1_qvel,
                "debug": mapped.debug,
            }
        )

    out = {
        "sessionId": session_id or input_path.stem,
        "source": "mediapipe_pose",
        "target": "unitree_h1",
        "jointNames": H1_JOINT_NAMES,
        "frames": out_frames,
    }

    output_path.write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Remap MediaPipe pose JSON to Unitree H1 motion JSON")
    parser.add_argument("input", type=Path, help="Input pose JSON from the browser recorder")
    parser.add_argument("output", type=Path, help="Output H1 motion JSON")
    parser.add_argument("--session-id", type=str, default=None, help="Optional session id override")
    args = parser.parse_args()

    try:
        remap_json(args.input, args.output, session_id=args.session_id)
    except Exception as e:
        print(f"Error while remapping: {e}")
        raise

    print(f"Wrote H1 remapped motion to {args.output.resolve()}")


if __name__ == "__main__":
    main()
