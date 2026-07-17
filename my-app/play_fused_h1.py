"""Live 3-camera fusion -> H1 in MuJoCo, WITH torso turning.

Reads all three camera streams (received_frames/latest_frame_<role>.json), fuses
them each loop using the saved body calibration, drives the H1 limbs from the
fused 3D, AND rotates the robot's root to match your facing direction -- so when
you turn, the H1 turns.

Prereqs:
  - the 3 phones streaming (server running + tunnel + phones foregrounded)
  - received_frames/calibration_body.json  (run body_fuse_3d.py once)

ROUGH by design: the body-based fusion is ~CV 0.22, so the turn is smoothed
heavily and will lag / jitter a bit. Tuning knobs are at the top.

Run: python play_fused_h1.py
"""

import json
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

import play_latest_h1_mujoco_ik as P
P.DEPTH_SCALE = 1.0  # fused 3D has real depth

RF = Path("received_frames")
VIS_JOINT = 0.5
# --- turn tuning: flip sign if the robot turns the WRONG way; add offset to
#     recenter "facing the front camera" to 0; smaller smooth = snappier/jerkier.
YAW_SIGN = 1.0
YAW_OFFSET_DEG = 0.0
YAW_SMOOTH = 0.15    # EMA factor (0..1); low = heavy smoothing


def _read(role):
    p = RF / f"latest_frame_{role}.json"
    if not p.exists():
        return None
    try:
        fr = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    wl = fr.get("worldLandmarks")
    if not (isinstance(wl, list) and len(wl) >= 33):
        return None
    Pt = np.array([[q.get("x", 0), q.get("y", 0), q.get("z", 0)]
                   for q in wl[:33]], float)
    V = np.array([q.get("visibility", 0) for q in wl[:33]], float)
    mt = p.stat().st_mtime
    return Pt, V, mt


def _center(Pt, w):
    W = w / w.sum()
    return Pt - (Pt * W[:, None]).sum(0)


def fuse(rots):
    """Fuse whatever camera streams are currently fresh into one 3D skeleton."""
    Ps, Ws = [], []
    for role, R in rots.items():
        s = _read(role)
        if s is None:
            continue
        Pt, V, _ = s
        Ps.append((R @ _center(Pt, V).T).T)
        Ws.append(V)
    if not Ps:
        return None
    Ps = np.array(Ps)
    Wv = np.array(Ws)
    gated = np.where(Wv >= VIS_JOINT, Wv, 0.0)
    gated = np.where(gated.sum(0)[None, :] < 1e-6, Wv, gated)
    w = gated[:, :, None]
    return (Ps * w).sum(0) / np.clip(w.sum(0), 1e-6, None)


def facing_yaw(P3):
    """Facing direction (radians) from the fused torso, robust shoulder azimuth."""
    s = P3[12] - P3[11]                 # right - left shoulder
    # horizontal plane is x-z (MediaPipe y is vertical/down)
    az = np.arctan2(s[2], s[0])         # shoulder-line azimuth
    return az + np.pi / 2               # facing is perpendicular to it


def yaw_to_quat(yaw):
    """Root quaternion: rotation of `yaw` about the robot's vertical (z) axis."""
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def main():
    calib = json.loads((RF / "calibration_body.json").read_text())
    rots = {r: np.array(m) for r, m in calib["rotations"].items()}
    print(f"cameras: {list(rots)}  (ref={calib['ref']})")

    model = mujoco.MjModel.from_xml_path(P._h1_model_path())
    P._set_background_gray(model, P.BACKGROUND_GRAY)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    tmpl = np.array(data.qpos, float).copy()
    tmpl[0:3] = [0, 0, P.STANDING_HEIGHT]
    tmpl[3:7] = [1, 0, 0, 0]
    ik = P.H1IK(model)
    data.qpos[:] = tmpl
    mujoco.mj_forward(model, data)

    yaw_ema = None
    print("Live fused playback + torso turn. Turn your body -> H1 turns.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            P3 = fuse(rots)
            if P3 is not None:
                frame = {"worldLandmarks": [
                    {"x": float(P3[j, 0]), "y": float(P3[j, 1]),
                     "z": float(P3[j, 2]), "visibility": 1.0} for j in range(33)]}
                qpos, rep = P.build_qpos(ik, tmpl, frame)
                if qpos is not None:
                    # facing yaw -> smooth -> root rotation (unfreeze the turn)
                    raw = YAW_SIGN * facing_yaw(P3) + np.radians(YAW_OFFSET_DEG)
                    if yaw_ema is None:
                        yaw_ema = raw
                    else:
                        # smooth on the circle (handle wrap)
                        d = np.arctan2(np.sin(raw - yaw_ema), np.cos(raw - yaw_ema))
                        yaw_ema += YAW_SMOOTH * d
                    qpos[3:7] = yaw_to_quat(yaw_ema)
                    data.qpos[:] = qpos
                    mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1 / 30)


if __name__ == "__main__":
    main()
