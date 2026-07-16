"""Live MediaPipe -> MuJoCo Humanoid playback using per-limb inverse kinematics.

This is an alternative to play_latest_mujoco.py. Instead of reading the server's
heuristic `latest_mujoco_frame.json`, it reads the RAW MediaPipe frame
(`received_frames/latest_frame.json`) and, for each limb, solves the shoulder/hip
joint angles so the robot's upper-arm / thigh points the same direction as yours.
The elbow/knee hinges are set directly from your joint-bend angle.

Why this is more accurate than the heuristic mapping:
  * The Humanoid shoulder/hip joints live on diagonal axes, so no single human
    angle maps cleanly to one joint. IK finds the *combination* that matches your
    limb direction, which is what makes "raise arm forward" actually go forward.
  * Direction matching (not position matching) means we don't need to calibrate
    your limb lengths against the robot's.

Requirements (install on the machine running the sim):
    pip install "gymnasium[mujoco]" mujoco scipy

Run (from my-app/, with the websocket server already running):
    python play_latest_mujoco_ik.py
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import gymnasium as gym
import mujoco
from scipy.optimize import least_squares

RAW_FILE = Path("received_frames/latest_frame.json")

# MediaPipe Pose landmark indices.
NOSE = 0
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24
L_KN, R_KN = 25, 26
L_AN, R_AN = 27, 28

# --- Orientation of the (upright, +x-facing) robot torso, in world axes -------
# MuJoCo world: +x forward, +y left, +z up. The robot's own right side is -y.
# If the whole body looks mirrored or turned around, flip a sign here.
FORWARD = np.array([1.0, 0.0, 0.0])   # robot torso forward
UP = np.array([0.0, 0.0, 1.0])        # robot torso up
RIGHT = np.array([0.0, -1.0, 0.0])    # robot torso right
MP_FORWARD_SIGN = 1.0                 # flip if front/back is inverted

# Hinge sign knobs (Humanoid knees/elbows flex toward negative angles).
HINGE_SIGN = {"elbow": -1.0, "knee": -1.0}

STANDING_HEIGHT = 1.35


# ---------------------------------------------------------------------------
# Landmark helpers
# ---------------------------------------------------------------------------
def _get_points(frame: Dict) -> List:
    pts = frame.get("worldLandmarks")
    if isinstance(pts, list) and pts:
        return pts
    pts = frame.get("landmarks")
    return pts if isinstance(pts, list) else []


def _lm(points: List, idx: int, min_vis: float = 0.3) -> Optional[np.ndarray]:
    if idx < 0 or idx >= len(points):
        return None
    p = points[idx]
    if not isinstance(p, dict) or "x" not in p or "y" not in p or "z" not in p:
        return None
    vis = p.get("visibility")
    if isinstance(vis, (int, float)) and vis < min_vis:
        return None
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])], dtype=np.float64)


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def _bend_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Bend at joint b for the chain a-b-c. 0 = straight, grows as it folds."""
    u = _unit(a - b)
    w = _unit(c - b)
    if u is None or w is None:
        return 0.0
    dot = float(np.clip(np.dot(u, w), -1.0, 1.0))
    return np.pi - np.arccos(dot)


class TorsoBasis:
    """Orthonormal body frame built from the human's shoulders/hips."""

    def __init__(self, l_sh, r_sh, l_hip, r_hip):
        shoulder_mid = (l_sh + r_sh) * 0.5
        hip_mid = (l_hip + r_hip) * 0.5
        up = _unit(shoulder_mid - hip_mid)                 # hips -> shoulders
        right = _unit(r_sh - l_sh)                          # subject left -> right
        if up is None or right is None:
            raise ValueError("degenerate torso")
        right = _unit(right - np.dot(right, up) * up)       # orthogonalize
        forward = _unit(np.cross(up, right)) * MP_FORWARD_SIGN
        self.right, self.up, self.forward = right, up, forward

    def to_robot_world(self, v_mp: np.ndarray) -> Optional[np.ndarray]:
        """Express a MediaPipe-world direction in the robot's world frame."""
        comp_r = float(np.dot(v_mp, self.right))
        comp_u = float(np.dot(v_mp, self.up))
        comp_f = float(np.dot(v_mp, self.forward))
        return _unit(comp_r * RIGHT + comp_u * UP + comp_f * FORWARD)


# ---------------------------------------------------------------------------
# Per-limb IK against the real MuJoCo model
# ---------------------------------------------------------------------------
class HumanoidIK:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)  # scratch data for IK, separate from the env
        # Reset to the model's reference pose so the free-joint quaternion is valid
        # (a zero quaternion would make mj_forward produce degenerate body positions).
        mujoco.mj_resetData(model, self.data)

        def jid(name):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if i < 0:
                raise RuntimeError(
                    f"Joint '{name}' not found in this Humanoid model. "
                    f"The joint names in this MuJoCo/gymnasium version differ from "
                    f"what this script expects — send me the model's joint list."
                )
            return i

        def bid(name):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if i < 0:
                raise RuntimeError(
                    f"Body '{name}' not found in this Humanoid model. "
                    f"The body names in this MuJoCo/gymnasium version differ from "
                    f"what this script expects — send me the model's body list."
                )
            return i

        # Each entry: the multi-DOF joints to solve, the two body origins whose
        # difference gives the segment direction, and the hinge joint we set directly.
        self.limbs = {
            "right_arm": dict(
                joints=["right_shoulder1", "right_shoulder2"],
                root=bid("right_upper_arm"), tip=bid("right_lower_arm"),
                hinge="right_elbow", hinge_kind="elbow",
            ),
            "left_arm": dict(
                joints=["left_shoulder1", "left_shoulder2"],
                root=bid("left_upper_arm"), tip=bid("left_lower_arm"),
                hinge="left_elbow", hinge_kind="elbow",
            ),
            "right_leg": dict(
                joints=["right_hip_x", "right_hip_z", "right_hip_y"],
                root=bid("right_thigh"), tip=bid("right_shin"),
                hinge="right_knee", hinge_kind="knee",
            ),
            "left_leg": dict(
                joints=["left_hip_x", "left_hip_z", "left_hip_y"],
                root=bid("left_thigh"), tip=bid("left_shin"),
                hinge="left_knee", hinge_kind="knee",
            ),
        }
        for limb in self.limbs.values():
            limb["qadr"] = [model.jnt_qposadr[jid(n)] for n in limb["joints"]]
            limb["bounds"] = self._bounds([jid(n) for n in limb["joints"]])
            limb["hinge_adr"] = model.jnt_qposadr[jid(limb["hinge"])]
            limb["hinge_bounds"] = self._bounds([jid(limb["hinge"])])
            limb["prev"] = np.zeros(len(limb["joints"]))

    def _bounds(self, jids):
        lo, hi = [], []
        for jid in jids:
            if self.model.jnt_limited[jid]:
                lo.append(float(self.model.jnt_range[jid][0]))
                hi.append(float(self.model.jnt_range[jid][1]))
            else:
                lo.append(-np.pi)
                hi.append(np.pi)
        return np.array(lo), np.array(hi)

    def _segment_dir(self, qadr, angles, root_bid, tip_bid) -> Optional[np.ndarray]:
        self.data.qpos[qadr] = angles
        mujoco.mj_forward(self.model, self.data)
        return _unit(self.data.xpos[tip_bid] - self.data.xpos[root_bid])

    def solve_limb(self, limb, target_dir: np.ndarray) -> np.ndarray:
        lo, hi = limb["bounds"]

        def residual(angles):
            d = self._segment_dir(limb["qadr"], angles, limb["root"], limb["tip"])
            if d is None:
                return np.ones(3)
            return d - target_dir

        x0 = np.clip(limb["prev"], lo, hi)
        sol = least_squares(residual, x0, bounds=(lo, hi), method="trf", max_nfev=40)
        limb["prev"] = sol.x
        return sol.x


# ---------------------------------------------------------------------------
# Frame -> qpos
# ---------------------------------------------------------------------------
def build_qpos(ik: HumanoidIK, template_qpos: np.ndarray, frame: Dict):
    """Return (qpos_or_None, report). report explains what happened this frame."""
    report: Dict = {"n_points": 0, "solved": [], "skipped": {}}
    pts = _get_points(frame)
    report["n_points"] = len(pts)
    report["source"] = "worldLandmarks" if frame.get("worldLandmarks") else "landmarks"

    l_sh, r_sh = _lm(pts, L_SH), _lm(pts, R_SH)
    l_hip, r_hip = _lm(pts, L_HIP), _lm(pts, R_HIP)
    if any(p is None for p in (l_sh, r_sh, l_hip, r_hip)):
        missing = [n for n, p in (("l_sh", l_sh), ("r_sh", r_sh),
                                  ("l_hip", l_hip), ("r_hip", r_hip)) if p is None]
        report["skipped"]["_torso"] = f"missing/low-visibility: {missing}"
        return None, report

    try:
        basis = TorsoBasis(l_sh, r_sh, l_hip, r_hip)
    except ValueError:
        report["skipped"]["_torso"] = "degenerate torso geometry"
        return None, report

    qpos = template_qpos.copy()
    qpos[0:3] = np.array([0.0, 0.0, STANDING_HEIGHT])
    qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # upright, facing +x

    # Landmarks per limb: (shoulder/hip, elbow/knee, wrist/ankle).
    limb_lms = {
        "right_arm": (r_sh, _lm(pts, R_EL), _lm(pts, R_WR)),
        "left_arm": (l_sh, _lm(pts, L_EL), _lm(pts, L_WR)),
        "right_leg": (r_hip, _lm(pts, R_KN), _lm(pts, R_AN)),
        "left_leg": (l_hip, _lm(pts, L_KN), _lm(pts, L_AN)),
    }

    for name, (root_lm, mid_lm, tip_lm) in limb_lms.items():
        limb = ik.limbs[name]
        if root_lm is None or mid_lm is None:
            report["skipped"][name] = "missing shoulder/elbow (or hip/knee) landmark"
            continue

        # Target: direction of the upper segment (shoulder->elbow / hip->knee).
        seg_mp = _unit(mid_lm - root_lm)
        if seg_mp is None:
            report["skipped"][name] = "zero-length segment"
            continue
        target = basis.to_robot_world(seg_mp)
        if target is None:
            report["skipped"][name] = "zero target direction"
            continue

        angles = ik.solve_limb(limb, target)
        qpos[limb["qadr"]] = angles

        # Hinge (elbow/knee) set directly from the human bend angle.
        if tip_lm is not None:
            bend = _bend_angle(root_lm, mid_lm, tip_lm)
            val = HINGE_SIGN[limb["hinge_kind"]] * bend
            lo, hi = limb["hinge_bounds"]
            qpos[limb["hinge_adr"]] = float(np.clip(val, lo[0], hi[0]))

        report["solved"].append(name)

    return qpos, report


def main():
    env = gym.make("Humanoid-v5", render_mode="human")
    env.reset(seed=0)

    model = env.unwrapped.model
    template_qpos = np.array(env.unwrapped.data.qpos, dtype=np.float64).copy()
    template_qvel = np.array(env.unwrapped.data.qvel, dtype=np.float64).copy()

    ik = HumanoidIK(model)

    # Stand upright to start.
    start = template_qpos.copy()
    start[0:3] = np.array([0.0, 0.0, STANDING_HEIGHT])
    start[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    env.unwrapped.set_state(start, template_qvel.copy())

    print(f"Watching {RAW_FILE} (IK retargeting)")
    if not RAW_FILE.exists():
        print(f"  NOTE: {RAW_FILE} does not exist yet. Start the websocket server and "
              f"begin RECORDING in the browser (frames are only sent while recording).")
    # Apply the existing file immediately so you see motion without waiting for a new frame.
    last_mtime = 0.0
    warned_no_move = False

    try:
        while True:
            if RAW_FILE.exists():
                mtime = RAW_FILE.stat().st_mtime
                if mtime != last_mtime:
                    last_mtime = mtime
                    try:
                        raw = RAW_FILE.read_text(encoding="utf-8")
                        if not raw.strip():
                            continue  # caught mid-write; try again next tick
                        frame = json.loads(raw)
                        qpos, report = build_qpos(ik, template_qpos, frame)
                        if qpos is not None:
                            env.unwrapped.set_state(qpos, template_qvel.copy())
                            print(f"frame {frame.get('frameIndex')}: "
                                  f"solved {report['solved']} "
                                  f"({report['source']}, {report['n_points']} pts)"
                                  + (f" skipped {report['skipped']}" if report['skipped'] else ""))
                            if not report["solved"] and not warned_no_move:
                                warned_no_move = True
                                print("  -> No limbs solved, so the robot won't move. "
                                      "See 'skipped' above for why (usually missing "
                                      "worldLandmarks or low visibility).")
                        else:
                            print(f"frame {frame.get('frameIndex')}: SKIPPED "
                                  f"({report['source']}, {report['n_points']} pts) "
                                  f"-> {report['skipped']}")
                    except json.JSONDecodeError:
                        continue  # partial file (should be rare with atomic writes)
                    except Exception as e:
                        print("IK playback error:", repr(e))

            env.render()
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
