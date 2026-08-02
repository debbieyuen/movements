"""Live MediaPipe -> Unitree H1 playback in MuJoCo using per-limb inverse kinematics.

This is the H1 counterpart to play_latest_mujoco_ik.py. It loads the Unitree H1
MuJoCo model (from MuJoCo Menagerie via the robot_descriptions package) and, for
each limb, solves the shoulder/hip joints so the robot's upper-arm / thigh points
the same direction as yours. Elbow/knee are set from your joint-bend angle.

Why MuJoCo and not ManiSkill/Isaac: ManiSkill needs Linux + an NVIDIA GPU and
Python <= 3.12 (no SAPIEN wheels for 3.14), so it can't run on this machine. The
H1 kinematics are identical whichever engine renders them, so this gives you live
H1 retargeting today; a ManiSkill version can be ported later on a Linux/GPU box.

Requirements (already present on this machine):
    pip install mujoco scipy robot_descriptions

Run (from my-app/, with the websocket server running and the browser recording):
    python play_latest_h1_mujoco_ik.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import mujoco
import mujoco.viewer
from scipy.optimize import least_squares

# The shared protocol package lives at the repo root (one level up).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.protocol import COORD_ZUP_XFWD  # noqa: E402

# Canonical v2 live frame written by the websocket server: z-up, depth already
# compressed at the server boundary (meta.depthScale), landmarks as
# [x, y, z, visibility] arrays. See server/protocol.py for the convention.
RAW_FILE = Path("live/latest_pose.json")

# MediaPipe Pose landmark indices.
NOSE = 0
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24
L_KN, R_KN = 25, 26
L_AN, R_AN = 27, 28

# --- Orientation of the (upright) H1 torso, in world axes ---------------------
# MuJoCo world: +x forward, +y left, +z up. The robot's own right side is -y.
FORWARD = np.array([1.0, 0.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])
RIGHT = np.array([0.0, -1.0, 0.0])

STANDING_HEIGHT = 1.0  # pelvis height for kinematic playback

# We solve each limb (shoulder/hip 3 DOF + elbow/knee) to match BOTH the upper
# segment direction (shoulder->elbow) AND the lower segment direction
# (elbow->wrist). That fully determines the arm/leg, so the forearm/shin points
# where yours does instead of bending into a guessed plane. When the limb is
# straight the twist about its long axis is unconstrained, so a tiny tie-breaker
# pulls it toward 0 to keep the pose neutral rather than randomly rolled.
TWIST_REG = 0.02

# The forearm has no body of its own (the arm chain ends at elbow_link, whose
# geometry extends along its local +x). This is the forearm/hand direction in
# the elbow_link frame; legs instead use the real ankle_link body.
FOREARM_LOCAL_AXIS = np.array([1.0, 0.0, 0.0])

# The bare h1.xml has no lights/floor/skybox, so the viewer is black and the
# near-black robot vanishes. scene.xml (shipped alongside it) adds a headlight,
# a ground plane and a skybox -- but that skybox fades to black at the bottom.
# We load scene.xml and then repaint the skybox a flat light gray so the robot
# stands out. 0 = black, 255 = white; ~205 is a comfortable light gray.
BACKGROUND_GRAY = 205


def _h1_model_path() -> str:
    from robot_descriptions import h1_mj_description
    robot = Path(h1_mj_description.MJCF_PATH)
    scene = robot.with_name("scene.xml")  # includes h1.xml + lights + floor
    return str(scene if scene.exists() else robot)


def _set_background_gray(model: mujoco.MjModel, level: int) -> None:
    """Repaint the skybox texture a flat gray so the dark robot is visible."""
    level = int(np.clip(level, 0, 255))
    for i in range(model.ntex):
        if model.tex_type[i] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            n = int(model.tex_width[i] * model.tex_height[i] * model.tex_nchannel[i])
            adr = int(model.tex_adr[i])
            model.tex_data[adr:adr + n] = level
            return


# ---------------------------------------------------------------------------
# Landmark helpers (shared logic with play_latest_mujoco_ik.py)
# ---------------------------------------------------------------------------
def _get_points(frame: Dict) -> List:
    """Return canonical z-up landmarks; refuse frames in any other convention."""
    coord = frame.get("coord")
    if coord != COORD_ZUP_XFWD:
        raise ValueError(
            f"expected coord={COORD_ZUP_XFWD!r} but got {coord!r} -- "
            f"is the websocket server writing live/latest_pose.json?")
    pts = frame.get("world")
    return pts if isinstance(pts, list) else []


def _lm(points: List, idx: int, min_vis: float = 0.3) -> Optional[np.ndarray]:
    if idx < 0 or idx >= len(points):
        return None
    p = points[idx]
    if not isinstance(p, (list, tuple)) or len(p) < 4:
        return None
    if float(p[3]) < min_vis:
        return None
    return np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float64)


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


class TorsoBasis:
    def __init__(self, l_sh, r_sh, l_hip, r_hip):
        shoulder_mid = (l_sh + r_sh) * 0.5
        hip_mid = (l_hip + r_hip) * 0.5
        up = _unit(shoulder_mid - hip_mid)
        right = _unit(r_sh - l_sh)
        if up is None or right is None:
            raise ValueError("degenerate torso")
        right = _unit(right - np.dot(right, up) * up)
        forward = _unit(np.cross(up, right))
        self.right, self.up, self.forward = right, up, forward

    def to_robot_world(self, v_mp: np.ndarray) -> Optional[np.ndarray]:
        comp_r = float(np.dot(v_mp, self.right))
        comp_u = float(np.dot(v_mp, self.up))
        comp_f = float(np.dot(v_mp, self.forward))
        return _unit(comp_r * RIGHT + comp_u * UP + comp_f * FORWARD)


# ---------------------------------------------------------------------------
# Per-limb IK against the H1 model
# ---------------------------------------------------------------------------
class H1IK:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        mujoco.mj_resetData(model, self.data)
        # Pin the scratch base upright so solved directions live in the same
        # world frame the targets are expressed in.
        self.data.qpos[0:3] = np.array([0.0, 0.0, STANDING_HEIGHT])
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

        def jid(name):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if i < 0:
                raise RuntimeError(f"H1 joint '{name}' not found.")
            return i

        def bid(name):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if i < 0:
                raise RuntimeError(f"H1 body '{name}' not found.")
            return i

        # Each limb solves 4 joints together: the 3-DOF shoulder/hip plus the
        # elbow/knee. root->mid is the upper segment (upper arm / thigh); the
        # lower segment (forearm / shin) is either the elbow_link's local axis
        # (arms have no distal body) or the mid->tip2 body vector (legs).
        self.limbs = {
            "right_arm": dict(
                joints=["right_shoulder_pitch", "right_shoulder_roll",
                        "right_shoulder_yaw", "right_elbow"],
                root=bid("right_shoulder_pitch_link"), mid=bid("right_elbow_link"),
                lower_axis=FOREARM_LOCAL_AXIS, tip2=None,
            ),
            "left_arm": dict(
                joints=["left_shoulder_pitch", "left_shoulder_roll",
                        "left_shoulder_yaw", "left_elbow"],
                root=bid("left_shoulder_pitch_link"), mid=bid("left_elbow_link"),
                lower_axis=FOREARM_LOCAL_AXIS, tip2=None,
            ),
            "right_leg": dict(
                joints=["right_hip_yaw", "right_hip_roll", "right_hip_pitch",
                        "right_knee"],
                root=bid("right_hip_yaw_link"), mid=bid("right_knee_link"),
                lower_axis=None, tip2=bid("right_ankle_link"),
            ),
            "left_leg": dict(
                joints=["left_hip_yaw", "left_hip_roll", "left_hip_pitch",
                        "left_knee"],
                root=bid("left_hip_yaw_link"), mid=bid("left_knee_link"),
                lower_axis=None, tip2=bid("left_ankle_link"),
            ),
        }
        for limb in self.limbs.values():
            limb["qadr"] = [model.jnt_qposadr[jid(n)] for n in limb["joints"]]
            limb["bounds"] = self._bounds([jid(n) for n in limb["joints"]])
            limb["prev"] = np.zeros(len(limb["joints"]))
            # Index of the twist joint (shoulder_yaw / hip_yaw) within `joints`.
            limb["twist_idx"] = next(
                (i for i, n in enumerate(limb["joints"]) if "yaw" in n), None)

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

    def _upper_dir(self, limb) -> Optional[np.ndarray]:
        return _unit(self.data.xpos[limb["mid"]] - self.data.xpos[limb["root"]])

    def _lower_dir(self, limb) -> Optional[np.ndarray]:
        if limb["tip2"] is not None:
            return _unit(self.data.xpos[limb["tip2"]] - self.data.xpos[limb["mid"]])
        rot = self.data.xmat[limb["mid"]].reshape(3, 3)
        return _unit(rot @ limb["lower_axis"])

    def solve_limb(self, limb, upper_target: np.ndarray,
                   lower_target: Optional[np.ndarray]) -> np.ndarray:
        """Solve the 4 joints so the upper AND lower segments point at their
        targets. If lower_target is None (wrist/ankle not visible), only the
        upper segment is matched and the elbow/knee is kept straight."""
        lo, hi = limb["bounds"]
        qadr, twist = limb["qadr"], limb["twist_idx"]

        def residual(angles):
            self.data.qpos[qadr] = angles
            mujoco.mj_forward(self.model, self.data)
            du = self._upper_dir(limb)
            res = list(np.ones(3) if du is None else (du - upper_target))
            if lower_target is not None:
                dl = self._lower_dir(limb)
                res += list(np.ones(3) if dl is None else (dl - lower_target))
            else:
                res.append(angles[-1])  # keep elbow/knee (last joint) straight
            if twist is not None:
                res.append(TWIST_REG * angles[twist])  # tie-break free twist
            return np.array(res)

        x0 = np.clip(limb["prev"], lo, hi)
        sol = least_squares(residual, x0, bounds=(lo, hi), method="trf", max_nfev=80)
        limb["prev"] = sol.x
        return sol.x


# ---------------------------------------------------------------------------
# Frame -> qpos
# ---------------------------------------------------------------------------
def build_qpos(ik: H1IK, template_qpos: np.ndarray, frame: Dict):
    report: Dict = {"n_points": 0, "solved": [], "skipped": {}}
    pts = _get_points(frame)
    report["n_points"] = len(pts)
    report["source"] = f"world ({frame.get('coord')})"

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
    qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

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

        seg_mp = _unit(mid_lm - root_lm)
        if seg_mp is None:
            report["skipped"][name] = "zero-length segment"
            continue
        upper_target = basis.to_robot_world(seg_mp)
        if upper_target is None:
            report["skipped"][name] = "zero target direction"
            continue

        # Forearm/shin direction target (skip if the wrist/ankle isn't visible).
        lower_target = None
        if tip_lm is not None:
            fore_mp = _unit(tip_lm - mid_lm)
            if fore_mp is not None:
                lower_target = basis.to_robot_world(fore_mp)

        angles = ik.solve_limb(limb, upper_target, lower_target)
        qpos[limb["qadr"]] = angles

        report["solved"].append(name)

    return qpos, report


def main():
    model = mujoco.MjModel.from_xml_path(_h1_model_path())
    _set_background_gray(model, BACKGROUND_GRAY)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    template_qpos = np.array(data.qpos, dtype=np.float64).copy()
    template_qpos[0:3] = np.array([0.0, 0.0, STANDING_HEIGHT])
    template_qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

    ik = H1IK(model)

    data.qpos[:] = template_qpos
    mujoco.mj_forward(model, data)

    print(f"Watching {RAW_FILE} (H1 IK retargeting)")
    if not RAW_FILE.exists():
        print(f"  NOTE: {RAW_FILE} does not exist yet. Start the websocket server "
              f"and begin RECORDING in the browser.")
    last_mtime = 0.0
    warned_no_move = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if RAW_FILE.exists():
                mtime = RAW_FILE.stat().st_mtime
                if mtime != last_mtime:
                    last_mtime = mtime
                    try:
                        raw = RAW_FILE.read_text(encoding="utf-8")
                        if raw.strip():
                            frame = json.loads(raw)
                            qpos, report = build_qpos(ik, template_qpos, frame)
                            if qpos is not None:
                                data.qpos[:] = qpos
                                mujoco.mj_forward(model, data)
                                print(f"frame {frame.get('frameIndex')}: "
                                      f"solved {report['solved']} "
                                      f"({report['source']}, {report['n_points']} pts)"
                                      + (f" skipped {report['skipped']}"
                                         if report['skipped'] else ""))
                                if not report["solved"] and not warned_no_move:
                                    warned_no_move = True
                                    print("  -> No limbs solved; see 'skipped' above.")
                            else:
                                print(f"frame {frame.get('frameIndex')}: SKIPPED "
                                      f"-> {report['skipped']}")
                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        print("H1 IK playback error:", repr(e))

            viewer.sync()
            time.sleep(1 / 30)


if __name__ == "__main__":
    main()
