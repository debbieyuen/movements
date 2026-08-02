"""Live MediaPipe -> Unitree H1 playback in MuJoCo using per-limb inverse
kinematics. Successor to my-app/play_latest_h1_mujoco_ik.py.

Reads the canonical z-up frame the pose server writes (live/latest_pose.json),
so there are no axis knobs here: if the frame says coord=zup-xfwd, up is +Z,
full stop. Adds over its predecessor:

  * OneEuro smoothing on landmark channels (jitter-free at rest, responsive
    in motion) before IK.
  * Root yaw from the shoulder line, so turning your body turns the robot.
  * Root height tracked from foot-floor contact, so squats read as squats.

Run (from the repo root, with `uvicorn server.app:app` running and the
browser streaming):

    python -m server.live_viewer_h1                 # window only
    python -m server.live_viewer_h1 --record        # window + mp4 + qpos log
    python -m server.live_viewer_h1 --headless      # record with no window

Recordings land in live/recordings/ as `live_h1_<timestamp>.mp4` alongside a
matching .npz of the robot's joint angles over time.

On macOS the interactive window needs `mjpython -m server.live_viewer_h1`
(a MuJoCo requirement); --headless works under plain python everywhere.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional

import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import least_squares

from mocap.conventions import H1_JOINT_ORDER

from .config import LIVE_DIR
from .one_euro import OneEuro
from .protocol import COORD_ZUP_XFWD

RAW_FILE = LIVE_DIR / "latest_pose.json"

# MediaPipe Pose landmark indices.
NOSE = 0
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24
L_KN, R_KN = 25, 26
L_AN, R_AN = 27, 28

# MuJoCo world: +x forward, +y left, +z up. The robot's own right side is -y.
FORWARD = np.array([1.0, 0.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])
RIGHT = np.array([0.0, -1.0, 0.0])

# Nominal standing pelvis height if the model ships no keyframe.
FALLBACK_STANDING_HEIGHT = 0.98
# The ankle_link origin sits roughly this far above the sole.
ANKLE_SOLE_OFFSET = 0.07
# Blend factors for root pose smoothing (0..1 per frame at ~30 Hz).
ROOT_Z_EMA = 0.3
YAW_EMA = 0.25

# See play_latest_h1_mujoco_ik.py history for the rationale of these.
TWIST_REG = 0.02
FOREARM_LOCAL_AXIS = np.array([1.0, 0.0, 0.0])
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
# Landmark helpers
# ---------------------------------------------------------------------------
def _get_points(frame: Dict) -> List:
    """Return canonical z-up landmarks; refuse frames in any other convention."""
    coord = frame.get("coord")
    if coord != COORD_ZUP_XFWD:
        raise ValueError(
            f"expected coord={COORD_ZUP_XFWD!r} but got {coord!r} -- "
            f"is `uvicorn server.app:app` running and the browser streaming?")
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
    """Body-local frame from shoulders/hips; limb targets are expressed in it,
    so retargeting is robust to where the camera was pointing."""

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

    def to_robot_world(self, v: np.ndarray) -> Optional[np.ndarray]:
        comp_r = float(np.dot(v, self.right))
        comp_u = float(np.dot(v, self.up))
        comp_f = float(np.dot(v, self.forward))
        return _unit(comp_r * RIGHT + comp_u * UP + comp_f * FORWARD)


def facing_yaw(l_sh: np.ndarray, r_sh: np.ndarray) -> float:
    """World yaw of the subject's facing direction from the shoulder line.

    In the canonical frame a subject facing +X has their right side at -Y, so
    the shoulder vector (right - left) points along -Y and
    atan2(-1, 0) + pi/2 = 0: facing straight ahead.
    """
    s = r_sh - l_sh
    return math.atan2(s[1], s[0]) + math.pi / 2


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Per-limb IK against the H1 model
# ---------------------------------------------------------------------------
class H1IK:
    def __init__(self, model: mujoco.MjModel, standing_height: float):
        self.model = model
        self.data = mujoco.MjData(model)
        mujoco.mj_resetData(model, self.data)
        # Pin the scratch base upright so solved directions live in the same
        # world frame the targets are expressed in.
        self.data.qpos[0:3] = np.array([0.0, 0.0, standing_height])
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
        self.ankle_bodies = [bid("left_ankle_link"), bid("right_ankle_link")]
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

    def lowest_foot_z(self, qpos: np.ndarray) -> float:
        """Forward kinematics at `qpos`, then the lower of the two foot soles."""
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        return min(float(self.data.xpos[b][2]) for b in self.ankle_bodies) \
            - ANKLE_SOLE_OFFSET


# ---------------------------------------------------------------------------
# Frame -> qpos
# ---------------------------------------------------------------------------
class Retargeter:
    def __init__(self, ik: H1IK, template_qpos: np.ndarray,
                 standing_height: float):
        self.ik = ik
        self.template_qpos = template_qpos
        self.standing_height = standing_height
        self.landmark_filter = OneEuro(min_cutoff=1.5, beta=0.3)
        self._yaw: Optional[float] = None
        self._root_z: Optional[float] = None

    def __call__(self, frame: Dict):
        report: Dict = {"n_points": 0, "solved": [], "skipped": {}}
        pts = _get_points(frame)
        report["n_points"] = len(pts)
        report["source"] = f"world ({frame.get('coord')})"

        if len(pts) == 33:
            arr = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float64)
            arr = self.landmark_filter.filter(
                arr, t=float(frame.get("tMs", 0.0)) / 1000.0)
            pts = [[*arr[i], pts[i][3]] for i in range(33)]

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

        qpos = self.template_qpos.copy()
        qpos[0:3] = np.array([0.0, 0.0, self.standing_height])

        # Root yaw: follow the subject's facing direction (circular EMA).
        yaw = facing_yaw(l_sh, r_sh)
        if self._yaw is None:
            self._yaw = yaw
        else:
            self._yaw += YAW_EMA * _wrap_angle(yaw - self._yaw)
        qpos[3:7] = yaw_to_quat_wxyz(self._yaw)

        limb_lms = {
            "right_arm": (r_sh, _lm(pts, R_EL), _lm(pts, R_WR)),
            "left_arm": (l_sh, _lm(pts, L_EL), _lm(pts, L_WR)),
            "right_leg": (r_hip, _lm(pts, R_KN), _lm(pts, R_AN)),
            "left_leg": (l_hip, _lm(pts, L_KN), _lm(pts, L_AN)),
        }

        for name, (root_lm, mid_lm, tip_lm) in limb_lms.items():
            limb = self.ik.limbs[name]
            if root_lm is None or mid_lm is None:
                report["skipped"][name] = "missing shoulder/elbow (or hip/knee) landmark"
                continue

            seg = _unit(mid_lm - root_lm)
            if seg is None:
                report["skipped"][name] = "zero-length segment"
                continue
            upper_target = basis.to_robot_world(seg)
            if upper_target is None:
                report["skipped"][name] = "zero target direction"
                continue

            # Forearm/shin direction target (skip if wrist/ankle not visible).
            lower_target = None
            if tip_lm is not None:
                fore = _unit(tip_lm - mid_lm)
                if fore is not None:
                    lower_target = basis.to_robot_world(fore)

            angles = self.ik.solve_limb(limb, upper_target, lower_target)
            qpos[limb["qadr"]] = angles
            report["solved"].append(name)

        # Root height: drop the pelvis so the lower foot touches the floor.
        # Bent knees pull the (kinematic) feet up; shifting the root down by
        # the same amount makes squats read as squats.
        if any("leg" in n for n in report["solved"]):
            foot_z = self.ik.lowest_foot_z(qpos)
            target_z = float(qpos[2]) - foot_z
            if self._root_z is None:
                self._root_z = target_z
            else:
                self._root_z += ROOT_Z_EMA * (target_z - self._root_z)
            qpos[2] = self._root_z

        return qpos, report


class LiveRecorder:
    """Records what the H1 is doing: an mp4 of the robot plus the qpos log.

    The interactive viewer draws to a window and keeps nothing, so without
    this the live retargeting is unrecoverable the moment it scrolls past.
    Rendering happens off the interactive context via mujoco.Renderer, which
    also means it works in --headless mode with no window at all.
    """

    def __init__(self, model: mujoco.MjModel, out_dir: Path, fps: int = 30,
                 width: int = 640, height: int = 480):
        import imageio.v2 as imageio

        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.mp4_path = out_dir / f"live_h1_{stamp}.mp4"
        self.npz_path = out_dir / f"live_h1_{stamp}.npz"
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.scratch = mujoco.MjData(model)
        self.camera = mujoco.MjvCamera()
        self.writer = imageio.get_writer(
            self.mp4_path, fps=fps, codec="libx264", quality=7,
            macro_block_size=2)
        self.model = model
        self.qpos_log: list[np.ndarray] = []
        self.t_log: list[float] = []
        self.frames = 0

    def add(self, qpos: np.ndarray, t_ms: float) -> None:
        self.qpos_log.append(np.asarray(qpos, dtype=np.float32).copy())
        self.t_log.append(float(t_ms) / 1000.0)
        self.scratch.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.scratch)
        self.camera.lookat[:] = [qpos[0], qpos[1], max(float(qpos[2]), 0.6)]
        self.camera.distance = 3.0
        self.camera.azimuth = 135.0
        self.camera.elevation = -12.0
        self.renderer.update_scene(self.scratch, camera=self.camera)
        self.writer.append_data(self.renderer.render())
        self.frames += 1

    def close(self) -> None:
        self.writer.close()
        self.renderer.close()
        if self.qpos_log:
            np.savez_compressed(
                self.npz_path,
                t=np.array(self.t_log, dtype=np.float64),
                qpos=np.stack(self.qpos_log),
                joint_names=np.array(H1_JOINT_ORDER),
            )
        print(f"[live] wrote {self.frames} frames -> {self.mp4_path.name}"
              + (f" + {self.npz_path.name}" if self.qpos_log else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", nargs="?", const=str(LIVE_DIR / "recordings"),
                    default=None, metavar="DIR",
                    help="save an mp4 of the robot plus a qpos .npz into DIR")
    ap.add_argument("--headless", action="store_true",
                    help="no viewer window; only record (implies --record)")
    args = ap.parse_args()

    record_dir = args.record
    if args.headless and record_dir is None:
        record_dir = str(LIVE_DIR / "recordings")

    model = mujoco.MjModel.from_xml_path(_h1_model_path())
    _set_background_gray(model, BACKGROUND_GRAY)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Nominal pelvis height: model keyframe if one exists, else fallback.
    standing_height = FALLBACK_STANDING_HEIGHT
    if model.nkey > 0:
        standing_height = float(model.key_qpos[0][2])

    template_qpos = np.array(data.qpos, dtype=np.float64).copy()
    template_qpos[0:3] = np.array([0.0, 0.0, standing_height])
    template_qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

    retarget = Retargeter(H1IK(model, standing_height), template_qpos,
                          standing_height)

    data.qpos[:] = template_qpos
    mujoco.mj_forward(model, data)

    recorder = LiveRecorder(model, Path(record_dir)) if record_dir else None

    print(f"Watching {RAW_FILE} (H1 IK retargeting, canonical z-up input)")
    if not RAW_FILE.exists():
        print("  NOTE: file does not exist yet. Start `uvicorn server.app:app` "
              "and begin streaming in the browser.")
    if recorder:
        print(f"  recording to {recorder.mp4_path}")
    last_mtime = 0.0

    # Shut down cleanly on ctrl-c AND on `kill` (SIGTERM): the qpos log is only
    # written at close, and a default SIGTERM would kill the process mid-loop
    # and throw the whole take away. Shells also set SIGINT to ignore for
    # background jobs, so a handler is the only reliable stop signal there.
    stopping = False

    def request_stop(signum, _frame) -> None:
        nonlocal stopping
        if not stopping:
            print(f"\n[live] signal {signum}: finishing the recording...")
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def step() -> bool:
        """Consume the newest frame if there is one. Returns False on fatal."""
        nonlocal last_mtime
        if not RAW_FILE.exists():
            return True
        mtime = RAW_FILE.stat().st_mtime
        if mtime == last_mtime:
            return True
        last_mtime = mtime
        try:
            raw = RAW_FILE.read_text(encoding="utf-8")
            if not raw.strip():
                return True
            frame = json.loads(raw)
            qpos, report = retarget(frame)
            if qpos is None:
                print(f"frame {frame.get('seq')}: SKIPPED -> {report['skipped']}")
                return True
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            if recorder:
                recorder.add(qpos, frame.get("tMs", 0.0))
        except json.JSONDecodeError:
            pass
        except Exception as e:  # noqa: BLE001 - never kill the live loop
            print("H1 IK playback error:", repr(e))
        return True

    last_report = time.monotonic()
    try:
        if args.headless:
            print("  headless: streaming to video only (ctrl-c to stop)")
            while not stopping:
                step()
                if recorder and time.monotonic() - last_report >= 2.0:
                    last_report = time.monotonic()
                    print(f"[live] {recorder.frames} frames recorded")
                time.sleep(1 / 30)
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running() and not stopping:
                    step()
                    viewer.sync()
                    time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        if recorder:
            recorder.close()


if __name__ == "__main__":
    main()
