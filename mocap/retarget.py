"""GMR wrapper: GVHMR output -> Unitree H1 qpos.

GMR — General Motion Retargeting (https://github.com/YanjieZe/GMR, MIT) —
does whole-body IK with a floating base and ships a converter that consumes
GVHMR's hmr4d_results.pt directly:

    python scripts/gvhmr_to_robot.py --gvhmr_pred_file <pt> \
        --robot unitree_h1 --save_path <pkl>

Its pickle carries robot_base_translation (T,3), robot_base_rotation (T,4,
wxyz), robot_joint_positions (T,J), the joint name list, and fps. GMR's H1
joint order is ITS model's order, so `gmr_to_qpos` reindexes BY NAME into
h1_mj_description qpos order — never positionally.

Environment knobs mirror recovery.py:
  MOCAP_GMR_DIR, MOCAP_GMR_PYTHON, MOCAP_GMR_CONDA_ENV
"""

from __future__ import annotations

import os
import pickle
import subprocess
from pathlib import Path

import numpy as np

from .conventions import H1_JOINT_ORDER, QPOS_DIM

REPO_ROOT = Path(__file__).resolve().parents[1]


def _gmr_dir() -> Path:
    return Path(os.environ.get("MOCAP_GMR_DIR", REPO_ROOT / "external" / "GMR"))


def _runner() -> list[str]:
    if env := os.environ.get("MOCAP_GMR_CONDA_ENV"):
        return ["conda", "run", "--no-capture-output", "-n", env, "python"]
    return [os.environ.get("MOCAP_GMR_PYTHON", "python")]


def run_gmr(pred_pt: Path, out_pkl: Path, *, robot: str = "unitree_h1") -> Path:
    if out_pkl.exists():
        print(f"[retarget] cached: {out_pkl}")
        return out_pkl
    gmr = _gmr_dir()
    script = gmr / "scripts" / "gvhmr_to_robot.py"
    if not script.exists():
        raise FileNotFoundError(
            f"GMR not found at {gmr} (see docs/GPU_SETUP.md; "
            f"set MOCAP_GMR_DIR if it lives elsewhere)")
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    cmd = [*_runner(), str(script),
           "--gvhmr_pred_file", str(pred_pt.resolve()),
           "--robot", robot,
           "--save_path", str(out_pkl.resolve())]
    print(f"[retarget] $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, cwd=gmr, check=True)
    if not out_pkl.exists():
        raise RuntimeError("GMR produced no output pickle")
    return out_pkl


def _find(d: dict, *names: str):
    for n in names:
        if n in d:
            return d[n]
    return None


def gmr_to_qpos(pkl_path: Path) -> tuple[np.ndarray, float]:
    """Load a GMR result pickle and assemble (T, 26) qpos in
    h1_mj_description order. Returns (qpos, source_fps)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    transl = np.asarray(_find(data, "robot_base_translation", "base_translation"))
    quat = np.asarray(_find(data, "robot_base_rotation", "base_rotation"))
    joints = np.asarray(_find(data, "robot_joint_positions", "joint_positions"))
    names = _find(data, "robot_joint_names", "joint_names", "dof_names")
    fps = float(_find(data, "fps", "framerate") or 30.0)

    if transl is None or quat is None or joints is None:
        raise KeyError(f"unrecognized GMR pickle layout: keys={sorted(data)}")

    T = len(joints)
    if names is None:
        raise KeyError(
            "GMR pickle carries no joint names; refusing to map positionally. "
            f"keys={sorted(data)}")
    names = [str(n).removesuffix("_joint") for n in names]
    missing = [n for n in H1_JOINT_ORDER if n not in names]
    if missing:
        raise KeyError(f"GMR result lacks joints {missing}; has {names}")
    index = [names.index(n) for n in H1_JOINT_ORDER]

    qpos = np.zeros((T, QPOS_DIM), dtype=np.float64)
    qpos[:, 0:3] = transl
    qpos[:, 3:7] = quat  # GMR uses wxyz — same as MuJoCo
    qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    qpos[:, 7:] = joints[:, index]
    return qpos, fps
