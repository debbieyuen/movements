"""Standalone exporter, run INSIDE the GVHMR environment (needs torch+smplx):

    python gvhmr_export.py <hmr4d_results.pt> <out.npz> <fps>

Reads GVHMR's global (world-frame) SMPL sequence and writes a plain npz:
    smpl_body_pose (T, P) axis-angle   smpl_global_orient (T, 3)
    smpl_transl (T, 3)                 smpl_betas (10,)
    joints_3d (T, J, 3)                fps ()

Coordinates are converted to the dataset convention: right-handed, Z-UP,
gravity -z. GVHMR's world frame is gravity-aligned but its up-axis label is
verified empirically here: we pick the axis transform that makes the
pelvis->head direction point +z, and hard-fail if none does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

CANDIDATE_ROTS = {
    # name -> rotation matrix applied to points (world_new = R @ world_old)
    "identity(z-up)": np.eye(3),
    "y-up->z-up": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    "y-down->z-up": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
}

SMPL_HEAD, SMPL_PELVIS = 15, 0


def main() -> None:
    pt_path, out_npz, fps = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
    pred = torch.load(pt_path, map_location="cpu")

    # GVHMR stores the world-frame sequence under smpl_params_global.
    params = pred["smpl_params_global"]
    body_pose = params["body_pose"].numpy()          # (T, P)
    betas = params["betas"].numpy()
    if betas.ndim == 2:
        betas = betas.mean(axis=0)                    # per-frame -> per-clip
    global_orient = params["global_orient"].numpy()  # (T, 3) axis-angle
    transl = params["transl"].numpy()                # (T, 3)

    joints = _smpl_joints(body_pose, betas, global_orient, transl)

    R, name = _pick_up_axis(joints)
    print(f"[gvhmr_export] axis transform: {name}")
    joints = joints @ R.T
    transl = transl @ R.T
    global_orient = _rotate_aa(global_orient, R)

    np.savez_compressed(
        out_npz,
        smpl_body_pose=body_pose.astype(np.float32),
        smpl_global_orient=global_orient.astype(np.float32),
        smpl_transl=transl.astype(np.float32),
        smpl_betas=betas.astype(np.float32)[:10],
        joints_3d=joints.astype(np.float32),
        fps=np.float64(fps),
    )
    print(f"[gvhmr_export] wrote {out_npz} ({len(body_pose)} frames)")


def _smpl_joints(body_pose, betas, global_orient, transl) -> np.ndarray:
    """Forward the SMPL(-X) body model to get world joint positions."""
    import smplx

    T, P = body_pose.shape
    model_type = "smplx" if P == 63 else "smpl"
    model = smplx.create(
        _body_model_dir(), model_type=model_type, gender="neutral",
        batch_size=T, use_pca=False,
    )
    with torch.no_grad():
        out = model(
            betas=torch.tensor(betas[:10], dtype=torch.float32).expand(T, -1),
            body_pose=torch.tensor(body_pose, dtype=torch.float32),
            global_orient=torch.tensor(global_orient, dtype=torch.float32),
            transl=torch.tensor(transl, dtype=torch.float32),
        )
    return out.joints[:, :24].cpu().numpy()


def _body_model_dir() -> str:
    """GVHMR keeps body models under inputs/checkpoints/body_models."""
    here = Path.cwd()
    for cand in [
        here / "inputs" / "checkpoints" / "body_models",
        here / "assets" / "body_models",
    ]:
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        "SMPL body models not found (expected under inputs/checkpoints/body_models)")


def _pick_up_axis(joints: np.ndarray):
    """Choose the axis rotation under which pelvis->head points +z."""
    up = (joints[:, SMPL_HEAD] - joints[:, SMPL_PELVIS]).mean(axis=0)
    up /= np.linalg.norm(up) + 1e-9
    best_name, best_R, best_score = None, None, -2.0
    for name, R in CANDIDATE_ROTS.items():
        score = float((R @ up)[2])
        if score > best_score:
            best_name, best_R, best_score = name, R, score
    if best_score < 0.7:
        raise RuntimeError(
            f"could not identify the up axis (best {best_name} scored {best_score:.2f})")
    return best_R, best_name


def _rotate_aa(aa: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate axis-angle orientations by R (world-side)."""
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_matrix(R) * Rotation.from_rotvec(aa)
    return rot.as_rotvec()


if __name__ == "__main__":
    main()
