"""Post-processing for retargeted H1 motion, in this order:

  1. resample to TARGET_FPS (linear joints/translation, SLERP base quat)
  2. floor alignment (5th-percentile min foot height -> z=0)
  3. zero-phase Butterworth smoothing (joints + base translation + base rot)
  4. joint-limit clamp (model ranges) + per-step velocity clamp (URDF limits)
  5. foot contact detection (+ foot-skate metric, reported not corrected)
  6. qvel from the CLEANED qpos via mujoco.mj_differentiatePos

The order matters: velocities are computed last, from cleaned poses, so the
dataset can never again contain the 56 rad/s finite-difference garbage the
v1 exporter produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import mujoco
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp

from .conventions import (
    CONTACT_HEIGHT_M,
    CONTACT_SPEED_MS,
    H1_JOINT_ORDER,
    H1_VELOCITY_LIMITS,
    TARGET_FPS,
)

# The ankle_link origin sits roughly this far above the sole.
ANKLE_SOLE_OFFSET = 0.07
SMOOTH_CUTOFF_HZ = 6.0


@dataclass
class ProcessedClip:
    t: np.ndarray          # (T,) seconds, uniform
    qpos: np.ndarray       # (T, 26)
    qvel: np.ndarray       # (T, 25)
    contacts: np.ndarray   # (T, 2) bool [left, right]
    quality: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def resample(qpos: np.ndarray, src_fps: float, dst_fps: float = TARGET_FPS):
    """Uniformly resample qpos rows: linear for translation+joints, SLERP for
    the base quaternion. Returns (t_new, qpos_new)."""
    T = len(qpos)
    t_src = np.arange(T) / src_fps
    duration = t_src[-1] if T > 1 else 0.0
    t_dst = np.arange(0.0, duration + 1e-9, 1.0 / dst_fps)

    out = np.empty((len(t_dst), qpos.shape[1]), dtype=np.float64)
    for col in [0, 1, 2, *range(7, qpos.shape[1])]:
        out[:, col] = np.interp(t_dst, t_src, qpos[:, col])

    # wxyz -> scipy xyzw
    quats = np.roll(qpos[:, 3:7], -1, axis=1)
    slerp = Slerp(t_src, Rotation.from_quat(quats))
    out[:, 3:7] = np.roll(slerp(np.clip(t_dst, 0, t_src[-1])).as_quat(), 1, axis=1)
    return t_dst, out


def foot_heights(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """(T, 2) world z of the [left, right] foot soles via forward kinematics."""
    data = mujoco.MjData(model)
    bids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_link")
        for side in ("left", "right")
    ]
    if any(b < 0 for b in bids):
        raise RuntimeError("ankle_link bodies not found in the H1 model")
    out = np.empty((len(qpos), 2))
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        out[i] = [data.xpos[b][2] - ANKLE_SOLE_OFFSET for b in bids]
    return out


def align_floor(model: mujoco.MjModel, qpos: np.ndarray) -> tuple[np.ndarray, float]:
    """Shift base z so the 5th-percentile minimum foot height sits at z=0,
    then clamp residual penetration."""
    heights = foot_heights(model, qpos)
    floor = float(np.percentile(heights.min(axis=1), 5))
    qpos = qpos.copy()
    qpos[:, 2] -= floor
    heights -= floor
    # Clamp per-frame penetration: lift the base just enough.
    penetration = np.minimum(heights.min(axis=1), 0.0)
    qpos[:, 2] -= penetration
    return qpos, floor


def smooth(qpos: np.ndarray, fps: float, cutoff_hz: float = SMOOTH_CUTOFF_HZ) -> np.ndarray:
    """Zero-phase Butterworth on translation + joints; rotation-vector domain
    for the base quaternion (relative to frame 0 so there is no wrap seam)."""
    if len(qpos) < 15:  # filtfilt needs padding room on short clips
        return qpos.copy()
    b, a = butter(4, cutoff_hz / (fps / 2.0))
    out = qpos.copy()
    cols = [0, 1, 2, *range(7, qpos.shape[1])]
    out[:, cols] = filtfilt(b, a, qpos[:, cols], axis=0)

    quats = Rotation.from_quat(np.roll(qpos[:, 3:7], -1, axis=1))
    rel = (quats[0].inv() * quats).as_rotvec()
    rel_smooth = filtfilt(b, a, rel, axis=0)
    smoothed = quats[0] * Rotation.from_rotvec(rel_smooth)
    out[:, 3:7] = np.roll(smoothed.as_quat(), 1, axis=1)
    return out


def clamp_limits(model: mujoco.MjModel, qpos: np.ndarray, fps: float) -> np.ndarray:
    """Clamp joints to model ranges, then forward-pass velocity clamp against
    the URDF limits: q[i] = q[i-1] + clip(dq, ±vmax*dt)."""
    out = qpos.copy()

    lo = np.full(len(H1_JOINT_ORDER), -np.inf)
    hi = np.full(len(H1_JOINT_ORDER), np.inf)
    for k, name in enumerate(H1_JOINT_ORDER):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0 and model.jnt_limited[jid]:
            lo[k], hi[k] = model.jnt_range[jid]
    out[:, 7:] = np.clip(out[:, 7:], lo, hi)

    vmax = np.array([H1_VELOCITY_LIMITS[n] for n in H1_JOINT_ORDER])
    dt = 1.0 / fps
    for i in range(1, len(out)):
        dq = out[i, 7:] - out[i - 1, 7:]
        out[i, 7:] = out[i - 1, 7:] + np.clip(dq, -vmax * dt, vmax * dt)
    return out


def detect_contacts(model: mujoco.MjModel, qpos: np.ndarray, fps: float):
    """(T, 2) stance flags with hysteresis, plus total foot-skate distance (m)
    per foot while in contact."""
    heights = foot_heights(model, qpos)

    # Foot xy positions for speed/skate measurement.
    data = mujoco.MjData(model)
    bids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_link")
        for side in ("left", "right")
    ]
    xy = np.empty((len(qpos), 2, 2))
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for f, b in enumerate(bids):
            xy[i, f] = data.xpos[b][:2]

    speed = np.zeros((len(qpos), 2))
    if len(qpos) > 1:
        speed[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=2) * fps

    contacts = np.zeros((len(qpos), 2), dtype=bool)
    skate = np.zeros(2)
    on_h, off_h = CONTACT_HEIGHT_M, CONTACT_HEIGHT_M * 1.5
    for f in range(2):
        in_contact = False
        for i in range(len(qpos)):
            if in_contact:
                if heights[i, f] > off_h:
                    in_contact = False
            else:
                if heights[i, f] < on_h and speed[i, f] < CONTACT_SPEED_MS:
                    in_contact = True
            contacts[i, f] = in_contact
            if in_contact and i > 0 and contacts[i - 1, f]:
                skate[f] += float(np.linalg.norm(xy[i, f] - xy[i - 1, f]))
    return contacts, skate


def differentiate(model: mujoco.MjModel, qpos: np.ndarray, fps: float) -> np.ndarray:
    """qvel via mj_differentiatePos (correct for the free-joint quaternion)."""
    qvel = np.zeros((len(qpos), model.nv))
    scratch = np.zeros(model.nv)
    dt = 1.0 / fps
    for i in range(1, len(qpos)):
        mujoco.mj_differentiatePos(model, scratch, dt, qpos[i - 1], qpos[i])
        qvel[i] = scratch
    return qvel


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def postprocess(
    qpos_raw: np.ndarray,
    src_fps: float,
    model: mujoco.MjModel,
    *,
    dst_fps: float = TARGET_FPS,
) -> ProcessedClip:
    t, qpos = resample(np.asarray(qpos_raw, dtype=np.float64), src_fps, dst_fps)
    qpos, floor_offset = align_floor(model, qpos)
    qpos = smooth(qpos, dst_fps)
    qpos = clamp_limits(model, qpos, dst_fps)
    # normalize quats (filtering can drift the norm slightly)
    qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    contacts, skate = detect_contacts(model, qpos, dst_fps)
    qvel = differentiate(model, qpos, dst_fps)

    quality = {
        "n_frames": int(len(qpos)),
        "duration_s": float(t[-1]) if len(t) else 0.0,
        "floor_offset_m": floor_offset,
        "max_abs_qvel_rad_s": float(np.abs(qvel[:, 6:]).max()) if len(qvel) > 1 else 0.0,
        "foot_skate_m": {"left": float(skate[0]), "right": float(skate[1])},
        "contact_fraction": {
            "left": float(contacts[:, 0].mean()) if len(contacts) else 0.0,
            "right": float(contacts[:, 1].mean()) if len(contacts) else 0.0,
        },
    }
    return ProcessedClip(t=t, qpos=qpos, qvel=qvel, contacts=contacts, quality=quality)
