"""Physics validation: can the H1 actually PERFORM this motion?

The retargeting pipeline is kinematic — it poses the robot without asking
whether physics agrees. This module replays a clip inside MuJoCo's dynamics:
a PD controller in torque space tracks the clip's joint trajectory while
gravity, contacts, and the robot's real 51 kg act on it. What survives is a
physically consistent rollout; what falls over is flagged.

Balance is the honest difficulty here. Joint-tracking PD alone cannot keep a
humanoid upright through dynamic motion — that normally takes a trained
whole-body controller, which is out of scope. So the rollout offers a BASE
ASSIST: a virtual spring-damper pulling the pelvis toward the reference base
pose, like a loose harness on a treadmill patient. With assist on (default)
the metrics measure joint-level executability; with `assist_stiffness=0` they
measure unassisted balance, which is brutal but honest. The assist forces are
themselves a metric: a clip needing large harness forces is far from feasible.

Everything reported lands in the clip's meta.json quality block, and the
simulated trajectory is stored as `qpos_sim` so downstream consumers (e.g.
egocentric rendering) can use physically consistent states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import mujoco
import numpy as np

from .conventions import H1_JOINT_ORDER

# PD gains in torque space (actuators are direct torque motors). Legs carry
# the body -> stiff; arms only carry themselves -> softer. Values sized
# against the URDF effort limits (hips/knees 200-300 Nm, shoulders 18-40 Nm).
PD_GAINS = {
    "hip": (180.0, 8.0),
    "knee": (250.0, 10.0),
    "ankle": (35.0, 3.0),
    "torso": (150.0, 8.0),
    "shoulder": (35.0, 2.0),
    "elbow": (16.0, 1.5),
}

# Base assist spring-damper (pelvis -> reference base pose).
ASSIST_STIFFNESS = 800.0   # N/m and Nm/rad
ASSIST_DAMPING = 80.0

FALL_HEIGHT_M = 0.45       # pelvis below this = fallen


def _gains_for(joint: str) -> tuple[float, float]:
    for key, kd_kp in PD_GAINS.items():
        if key in joint:
            return kd_kp
    return (50.0, 3.0)


@dataclass
class RolloutResult:
    qpos_sim: np.ndarray          # (T, 26) simulated states on the clip's grid
    fell_at_s: float | None
    metrics: Dict = field(default_factory=dict)


def rollout(
    model: mujoco.MjModel,
    t: np.ndarray,
    qpos_ref: np.ndarray,
    *,
    assist_stiffness: float = ASSIST_STIFFNESS,
    assist_damping: float = ASSIST_DAMPING,
) -> RolloutResult:
    """Track the reference trajectory under full dynamics."""
    data = mujoco.MjData(model)
    nq_j = len(H1_JOINT_ORDER)

    kp = np.empty(nq_j)
    kd = np.empty(nq_j)
    for i, name in enumerate(H1_JOINT_ORDER):
        kp[i], kd[i] = _gains_for(name)
    torque_lim = model.actuator_ctrlrange[:, 1].copy()

    # Start exactly on the reference, at rest.
    data.qpos[:] = qpos_ref[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    dt_sim = model.opt.timestep
    fps = 1.0 / float(t[1] - t[0]) if len(t) > 1 else 30.0
    steps_per_frame = max(1, int(round((1.0 / fps) / dt_sim)))

    qpos_sim = np.empty_like(qpos_ref)
    qpos_sim[0] = data.qpos
    fell_at: float | None = None
    joint_err_acc = 0.0
    assist_force_acc = 0.0
    max_assist_force = 0.0
    n_acc = 0

    pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    for f in range(1, len(qpos_ref)):
        # Linear interpolation of the joint reference across sim substeps.
        q_prev, q_next = qpos_ref[f - 1], qpos_ref[f]
        for s in range(steps_per_frame):
            alpha = (s + 1) / steps_per_frame
            q_ref_j = (1 - alpha) * q_prev[7:] + alpha * q_next[7:]

            # PD torque on the 19 actuated joints.
            tau = kp * (q_ref_j - data.qpos[7:]) - kd * data.qvel[6:]
            data.ctrl[:] = np.clip(tau, -torque_lim, torque_lim)

            # Base assist: virtual spring-damper wrench on the pelvis.
            data.xfrc_applied[pelvis_bid][:] = 0.0
            if assist_stiffness > 0.0:
                base_ref = (1 - alpha) * q_prev[:3] + alpha * q_next[:3]
                force = (assist_stiffness * (base_ref - data.qpos[:3])
                         - assist_damping * data.qvel[:3])
                # Orientation assist: torque toward the reference quaternion.
                q_err = np.zeros(3)
                q_ref_quat = q_next[3:7] / np.linalg.norm(q_next[3:7])
                mujoco.mju_subQuat(q_err, q_ref_quat, data.qpos[3:7])
                torque = (assist_stiffness * 0.5 * q_err
                          - assist_damping * 0.5 * data.qvel[3:6])
                data.xfrc_applied[pelvis_bid][:3] = force
                data.xfrc_applied[pelvis_bid][3:] = torque
                fmag = float(np.linalg.norm(force))
                assist_force_acc += fmag
                max_assist_force = max(max_assist_force, fmag)

            mujoco.mj_step(model, data)

        qpos_sim[f] = data.qpos
        joint_err_acc += float(np.abs(data.qpos[7:] - q_next[7:]).mean())
        n_acc += 1

        if fell_at is None and float(data.qpos[2]) < FALL_HEIGHT_M:
            fell_at = float(t[f])
            # Keep simulating so qpos_sim stays complete; the flag is enough.

    weight = 9.81 * float(sum(model.body_mass))
    metrics = {
        "assisted": assist_stiffness > 0.0,
        "assist_stiffness": assist_stiffness,
        "fell_at_s": fell_at,
        "survived": fell_at is None,
        "mean_joint_tracking_error_rad": round(joint_err_acc / max(n_acc, 1), 4),
        "mean_assist_force_N": round(assist_force_acc / max(n_acc * steps_per_frame, 1), 1),
        "max_assist_force_N": round(max_assist_force, 1),
        "assist_force_vs_bodyweight": round(max_assist_force / weight, 3),
        "base_position_rmse_m": round(float(np.sqrt(
            ((qpos_sim[:, :3] - qpos_ref[:, :3]) ** 2).mean())), 4),
    }
    return RolloutResult(qpos_sim=qpos_sim, fell_at_s=fell_at, metrics=metrics)
