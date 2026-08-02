"""Dataset v2 conventions — the single source of truth for units and frames.

World frame: right-handed, +Z up, gravity -Z, floor at z=0, meters, radians,
seconds, quaternions **wxyz** (MuJoCo order), 30 Hz uniform sampling,
trajectories in the WORLD frame (not hip-centered), initial pelvis heading
roughly +X.

Every clip's meta.json restates these so a clip is self-describing.
"""

from __future__ import annotations

TARGET_FPS = 30.0

CONVENTIONS = {
    "handedness": "right",
    "up_axis": "+z",
    "gravity": "-z",
    "floor": "z=0",
    "units": {"length": "m", "angle": "rad", "time": "s"},
    "quaternion_order": "wxyz",
    "frame": "world (not hip-centered)",
    "fps": TARGET_FPS,
}

# Unitree H1 actuated joints in h1_mj_description qpos order (qpos[7:26]).
# Verified against the MuJoCo Menagerie model: free joint first, then these 19.
H1_JOINT_ORDER = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
    "torso",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
]

# Joint velocity limits (rad/s) from the official Unitree H1 URDF
# (unitree_ros/robots/h1_description/urdf/h1.urdf, <limit velocity=...>).
# The Menagerie MJCF does not carry velocity limits, so they live here.
H1_VELOCITY_LIMITS = {
    "left_hip_yaw": 23.0,
    "left_hip_roll": 23.0,
    "left_hip_pitch": 23.0,
    "left_knee": 14.0,
    "left_ankle": 9.0,
    "right_hip_yaw": 23.0,
    "right_hip_roll": 23.0,
    "right_hip_pitch": 23.0,
    "right_knee": 14.0,
    "right_ankle": 9.0,
    "torso": 23.0,
    "left_shoulder_pitch": 9.0,
    "left_shoulder_roll": 9.0,
    "left_shoulder_yaw": 20.0,
    "left_elbow": 20.0,
    "right_shoulder_pitch": 9.0,
    "right_shoulder_roll": 9.0,
    "right_shoulder_yaw": 20.0,
    "right_elbow": 20.0,
}

QPOS_DIM = 7 + len(H1_JOINT_ORDER)  # 26: free base [x y z, qw qx qy qz] + 19
QVEL_DIM = 6 + len(H1_JOINT_ORDER)  # 25

# SMPL body: 24 joints (including pelvis root), 23 * 3 body pose params.
SMPL_NUM_JOINTS = 24
SMPL_PELVIS = 0

# MediaPipe ankle-height heuristics used by contact detection.
CONTACT_HEIGHT_M = 0.08
CONTACT_SPEED_MS = 0.15


def h1_model_path() -> str:
    """The Menagerie H1 MJCF (scene.xml with floor+light when available)."""
    from pathlib import Path

    from robot_descriptions import h1_mj_description

    robot = Path(h1_mj_description.MJCF_PATH)
    scene = robot.with_name("scene.xml")
    return str(scene if scene.exists() else robot)


def load_h1_model():
    import mujoco

    return mujoco.MjModel.from_xml_path(h1_model_path())
