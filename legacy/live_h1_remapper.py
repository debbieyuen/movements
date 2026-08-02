import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

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

# H1 body layout used by the live JSON output
H1_JOINT_NAMES = [
    "base_x",
    "base_y",
    "base_z",
    "base_qx",
    "base_qy",
    "base_qz",
    "base_qw",
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


def _vec(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [float(b[0] - a[0]), float(b[1] - a[1]), float(b[2] - a[2])]


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2)


def _unit(v: Sequence[float]) -> List[float]:
    n = _norm(v)
    if n < 1e-9:
        return [0.0, 0.0, 0.0]
    return [float(v[0]) / n, float(v[1]) / n, float(v[2]) / n]


def _mid(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[List[float]]:
    if a is None or b is None:
        return None
    return [
        0.5 * (float(a[0]) + float(b[0])),
        0.5 * (float(a[1]) + float(b[1])),
        0.5 * (float(a[2]) + float(b[2])),
    ]


def _landmark(points: Sequence[Dict[str, Any]], idx: int) -> Optional[List[float]]:
    if idx < 0 or idx >= len(points):
        return None
    p = points[idx]
    if not p:
        return None
    if "x" not in p or "y" not in p or "z" not in p:
        return None
    return [float(p["x"]), float(p["y"]), float(p["z"])]


def _angle(a: Sequence[float], b: Sequence[float]) -> float:
    ua = _unit(a)
    ub = _unit(b)
    dot = max(-1.0, min(1.0, ua[0] * ub[0] + ua[1] * ub[1] + ua[2] * ub[2]))
    return math.acos(dot)


def _quat_from_axes(x_axis: Sequence[float], y_axis: Sequence[float], z_axis: Sequence[float]) -> List[float]:
    # Rotation matrix with axes as columns
    r00, r01, r02 = float(x_axis[0]), float(y_axis[0]), float(z_axis[0])
    r10, r11, r12 = float(x_axis[1]), float(y_axis[1]), float(z_axis[1])
    r20, r21, r22 = float(x_axis[2]), float(y_axis[2]), float(z_axis[2])

    trace = r00 + r11 + r22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r21 - r12) / s
        qy = (r02 - r20) / s
        qz = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = math.sqrt(max(1.0 + r00 - r11 - r22, 1e-12)) * 2.0
        qw = (r21 - r12) / s
        qx = 0.25 * s
        qy = (r01 + r10) / s
        qz = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(max(1.0 + r11 - r00 - r22, 1e-12)) * 2.0
        qw = (r02 - r20) / s
        qx = (r01 + r10) / s
        qy = 0.25 * s
        qz = (r12 + r21) / s
    else:
        s = math.sqrt(max(1.0 + r22 - r00 - r11, 1e-12)) * 2.0
        qw = (r10 - r01) / s
        qx = (r02 + r20) / s
        qy = (r12 + r21) / s
        qz = 0.25 * s

    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    return [qx / n, qy / n, qz / n, qw / n]


@dataclass
class LiveH1State:
    prev_qpos: Optional[List[float]] = None
    prev_time_ms: Optional[float] = None


STATE = LiveH1State()


def pose_to_h1(frame: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one MediaPipe pose frame into a live H1-friendly motion frame.
    This is a heuristic starter mapping for visualization.
    """

    lm = frame.get("worldLandmarks") or frame.get("landmarks") or []
    if not isinstance(lm, list):
        lm = []

    ls = _landmark(lm, LEFT_SHOULDER)
    rs = _landmark(lm, RIGHT_SHOULDER)
    lh = _landmark(lm, LEFT_HIP)
    rh = _landmark(lm, RIGHT_HIP)
    le = _landmark(lm, LEFT_ELBOW)
    re = _landmark(lm, RIGHT_ELBOW)
    lw = _landmark(lm, LEFT_WRIST)
    rw = _landmark(lm, RIGHT_WRIST)
    lk = _landmark(lm, LEFT_KNEE)
    rk = _landmark(lm, RIGHT_KNEE)
    la = _landmark(lm, LEFT_ANKLE)
    ra = _landmark(lm, RIGHT_ANKLE)
    nose = _landmark(lm, NOSE)

    pelvis = _mid(lh, rh)
    chest = _mid(ls, rs)
    if pelvis is None:
        pelvis = [0.0, 0.0, 0.0]
    if chest is None:
        chest = [pelvis[0], pelvis[1] + 0.25, pelvis[2]]

    if ls is None:
        ls = [chest[0] - 0.15, chest[1], chest[2]]
    if rs is None:
        rs = [chest[0] + 0.15, chest[1], chest[2]]
    if lh is None:
        lh = [pelvis[0] - 0.10, pelvis[1] - 0.10, pelvis[2]]
    if rh is None:
        rh = [pelvis[0] + 0.10, pelvis[1] - 0.10, pelvis[2]]
    if le is None:
        le = [ls[0] - 0.12, ls[1] - 0.10, ls[2]]
    if re is None:
        re = [rs[0] + 0.12, rs[1] - 0.10, rs[2]]
    if lw is None:
        lw = [le[0] - 0.10, le[1] - 0.10, le[2]]
    if rw is None:
        rw = [re[0] + 0.10, re[1] - 0.10, re[2]]
    if lk is None:
        lk = [lh[0], lh[1] - 0.25, lh[2]]
    if rk is None:
        rk = [rh[0], rh[1] - 0.25, rh[2]]
    if la is None:
        la = [lk[0], lk[1] - 0.28, lk[2] + 0.05]
    if ra is None:
        ra = [rk[0], rk[1] - 0.28, rk[2] - 0.05]
    if nose is None:
        nose = [chest[0], chest[1] + 0.25, chest[2]]

    x_axis = _unit(_vec(ls, rs))
    y_axis = _unit(_vec(pelvis, chest))
    z_axis = _unit([
        x_axis[1] * y_axis[2] - x_axis[2] * y_axis[1],
        x_axis[2] * y_axis[0] - x_axis[0] * y_axis[2],
        x_axis[0] * y_axis[1] - x_axis[1] * y_axis[0],
    ])
    y_axis = _unit([
        z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
        z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
        z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0],
    ])

    root_q = _quat_from_axes(x_axis, y_axis, z_axis)
    root_pose = [
        float(pelvis[0]),
        float(pelvis[1]),
        float(pelvis[2]),
        float(root_q[0]),
        float(root_q[1]),
        float(root_q[2]),
        float(root_q[3]),
    ]

    # Simple heuristic joint values.
    torso_yaw = math.atan2((rs[2] - ls[2]), (rs[0] - ls[0]) + 1e-8)
    torso_pitch = math.atan2((chest[1] - pelvis[1]), abs(chest[0] - pelvis[0]) + abs(chest[2] - pelvis[2]) + 1e-8)
    torso_roll = math.atan2((rs[1] - ls[1]), abs(rs[0] - ls[0]) + 1e-8)

    neck_yaw = 0.0
    neck_pitch = math.atan2((nose[1] - chest[1]), abs(nose[0] - chest[0]) + abs(nose[2] - chest[2]) + 1e-8)

    left_upper_arm = _vec(ls, le)
    left_lower_arm = _vec(le, lw)
    right_upper_arm = _vec(rs, re)
    right_lower_arm = _vec(re, rw)

    left_upper_leg = _vec(lh, lk)
    left_lower_leg = _vec(lk, la)
    right_upper_leg = _vec(rh, rk)
    right_lower_leg = _vec(rk, ra)

    left_elbow = _angle(left_upper_arm, left_lower_arm)
    right_elbow = _angle(right_upper_arm, right_lower_arm)
    left_knee = _angle(left_upper_leg, left_lower_leg)
    right_knee = _angle(right_upper_leg, right_lower_leg)

    left_shoulder_pitch = math.atan2(-left_upper_arm[1], abs(left_upper_arm[0]) + abs(left_upper_arm[2]) + 1e-8)
    left_shoulder_roll = math.atan2(left_upper_arm[2], abs(left_upper_arm[0]) + 1e-8)
    left_shoulder_yaw = math.atan2(left_upper_arm[0], abs(left_upper_arm[2]) + 1e-8)

    right_shoulder_pitch = math.atan2(-right_upper_arm[1], abs(right_upper_arm[0]) + abs(right_upper_arm[2]) + 1e-8)
    right_shoulder_roll = math.atan2(right_upper_arm[2], abs(right_upper_arm[0]) + 1e-8)
    right_shoulder_yaw = math.atan2(right_upper_arm[0], abs(right_upper_arm[2]) + 1e-8)

    left_hip_yaw = 0.0
    right_hip_yaw = 0.0
    left_hip_roll = math.atan2(left_upper_leg[2], abs(left_upper_leg[1]) + 1e-8)
    right_hip_roll = math.atan2(right_upper_leg[2], abs(right_upper_leg[1]) + 1e-8)
    left_hip_pitch = math.atan2(-left_upper_leg[1], abs(left_upper_leg[0]) + abs(left_upper_leg[2]) + 1e-8)
    right_hip_pitch = math.atan2(-right_upper_leg[1], abs(right_upper_leg[0]) + abs(right_upper_leg[2]) + 1e-8)

    left_ankle_pitch = math.atan2(-left_lower_leg[1], abs(left_lower_leg[0]) + abs(left_lower_leg[2]) + 1e-8)
    left_ankle_roll = math.atan2(left_lower_leg[2], abs(left_lower_leg[0]) + 1e-8)
    right_ankle_pitch = math.atan2(-right_lower_leg[1], abs(right_lower_leg[0]) + abs(right_lower_leg[2]) + 1e-8)
    right_ankle_roll = math.atan2(right_lower_leg[2], abs(right_lower_leg[0]) + 1e-8)

    qpos = [
        root_pose[0],
        root_pose[1],
        root_pose[2],
        root_pose[3],
        root_pose[4],
        root_pose[5],
        root_pose[6],
        torso_yaw,
        torso_pitch,
        torso_roll,
        neck_yaw,
        neck_pitch,
        left_shoulder_pitch,
        left_shoulder_roll,
        left_shoulder_yaw,
        left_elbow,
        right_shoulder_pitch,
        right_shoulder_roll,
        right_shoulder_yaw,
        right_elbow,
        left_hip_yaw,
        left_hip_roll,
        left_hip_pitch,
        left_knee,
        left_ankle_pitch,
        left_ankle_roll,
        right_hip_yaw,
        right_hip_roll,
        right_hip_pitch,
        right_knee,
        right_ankle_pitch,
        right_ankle_roll,
    ]

    time_ms = float(frame.get("timeMs", frame.get("t", 0.0)))
    unix_ms = int(frame.get("unixMs", 0))

    if STATE.prev_qpos is None or STATE.prev_time_ms is None:
        qvel = [0.0] * len(qpos)
    else:
        dt = max((time_ms - STATE.prev_time_ms) / 1000.0, 1e-3)
        qvel = [(qpos[i] - STATE.prev_qpos[i]) / dt for i in range(len(qpos))]

    STATE.prev_qpos = list(qpos)
    STATE.prev_time_ms = time_ms

    return {
        "frameIndex": frame.get("frameIndex"),
        "timeMs": time_ms,
        "unixMs": unix_ms,
        "root_pose": root_pose,
        "h1_qpos": qpos,
        "h1_qvel": qvel,
        "jointNames": H1_JOINT_NAMES,
        "debug": {
            "pelvis": pelvis,
            "chest": chest,
            "landmark_count": len(lm),
        },
    }