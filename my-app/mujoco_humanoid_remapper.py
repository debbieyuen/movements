import json
import time
from pathlib import Path
from typing import Any, Dict, List

SOURCE_FILE = Path("received_frames/latest_h1_frame.json")
OUTPUT_FILE = Path("received_frames/latest_mujoco_frame.json")

# Approximate Gymnasium / MuJoCo Humanoid-v5 joint targets.
# This is a visualization mapping, not an anatomically exact retargeting.
# Humanoid-v5 joint order (17 actuated DOFs):
#   abdomen_y, abdomen_z, abdomen_x,
#   right_hip_x, right_hip_z, right_hip_y, right_knee,
#   left_hip_x, left_hip_z, left_hip_y, left_knee,
#   right_shoulder1, right_shoulder2, right_elbow,
#   left_shoulder1, left_shoulder2, left_elbow
HUMANOID_JOINT_NAMES = [
    "abdomen_y",
    "abdomen_z",
    "abdomen_x",
    "right_hip_x",
    "right_hip_z",
    "right_hip_y",
    "right_knee",
    "left_hip_x",
    "left_hip_z",
    "left_hip_y",
    "left_knee",
    "right_shoulder1",
    "right_shoulder2",
    "right_elbow",
    "left_shoulder1",
    "left_shoulder2",
    "left_elbow",
]


def _get(values: List[float], idx: int, default: float = 0.0) -> float:
    if idx < 0 or idx >= len(values):
        return default
    try:
        return float(values[idx])
    except Exception:
        return default



def _clamp(v: float, lo: float = -2.5, hi: float = 2.5) -> float:
    return max(lo, min(hi, v))



def pose_to_mujoco(frame: Dict[str, Any]) -> Dict[str, Any]:
    h1_qpos = frame.get("h1_qpos", [])
    h1_qvel = frame.get("h1_qvel", [])
    root_pose = frame.get("root_pose", [0, 0, 1.4, 1, 0, 0, 0])

    # Map H1-style joint values into the Humanoid-v5 ordering.
    mujoco_qpos = [
        _get(h1_qpos, 8),   # torso_pitch -> abdomen_y
        _get(h1_qpos, 7),   # torso_yaw   -> abdomen_z
        _get(h1_qpos, 9),   # torso_roll  -> abdomen_x

        _get(h1_qpos, 27),  # right_hip_roll  -> right_hip_x
        _get(h1_qpos, 26),  # right_hip_yaw   -> right_hip_z
        _get(h1_qpos, 28),  # right_hip_pitch -> right_hip_y
        _get(h1_qpos, 29),  # right_knee      -> right_knee

        _get(h1_qpos, 21),  # left_hip_roll   -> left_hip_x (approx)
        _get(h1_qpos, 20),  # left_hip_yaw    -> left_hip_z
        _get(h1_qpos, 22),  # left_hip_pitch  -> left_hip_y
        _get(h1_qpos, 23),  # left_knee       -> left_knee

        _get(h1_qpos, 16),  # right_shoulder_pitch -> right_shoulder1
        _get(h1_qpos, 17),  # right_shoulder_roll  -> right_shoulder2
        _get(h1_qpos, 19),  # right_elbow          -> right_elbow

        _get(h1_qpos, 12),  # left_shoulder_pitch  -> left_shoulder1
        _get(h1_qpos, 13),  # left_shoulder_roll   -> left_shoulder2
        _get(h1_qpos, 15),  # left_elbow           -> left_elbow
    ]

    mujoco_qvel = [
        _get(h1_qvel, 8),
        _get(h1_qvel, 7),
        _get(h1_qvel, 9),

        _get(h1_qvel, 27),
        _get(h1_qvel, 26),
        _get(h1_qvel, 28),
        _get(h1_qvel, 29),

        _get(h1_qvel, 21),
        _get(h1_qvel, 20),
        _get(h1_qvel, 22),
        _get(h1_qvel, 23),

        _get(h1_qvel, 16),
        _get(h1_qvel, 17),
        _get(h1_qvel, 19),

        _get(h1_qvel, 12),
        _get(h1_qvel, 13),
        _get(h1_qvel, 15),
    ]

    return {
        "frameIndex": frame.get("frameIndex"),
        "timeMs": frame.get("timeMs"),
        "unixMs": frame.get("unixMs"),
        "serverUnixMs": frame.get("serverUnixMs"),
        "root_pose": root_pose,
        "mujoco_joint_names": HUMANOID_JOINT_NAMES,
        "mujoco_qpos": [_clamp(v) for v in mujoco_qpos],
        "mujoco_qvel": [_clamp(v) for v in mujoco_qvel],
    }



def main() -> None:
    print(f"Watching {SOURCE_FILE}")
    last_mtime = 0.0

    while True:
        if SOURCE_FILE.exists():
            mtime = SOURCE_FILE.stat().st_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                try:
                    frame = json.loads(SOURCE_FILE.read_text(encoding="utf-8"))
                    out = pose_to_mujoco(frame)
                    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
                    OUTPUT_FILE.write_text(json.dumps(out, indent=2), encoding="utf-8")
                    print(f"Wrote {OUTPUT_FILE} (frame {out.get('frameIndex')})")
                except Exception as e:
                    print("Remap error:", e)

        time.sleep(1 / 30)


if __name__ == "__main__":
    main()
