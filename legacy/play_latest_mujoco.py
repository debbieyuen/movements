import json
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import gymnasium as gym
import numpy as np

MOTION_FILE = Path('received_frames/latest_mujoco_frame.json')


def _as_float_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1)


def apply_frame(env, template_qpos, template_qvel, frame: Dict[str, Any]):
    qpos = template_qpos.copy()
    qvel = template_qvel.copy()

    # Keep the model upright and at a fixed standing height.
    qpos[0:3] = np.array([0.0, 0.0, 1.35], dtype=np.float32)
    qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Map the live MuJoCo joint targets into the Humanoid-v5 state.
    targets = _as_float_array(frame.get('mujoco_qpos', []))
    n = min(max(len(qpos) - 7, 0), targets.shape[0])
    if n > 0:
        qpos[7:7 + n] = targets[:n]

    # Optional velocities. If they are missing, keep them zero.
    qvel[0:6] = 0.0
    vels = _as_float_array(frame.get('mujoco_qvel', []))
    nv = min(max(len(qvel) - 6, 0), vels.shape[0])
    if nv > 0:
        qvel[6:6 + nv] = vels[:nv]

    try:
        env.unwrapped.set_state(qpos, qvel)
    except Exception:
        env.unwrapped.data.qpos[:] = qpos
        env.unwrapped.data.qvel[:] = qvel
        try:
            import mujoco
            mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
        except Exception:
            pass


def main():
    env = gym.make("Humanoid-v5", render_mode="human")
    env.reset(seed=0)

    template_qpos = np.array(env.unwrapped.data.qpos, dtype=np.float32).copy()
    template_qvel = np.array(env.unwrapped.data.qvel, dtype=np.float32).copy()

    # Start upright in a neutral pose.
    start_qpos = template_qpos.copy()
    start_qvel = template_qvel.copy()
    start_qpos[0:3] = np.array([0.0, 0.0, 1.35], dtype=np.float32)
    start_qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    env.unwrapped.set_state(start_qpos, start_qvel)

    print(f"Watching {MOTION_FILE}")

    # Do not apply the already-existing file immediately on startup.
    last_mtime = MOTION_FILE.stat().st_mtime if MOTION_FILE.exists() else 0.0

    try:
        while True:
            if MOTION_FILE.exists():
                mtime = MOTION_FILE.stat().st_mtime
                if mtime != last_mtime:
                    last_mtime = mtime
                    try:
                        frame = json.loads(MOTION_FILE.read_text(encoding="utf-8"))
                        apply_frame(env, template_qpos, template_qvel, frame)
                        print(
                            "Applied frame",
                            frame.get("frameIndex"),
                            "timeMs=",
                            frame.get("timeMs"),
                        )
                    except Exception as e:
                        print("Playback error:", e)

            env.render()
            time.sleep(1 / 30)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == '__main__':
    main()
