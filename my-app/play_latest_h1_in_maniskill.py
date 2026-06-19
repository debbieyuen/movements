import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

MOTION_FILE = Path("received_frames/latest_h1_frame.json")


def get_robot(env):
    agent = getattr(env.unwrapped, "agent", None)
    if agent is None:
        raise RuntimeError("Could not find env.unwrapped.agent")

    for attr in ("robot", "articulation"):
        obj = getattr(agent, attr, None)
        if obj is not None:
            return obj

    raise RuntimeError("Could not find robot/articulation on env.unwrapped.agent")


def adapt_qpos(robot, source_qpos):
    current = np.asarray(robot.get_qpos(), dtype=np.float32).copy()
    source = np.asarray(source_qpos, dtype=np.float32).reshape(-1)

    n = min(current.shape[0], source.shape[0])
    current[:n] = source[:n]

    if current.shape[0] != source.shape[0]:
        print(f"qpos length mismatch: env={current.shape[0]} file={source.shape[0]} using={n}")

    return current


def adapt_qvel(robot, source_qvel):
    current = np.asarray(robot.get_qvel(), dtype=np.float32).copy()
    source = np.asarray(source_qvel, dtype=np.float32).reshape(-1)

    n = min(current.shape[0], source.shape[0])
    current[:n] = source[:n]
    return current


def main():
    env = gym.make("UnitreeH1Stand-v1", render_mode="human")
    env.reset()

    robot = get_robot(env)

    last_mtime = 0.0
    print(f"Watching {MOTION_FILE}")

    try:
        while True:
            if MOTION_FILE.exists():
                mtime = MOTION_FILE.stat().st_mtime
                if mtime != last_mtime:
                    last_mtime = mtime

                    frame = json.loads(MOTION_FILE.read_text(encoding="utf-8"))

                    if "h1_qpos" in frame:
                        qpos = adapt_qpos(robot, frame["h1_qpos"])
                        robot.set_qpos(qpos)

                    if "h1_qvel" in frame:
                        try:
                            qvel = adapt_qvel(robot, frame["h1_qvel"])
                            robot.set_qvel(qvel)
                        except Exception:
                            pass

                    print(
                        "Applied frame",
                        frame.get("frameIndex"),
                        "timeMs=",
                        frame.get("timeMs"),
                    )

            env.render()
            time.sleep(1 / 30)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()