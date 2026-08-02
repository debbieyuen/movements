"""Package a fused capture into a GPU-ready dataset for the world model.

For a session it writes dataset/<sessionId>/:
  data.npz  -> timestamps, landmarks_3d (T,33,3), qpos (T,nq), qvel (T,nv),
               landmarks_2d_<role> (T,33,3) for each camera
  meta.json -> session, height, fps, joint names, units, calibration note

qpos is the robot's full state (root pose + joints), retargeted from the FUSED
3D by the same IK the live robot uses -- i.e. the "motion data" that used to be
computed and discarded. Everything is re-runnable: improve calibration/fusion,
re-run, same outputs.

Usage: python dataset_builder.py [sessionId]
"""

import json
import sys
from pathlib import Path

import numpy as np
import mujoco

import play_latest_h1_mujoco_ik as P
P.DEPTH_SCALE = 1.0  # fused 3D already has real depth -- don't compress it

RF = Path("received_frames")
OUT = Path("dataset")
HEIGHT_M = 1.5748


def load_2d(role, sid, clap_ms):
    out = []
    p = RF / f"frames_{role}.jsonl"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                fr = json.loads(line)
            except json.JSONDecodeError:
                continue
            if fr.get("sessionId") != sid:
                continue
            lm = fr.get("landmarks")
            if not (isinstance(lm, list) and len(lm) >= 33):
                continue
            A = np.array([[q.get("x", 0), q.get("y", 0), q.get("visibility", 0)]
                          for q in lm[:33]], float)
            out.append((fr.get("timeMs", 0) - clap_ms, A))
    if not out:
        return None
    out.sort(key=lambda x: x[0])
    return np.array([t for t, _ in out]), np.array([a for _, a in out])


def resample_2d(loaded, grid, tol=40.0):
    """Nearest-neighbor resample a camera's 2D landmarks onto the fused grid."""
    T = len(grid)
    R = np.full((T, 33, 3), np.nan)
    if loaded is None:
        return R
    ts, A = loaded
    for i, t in enumerate(grid):
        j = int(np.argmin(np.abs(ts - t)))
        if abs(ts[j] - t) < tol:
            R[i] = A[j]
    return R


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else \
        json.load(open(RF / "latest_frame.json"))["sessionId"]
    fused = np.load(RF / f"fused_3d_{sid}.npz")
    ts, P3 = fused["timestamps"], fused["landmarks_3d"]
    T = len(P3)
    print(f"session {sid}: {T} fused frames")

    # --- robot qpos from the fused 3D (same IK as the live robot) ---
    model = mujoco.MjModel.from_xml_path(P._h1_model_path())
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    tmpl = np.array(data.qpos, float).copy()
    tmpl[0:3] = [0, 0, P.STANDING_HEIGHT]
    tmpl[3:7] = [1, 0, 0, 0]
    ik = P.H1IK(model)

    qpos = np.zeros((T, model.nq))
    last = tmpl.copy()
    solved = 0
    for i in range(T):
        frame = {"worldLandmarks": [
            {"x": float(P3[i, j, 0]), "y": float(P3[i, j, 1]),
             "z": float(P3[i, j, 2]), "visibility": 1.0} for j in range(33)]}
        q, _ = P.build_qpos(ik, tmpl, frame)
        if q is not None:
            last = q
            solved += 1
        qpos[i] = last
    print(f"IK solved {solved}/{T} frames")

    # qvel via MuJoCo finite difference (correct for the free-joint quaternion)
    qvel = np.zeros((T, model.nv))
    for i in range(1, T):
        dt = max((ts[i] - ts[i - 1]) / 1000.0, 1e-3)
        v = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, v, dt, qpos[i - 1], qpos[i])
        qvel[i] = v

    # --- per-camera 2D observations, resampled onto the fused timeline ---
    sync = json.load(open(RF / f"sync_{sid}.json"))
    clap = {r: c["clap_timeMs"] for r, c in sync["cameras"].items()}
    lm2d = {r: resample_2d(load_2d(r, sid, clap[r]), ts) for r in clap}

    jnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
              for j in range(model.njnt)]

    OUT.mkdir(exist_ok=True)
    sd = OUT / sid
    sd.mkdir(exist_ok=True)
    arrays = {"timestamps": ts, "landmarks_3d": P3, "qpos": qpos, "qvel": qvel}
    for r, A in lm2d.items():
        arrays[f"landmarks_2d_{r.replace('-', '_')}"] = A
    np.savez(sd / "data.npz", **arrays)

    meta = {
        "sessionId": sid, "n_frames": int(T), "fps_nominal": 30,
        "height_m": HEIGHT_M, "robot": "Unitree H1",
        "nq": int(model.nq), "nv": int(model.nv), "joint_names": jnames,
        "qpos_layout": "free base [px,py,pz, qw,qx,qy,qz] then 19 hinge joints",
        "units": {"landmarks_3d": "meters, hip-centered",
                  "landmarks_2d": "normalized image x,y + visibility",
                  "qpos": "meters + radians", "qvel": "per-second",
                  "timestamps": "ms since clap"},
        "cameras": list(clap),
        "calibration": "body-based multi-camera fusion (ROUGH; re-run after upgrade)",
        "notes": "root/torso currently pinned in qpos (unfreeze is a later step)",
    }
    (sd / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nsaved dataset -> {sd}/")
    print(f"  data.npz: landmarks_3d {P3.shape}, qpos {qpos.shape}, "
          f"qvel {qvel.shape}, + 2D for {list(clap)}")
    print(f"  meta.json")


if __name__ == "__main__":
    main()
