"""Body-based multi-camera 3D fusion (no checkerboard).

Your body is the calibration reference. Each camera gives a rough 3D skeleton
(MediaPipe worldLandmarks); the clap synced them, so the same pose is seen from
3 angles. We estimate the rotation between cameras (Kabsch on your joints across
many frames), rotate all skeletons into one common frame, and blend them
(visibility-weighted). Because each camera's depth error points a different way,
blending cancels much of it -> cleaner full-body 3D than any single view.

Output: a fused, hip-centered 3D pose per frame -- what the robot IK and a motion
dataset need (pose/limb directions, root handled separately). Scale from height.

Rough by design; upgrade later (big checkerboard or clean full-body clip) and
re-run -- downstream tools read the same output.

Usage: python body_fuse_3d.py [sessionId]
Output: received_frames/fused_3d_<sessionId>.npz  (timestamps, landmarks_3d)
"""

import json
import sys
from pathlib import Path

import numpy as np

RF = Path("received_frames")
HEIGHT_M = 1.5748       # 5 ft 2 in
REF = "front-camera"    # common frame everything rotates into
VIS_MIN = 0.4           # min avg visibility to use a frame for rotation fitting
VIS_JOINT = 0.5         # per-joint gate: ignore a camera's joint below this
SYNC_TOL_MS = 40.0
GRID_MS = 33.33         # ~30 fps common timeline

# Bones for the length-consistency quality check (MediaPipe indices).
BONES = [("L-upperarm", 11, 13), ("L-forearm", 13, 15),
         ("R-upperarm", 12, 14), ("R-forearm", 14, 16),
         ("L-thigh", 23, 25), ("L-shin", 25, 27),
         ("R-thigh", 24, 26), ("R-shin", 26, 28)]


def load(role, sid, clap_ms):
    out = []
    p = RF / f"frames_{role}.jsonl"
    if not p.exists():
        return out
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
            wl = fr.get("worldLandmarks")
            if not (isinstance(wl, list) and len(wl) >= 33):
                continue
            P = np.array([[q.get("x", 0), q.get("y", 0), q.get("z", 0)]
                          for q in wl[:33]], float)
            V = np.array([q.get("visibility", 0) for q in wl[:33]], float)
            out.append((fr.get("timeMs", 0) - clap_ms, P, V))
    out.sort(key=lambda x: x[0])
    return out


def center(P, w):
    W = w / w.sum()
    return P - (P * W[:, None]).sum(0)


def kabsch_R(A, B, w):
    """Best rotation mapping centered A onto centered B (weighted)."""
    W = w / w.sum()
    H = (A * W[:, None]).T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def smooth_time(P3, win=11, poly=2):
    """Low-pass each joint/axis over time to reduce jitter. Savitzky-Golay if
    available (shape-preserving), else a moving average."""
    T = P3.shape[0]
    if T < win:
        return P3
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(P3, win, poly, axis=0)
    except Exception:
        k = np.ones(win) / win
        out = P3.copy()
        for j in range(P3.shape[1]):
            for a in range(3):
                out[:, j, a] = np.convolve(P3[:, j, a], k, mode="same")
        return out


def bone_cv(P3, bones):
    cvs = []
    for name, a, b in bones:
        L = np.linalg.norm(P3[:, a] - P3[:, b], axis=1)
        cv = L.std() / L.mean() if L.mean() > 0 else 0.0
        cvs.append((name, L.mean(), cv))
    return cvs


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else \
        json.load(open(RF / "latest_frame.json")).get("sessionId")
    sync = json.load(open(RF / f"sync_{sid}.json"))
    clap = {r: c["clap_timeMs"] for r, c in sync["cameras"].items()}
    print(f"Session {sid}, cameras: {list(clap)}")

    cams = {r: load(r, sid, clap[r]) for r in clap}
    cams = {r: d for r, d in cams.items() if d}
    if REF not in cams:
        print(f"No {REF} data; aborting.")
        return

    # per-camera time arrays for fast nearest lookup
    tarr = {r: np.array([d[0] for d in cams[r]]) for r in cams}

    def nearest(role, t):
        i = int(np.argmin(np.abs(tarr[role] - t)))
        return cams[role][i] if abs(tarr[role][i] - t) < SYNC_TOL_MS else None

    lo = max(d[0][0] for d in cams.values())
    hi = min(d[-1][0] for d in cams.values())
    grid = np.arange(lo, hi, GRID_MS)
    print(f"overlap {lo:.0f}..{hi:.0f} ms, {len(grid)} grid times")

    # 1) relative rotation: each other camera -> REF
    R = {REF: np.eye(3)}
    for r in [x for x in cams if x != REF]:
        A, B, W = [], [], []
        for t in grid:
            f = nearest(REF, t)
            o = nearest(r, t)
            if not (f and o):
                continue
            w = np.minimum(f[2], o[2])
            if w.mean() < VIS_MIN:
                continue
            A.append(center(o[1], w))
            B.append(center(f[1], w))
            W.append(w)
        if not A:
            print(f"  {r}: no synced frames -> dropped")
            continue
        R[r] = kabsch_R(np.vstack(A), np.vstack(B), np.concatenate(W))
        print(f"  {r}: rotation to {REF} from {len(A)} frames")

    # 2) fuse per grid time (need >= 2 cameras)
    ts, P3 = [], []
    for t in grid:
        Ps, Ws = [], []
        for r in cams:
            if r not in R:
                continue
            s = nearest(r, t)
            if not s:
                continue
            Ps.append((R[r] @ center(s[1], s[2]).T).T)
            Ws.append(s[2])
        if len(Ps) < 2:
            continue
        Ps = np.array(Ps)
        Wv = np.array(Ws)                       # (C, 33)
        # per-joint gate: drop a camera's joint if it barely sees it; but if
        # NO camera clears the gate for a joint, fall back to raw weights.
        gated = np.where(Wv >= VIS_JOINT, Wv, 0.0)
        gated = np.where(gated.sum(0)[None, :] < 1e-6, Wv, gated)
        w = gated[:, :, None]
        fused = (Ps * w).sum(0) / np.clip(w.sum(0), 1e-6, None)
        ts.append(t)
        P3.append(fused)
    P3 = np.array(P3)
    print(f"fused {len(P3)} frames (>=2 cameras each)")
    if len(P3) == 0:
        print("nothing fused; aborting.")
        return

    # 3) scale to real size via nose(0)-to-mid-ankle (~0.87 * height)
    nose = P3[:, 0]
    ankle = (P3[:, 27] + P3[:, 28]) / 2
    med = np.median(np.linalg.norm(nose - ankle, axis=1))
    scale = (0.87 * HEIGHT_M) / med if med > 1e-6 else 1.0
    P3 *= scale
    print(f"scaled x{scale:.3f} (nose-ankle {med:.3f} -> {0.87*HEIGHT_M:.3f} m)")

    # Save the body-based calibration (per-camera rotation into REF frame +
    # scale) so the live fused player can reuse it without recomputing.
    calib = {"ref": REF, "height_m": HEIGHT_M, "scale": float(scale),
             "rotations": {r: R[r].tolist() for r in R}}
    (RF / "calibration_body.json").write_text(
        json.dumps(calib, indent=2), encoding="utf-8")
    print(f"saved calibration -> {RF / 'calibration_body.json'}")

    # 4) sharpen: temporal smoothing, and report bone-length CV before/after
    raw_cv = np.median([cv for _, _, cv in bone_cv(P3, BONES)])
    P3s = smooth_time(P3)
    print("\nbone-length consistency (CV, lower=better; <0.15 is decent):")
    cvs = bone_cv(P3s, BONES)
    for name, mean_len, cv in cvs:
        print(f"  {name:12s}: {mean_len*100:5.1f} cm   CV={cv:.2f}")
    smoothed_cv = np.median([cv for _, _, cv in cvs])
    print(f"  --> median CV: raw={raw_cv:.2f}  smoothed={smoothed_cv:.2f}")

    out = RF / f"fused_3d_{sid}.npz"
    np.savez(str(out), timestamps=np.array(ts), landmarks_3d=P3s)
    print(f"\nsaved -> {out}  shape {P3s.shape}")


if __name__ == "__main__":
    main()
