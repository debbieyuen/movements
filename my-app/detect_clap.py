"""Detect the sync clap in each camera's recorded stream and compute per-camera
time offsets, so the multi-camera streams can be aligned to a common timeline.

The clap is a physical event every camera sees at the same real instant. We find
it as the sharp MINIMUM in the distance between the left and right wrists (hands
together). Because it lives inside each camera's own data, it is immune to network
lag and clock skew -- detecting it in every stream tells us each camera's clock
offset directly (this is the film "clapperboard" trick).

Usage:
    python detect_clap.py [sessionId]
If sessionId is omitted, the session of the most recent latest_frame.json is used.

Reads received_frames/frames_<role>.jsonl (filtered to the session) and writes
received_frames/sync_<sessionId>.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

RF = Path("received_frames")
L_WR, R_WR = 15, 16  # MediaPipe wrist landmark indices

# --- Tuning knobs ------------------------------------------------------------
SMOOTH_FRAMES = 3      # moving-average window to tame per-frame jitter
CLAP_MAX_DIST = 0.20   # meters; a real clap min should be at/under this
MIN_PROMINENCE = 0.40  # (baseline - min) / baseline; how far the dip stands out


def _wrist_dist(frame: dict):
    pts = frame.get("worldLandmarks") or frame.get("landmarks")
    if not (isinstance(pts, list) and len(pts) > R_WR):
        return None
    a, b = pts[L_WR], pts[R_WR]
    try:
        return float(np.linalg.norm(
            [a["x"] - b["x"], a["y"] - b["y"], a["z"] - b["z"]]))
    except (KeyError, TypeError):
        return None


def load_camera(path: Path, session_id):
    ts, dist = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                fr = json.loads(line)
            except json.JSONDecodeError:
                continue
            if session_id and fr.get("sessionId") != session_id:
                continue
            d = _wrist_dist(fr)
            t = fr.get("timeMs")
            if d is None or t is None:
                continue
            ts.append(float(t))
            dist.append(d)
    if not ts:
        return None
    ts, dist = np.array(ts), np.array(dist)
    order = np.argsort(ts)
    return ts[order], dist[order]


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def detect_clap(ts: np.ndarray, dist: np.ndarray) -> dict:
    d = _smooth(dist, SMOOTH_FRAMES)
    i = int(np.argmin(d))
    baseline = float(np.median(d))
    dmin = float(d[i])
    prominence = (baseline - dmin) / baseline if baseline > 1e-6 else 0.0
    span = float(ts[-1] - ts[0]) or 1.0
    return {
        "clap_timeMs": float(ts[i]),
        "min_dist_m": round(dmin, 3),
        "baseline_m": round(baseline, 3),
        "prominence": round(prominence, 2),
        "pct_into_clip": round(100.0 * (ts[i] - ts[0]) / span, 1),
        "confident": bool(dmin <= CLAP_MAX_DIST and prominence >= MIN_PROMINENCE),
        "n_frames": int(len(ts)),
    }


def main():
    session_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not session_id:
        lf = RF / "latest_frame.json"
        if lf.exists():
            session_id = json.loads(lf.read_text()).get("sessionId")
    print(f"Session: {session_id}\n")

    cameras = {}
    for path in sorted(RF.glob("frames_*.jsonl")):
        role = path.stem[len("frames_"):]
        loaded = load_camera(path, session_id)
        if loaded is None:
            print(f"  {role:14s}: no frames for this session")
            continue
        res = detect_clap(*loaded)
        cameras[role] = res
        flag = "CLAP OK" if res["confident"] else "LOW CONFIDENCE (no clean clap?)"
        print(f"  {role:14s}: clap@ {res['clap_timeMs']:.0f}ms  "
              f"min={res['min_dist_m']}m  prominence={res['prominence']}  "
              f"({res['pct_into_clip']}% in)  [{flag}]")

    confident = {r: c for r, c in cameras.items() if c["confident"]}
    if len(confident) >= 2:
        print("\nAlignment -- subtract each camera's clap_timeMs to put them on a"
              " shared timeline:")
        for role, c in confident.items():
            print(f"  {role:14s}: aligned_t = timeMs - {c['clap_timeMs']:.0f}")
    elif cameras:
        print("\n(Need >= 2 cameras with a confident clap to align. With one "
              "camera there is nothing to align to yet.)")

    if session_id and cameras:
        out = RF / f"sync_{session_id}.json"
        out.write_text(json.dumps(
            {"sessionId": session_id, "cameras": cameras}, indent=2), encoding="utf-8")
        print(f"\nSaved sync report -> {out}")


if __name__ == "__main__":
    main()
