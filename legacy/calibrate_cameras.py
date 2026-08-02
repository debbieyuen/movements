"""Multi-camera calibration from checkerboard images.

Computes, for each camera:
  - intrinsics: the lens (camera matrix K + distortion) -- from many board views.
  - extrinsics: the camera's pose (R, t) relative to a SHARED board position,
    which becomes the common world frame -- from one board view seen by all
    cameras at the same spot.

With K, dist, R, t per camera we can triangulate 2D joints from the 3 views
into real 3D (the next step, fuse_3d.py).

Needs OpenCV:  python -m pip install opencv-python

Board: print checkerboard_9x6_18mm.svg  =>  9x6 INNER corners.
IMPORTANT: printers rescale, so MEASURE one printed square with a ruler and set
SQUARE_MM below. That measurement sets the real-world scale of everything.

Folder layout (create these and drop your captured board frames in):
  calibration/
    intrinsics/
      front-camera/ *.png   # ~15-20 board views at VARIED angles, per camera
      left-phone/   *.png
      right-phone/  *.png
    extrinsics/
      front-camera.png      # ONE frame per camera of the board held STILL in
      left-phone.png        # the SAME spot, visible to all three at once
      right-phone.png
  (use extract_frames.py to pull these from the phone videos)

Run:    python calibrate_cameras.py
Output: calibration/calibration.json
"""

import glob
import json
from pathlib import Path

import cv2
import numpy as np

INNER = (9, 6)     # inner corners: (columns, rows)
SQUARE_MM = 18.0   # <-- MEASURE your printed square and set this!
CAL = Path("calibration")
CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def board_points() -> np.ndarray:
    """3D corner coordinates in the board's own frame (z=0 plane), in mm."""
    objp = np.zeros((INNER[0] * INNER[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:INNER[0], 0:INNER[1]].T.reshape(-1, 2)
    return objp * SQUARE_MM


def find_corners(gray: np.ndarray):
    ok, corners = cv2.findChessboardCorners(
        gray, INNER,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRIT)


def calibrate_intrinsics(role: str):
    files = sorted(glob.glob(str(CAL / "intrinsics" / role / "*.png")) +
                   glob.glob(str(CAL / "intrinsics" / role / "*.jpg")))
    objpts, imgpts, size, used = [], [], None, 0
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]  # (w, h)
        c = find_corners(gray)
        if c is None:
            continue
        objpts.append(board_points())
        imgpts.append(c)
        used += 1
    if used < 8:
        raise RuntimeError(
            f"{role}: only {used} usable board views (want >= ~10-15). "
            "Capture more with the board at varied angles/distances.")
    rms, K, dist, _, _ = cv2.calibrateCamera(objpts, imgpts, size, None, None)
    print(f"  {role:14s}: intrinsics from {used} views, reproj RMS={rms:.3f}px "
          + ("(good)" if rms < 1.0 else "(HIGH - recapture for better accuracy)"))
    return K, dist, list(size)


def extrinsics(role: str, K: np.ndarray, dist: np.ndarray):
    path = None
    for ext in ("png", "jpg"):
        p = CAL / "extrinsics" / f"{role}.{ext}"
        if p.exists():
            path = str(p)
            break
    if path is None:
        raise RuntimeError(f"{role}: missing calibration/extrinsics/{role}.png")
    gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    c = find_corners(gray)
    if c is None:
        raise RuntimeError(f"{role}: board not found in extrinsics frame {path} "
                           "(make sure the whole board is visible + in focus)")
    ok, rvec, tvec = cv2.solvePnP(board_points(), c, K, dist)
    if not ok:
        raise RuntimeError(f"{role}: solvePnP failed")
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec


def main():
    intr_dir = CAL / "intrinsics"
    if not intr_dir.exists():
        print("No calibration/intrinsics/ folder. See the header for the layout.")
        return
    roles = sorted(p.name for p in intr_dir.iterdir() if p.is_dir())
    if not roles:
        print("No camera subfolders under calibration/intrinsics/.")
        return

    out = {"inner_corners": list(INNER), "square_mm": SQUARE_MM,
           "world_frame": "shared checkerboard", "cameras": {}}
    for role in roles:
        K, dist, size = calibrate_intrinsics(role)
        R, t = extrinsics(role, K, dist)
        out["cameras"][role] = {
            "image_size": size,
            "K": K.tolist(),
            "dist": np.asarray(dist).ravel().tolist(),
            "R": R.tolist(),
            "t": np.asarray(t).ravel().tolist(),
        }

    CAL.mkdir(exist_ok=True)
    (CAL / "calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {CAL / 'calibration.json'}  ({len(out['cameras'])} cameras)")


if __name__ == "__main__":
    main()
