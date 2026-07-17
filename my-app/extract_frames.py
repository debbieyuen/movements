"""Extract still frames from a recorded clip (.webm/.mp4) for calibration.

Your phones download a video per session; pull frames from the board clip so
the calibrator can find the checkerboard in them.

Needs OpenCV:  python -m pip install opencv-python

Usage:
  python extract_frames.py <video> <out_dir> [--every N]
    --every N   save every Nth frame (default 15, ~2/sec at 30fps)
"""

import argparse
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--every", type=int, default=15, help="save every Nth frame")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open {args.video}")
        return

    i = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % args.every == 0:
            cv2.imwrite(str(out / f"{saved:04d}.png"), frame)
            saved += 1
        i += 1
    cap.release()
    print(f"Extracted {saved} frames (of {i}) -> {out}")


if __name__ == "__main__":
    main()
