"""GVHMR wrapper: video -> world-grounded SMPL motion.

GVHMR (https://github.com/zju3dv/GVHMR) recovers gravity-aligned world-frame
human motion from a single video; with a static camera (`-s`) it needs no
visual odometry. It runs in ITS OWN environment (torch pins differ from ours),
so this module shells out rather than importing it.

Environment knobs (all optional):
  MOCAP_GVHMR_DIR     path to the GVHMR clone      (default: external/GVHMR)
  MOCAP_GVHMR_PYTHON  python executable of its env (default: "python" via
                      `conda run -n gvhmr` if MOCAP_GVHMR_CONDA_ENV is set)
  MOCAP_GVHMR_CONDA_ENV  conda env name to run inside

⚠ Licensing: GVHMR code is research/non-commercial (ZJU license) and SMPL
body models are MPI research-only. Fine for research use; disclose to the
team and swap the backend (WHAM is MIT) if that ever changes.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTER = Path(__file__).resolve().parent / "exporters" / "gvhmr_export.py"


def _gvhmr_dir() -> Path:
    return Path(os.environ.get("MOCAP_GVHMR_DIR", REPO_ROOT / "external" / "GVHMR"))


def _runner() -> list[str]:
    """Command prefix that executes python inside the GVHMR environment."""
    if env := os.environ.get("MOCAP_GVHMR_CONDA_ENV"):
        return ["conda", "run", "--no-capture-output", "-n", env, "python"]
    return [os.environ.get("MOCAP_GVHMR_PYTHON", "python")]


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"[recovery] $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def git_commit(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_gvhmr(video: Path, work_dir: Path, *, static_camera: bool = True) -> Path:
    """Run GVHMR's demo pipeline; returns the hmr4d_results.pt path.

    Skips the run if the output already exists in work_dir (stage caching).
    """
    out_pt = work_dir / "hmr4d_results.pt"
    if out_pt.exists():
        print(f"[recovery] cached: {out_pt}")
        return out_pt

    gvhmr = _gvhmr_dir()
    demo = gvhmr / "tools" / "demo" / "demo.py"
    if not demo.exists():
        raise FileNotFoundError(
            f"GVHMR not found at {gvhmr} (see docs/GPU_SETUP.md; "
            f"set MOCAP_GVHMR_DIR if it lives elsewhere)")

    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [*_runner(), str(demo), "--video", str(video.resolve()),
           "--output_root", str(work_dir.resolve())]
    if static_camera:
        cmd.append("-s")
    _run(cmd, cwd=gvhmr)

    # GVHMR writes output_root/<video_stem>/hmr4d_results.pt
    candidates = sorted(work_dir.rglob("hmr4d_results.pt"))
    if not candidates:
        raise RuntimeError(f"GVHMR produced no hmr4d_results.pt under {work_dir}")
    if candidates[0] != out_pt:
        candidates[0].replace(out_pt)
    return out_pt


def collect_render_videos(work_dir: Path, clip_dir: Path) -> list[Path]:
    """Copy GVHMR's own overlay videos into the clip directory.

    The demo renders the fitted body back over the footage (in-camera view)
    and from a free viewpoint (global view). They are the fastest way to see
    whether GVHMR actually tracked the person, so they belong next to the clip
    rather than buried in intermediate/.
    """
    import shutil

    out: list[Path] = []
    for src in sorted(work_dir.rglob("*.mp4")):
        if src.parent == clip_dir:
            continue
        dst = clip_dir / f"gvhmr_{src.stem}.mp4"
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            clip_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        out.append(dst)
    if out:
        print(f"[recovery] kept {len(out)} gvhmr render(s): "
              f"{', '.join(p.name for p in out)}")
    return out


def export_smpl_npz(pt_path: Path, out_npz: Path, video_fps: float) -> Path:
    """Extract SMPL params + world joints from hmr4d_results.pt into a plain
    npz (z-up), using the exporter script run INSIDE the GVHMR env (it needs
    torch + the SMPL body model)."""
    if out_npz.exists():
        print(f"[recovery] cached: {out_npz}")
        return out_npz
    cmd = [*_runner(), str(EXPORTER), str(pt_path), str(out_npz), str(video_fps)]
    _run(cmd, cwd=_gvhmr_dir())
    if not out_npz.exists():
        raise RuntimeError("SMPL export produced no output")
    return out_npz
