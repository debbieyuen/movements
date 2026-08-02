"""One-command offline pipeline: video -> dataset/<clip_id>/.

    python -m mocap.process capture.mp4 --subject-height 1.57

Stages (each cached in <clip>/intermediate/, re-run with --force):
    1. GVHMR   video -> hmr4d_results.pt          (GPU, GVHMR env)
    2. export  .pt -> smpl_world.npz (z-up)       (GVHMR env)
    3. GMR     .pt -> gmr_unitree_h1.pkl          (CPU, GMR env)
    4. postprocess + save data.npz/meta.json      (this env)
    5. preview.mp4                                 (this env)
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from . import recovery, retarget
from .conventions import TARGET_FPS, load_h1_model
from .physics import rollout
from .postprocess import postprocess
from .render import EGO_CAM, render_egocentric, render_preview
from .schema import build_meta, save_clip, validate_clip

REPO_ROOT = Path(__file__).resolve().parents[1]


def probe_video(video: Path) -> dict:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "v:0", str(video)],
            check=True, capture_output=True, text=True).stdout
        stream = json.loads(out)["streams"][0]
        num, den = stream.get("avg_frame_rate", "30/1").split("/")
        fps = float(num) / float(den) if float(den) else 30.0
        return {
            "filename": video.name,
            "fps": round(fps, 3),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "duration_s": round(float(stream.get("duration", 0.0)), 2),
            "camera_static": True,
        }
    except Exception:  # noqa: BLE001 - ffprobe missing/odd container
        return {"filename": video.name, "fps": 30.0, "camera_static": True}


def load_annotation(video: Path) -> dict:
    """Read the `<stem>.annotation.json` sidecar the capture app uploads next
    to each recording (motion label + notes). Missing file is fine."""
    sidecar = video.with_suffix(".annotation.json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 - a broken sidecar must not block a clip
        print(f"[process] could not read {sidecar.name}: {e!r}")
        return {}


def resample_to(t_dst: np.ndarray, src_fps: float, arr: np.ndarray) -> np.ndarray:
    """Linear time-resample of an (T, ...) float array onto t_dst."""
    T = len(arr)
    t_src = np.arange(T) / src_fps
    flat = arr.reshape(T, -1)
    out = np.stack(
        [np.interp(t_dst, t_src, flat[:, k]) for k in range(flat.shape[1])], axis=1)
    return out.reshape(len(t_dst), *arr.shape[1:])


def process(video: Path, out_root: Path, *, clip_id: str | None,
            subject_height: float, subject_id: str, force: bool,
            label: str | None = None, notes: str | None = None,
            physics: bool = True, egocentric: bool = True) -> Path:
    clip_id = clip_id or video.stem
    clip_dir = out_root / clip_id
    inter = clip_dir / "intermediate"
    inter.mkdir(parents=True, exist_ok=True)
    if force:
        for f in ["hmr4d_results.pt", "smpl_world.npz", "gmr_unitree_h1.pkl"]:
            (inter / f).unlink(missing_ok=True)

    source = probe_video(video)
    src_video_fps = float(source.get("fps", 30.0))

    annotation = load_annotation(video)
    if label is not None:
        annotation["label"] = label
    if notes is not None:
        annotation["notes"] = notes
    if not annotation.get("label"):
        print("[process] WARNING: this clip has no motion label. Pass --label "
              "or record with the label field filled in.")

    # 1-2. recovery + SMPL export (GVHMR env)
    pt = recovery.run_gvhmr(video, inter)
    overlays = recovery.collect_render_videos(inter, clip_dir)
    smpl_npz = recovery.export_smpl_npz(pt, inter / "smpl_world.npz", src_video_fps)
    smpl = dict(np.load(smpl_npz))

    # 3. retarget (GMR env)
    pkl = retarget.run_gmr(pt, inter / "gmr_unitree_h1.pkl")
    qpos_raw, gmr_fps = retarget.gmr_to_qpos(pkl)

    # 4. postprocess + assemble
    model = load_h1_model()
    clip = postprocess(qpos_raw, gmr_fps, model)

    # 4b. physics validation: replay under full dynamics with a PD tracker
    # (+ base-assist harness) and report whether the robot could do this.
    sim = None
    if physics:
        print("[process] physics rollout...")
        sim = rollout(model, clip.t, clip.qpos)
        clip.quality["physics"] = sim.metrics
        if sim.fell_at_s is not None:
            print(f"[process] NOTE: robot fell at t={sim.fell_at_s:.1f}s "
                  f"even with the assist harness — motion is likely infeasible")

    smpl_fps = float(smpl.get("fps", src_video_fps))
    arrays = {
        "t": clip.t,
        "qpos": clip.qpos,
        "qvel": clip.qvel,
        "contacts": clip.contacts,
        **({"qpos_sim": sim.qpos_sim} if sim is not None else {}),
        "smpl_body_pose": resample_to(clip.t, smpl_fps, smpl["smpl_body_pose"]),
        "smpl_global_orient": resample_to(clip.t, smpl_fps, smpl["smpl_global_orient"]),
        "smpl_transl": resample_to(clip.t, smpl_fps, smpl["smpl_transl"]),
        "joints_3d": resample_to(clip.t, smpl_fps, smpl["joints_3d"]),
        "smpl_betas": smpl["smpl_betas"],
    }

    meta = build_meta(
        clip_id=clip_id,
        annotation=annotation,
        source_video=source,
        subject={"id": subject_id, "height_m": subject_height},
        models={
            "recovery": {
                "name": "GVHMR", "repo": "https://github.com/zju3dv/GVHMR",
                "commit": recovery.git_commit(recovery._gvhmr_dir()),
                "mode": "static-camera",
                "license_note": "research-only (ZJU); SMPL models MPI research-only",
            },
            "retarget": {
                "name": "GMR", "repo": "https://github.com/YanjieZe/GMR",
                "commit": recovery.git_commit(retarget._gmr_dir()),
                "robot": "unitree_h1",
            },
        },
        processing={
            "resample_fps": TARGET_FPS,
            "smoothing": "butterworth4 @ 6 Hz, zero-phase",
            "floor_alignment": "p5 min foot height -> z=0",
            "velocity_clamp": "per-step, Unitree URDF limits",
            "foot_skate": "reported, not corrected",
            "physics_rollout": (
                "PD torque tracking + base-assist harness; qpos_sim is the "
                "simulated trajectory" if physics else "skipped"),
            "egocentric_camera": (EGO_CAM if egocentric else "skipped"),
        },
        quality=clip.quality,
        renders=[p.name for p in overlays],
    )

    save_clip(clip_dir, arrays, meta)

    # 5. renders: preview (exocentric review artifact) + egocentric
    # (observation-shaped; uses the PHYSICS trajectory when available so the
    # observations come from physically consistent states).
    render_preview(clip_dir, video, t=clip.t, qpos=clip.qpos,
                   joints_3d=arrays["joints_3d"], fps=int(TARGET_FPS))
    if egocentric:
        ego_qpos = sim.qpos_sim if sim is not None else clip.qpos
        render_egocentric(clip_dir, qpos=ego_qpos, fps=int(TARGET_FPS))

    problems = validate_clip(clip_dir)
    if problems:
        print("[process] VALIDATION PROBLEMS:")
        for p in problems:
            print("  -", p)
    else:
        print("[process] validation passed")
    print(f"[process] quality: {json.dumps(clip.quality, indent=2)}")
    print(f"[process] done: {clip_dir}")
    return clip_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--subject-height", type=float, required=True,
                    help="subject height in meters (recorded in meta)")
    ap.add_argument("--subject-id", default="debbie")
    ap.add_argument("--clip-id", default=None)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "dataset")
    ap.add_argument("--force", action="store_true",
                    help="ignore cached intermediate results")
    ap.add_argument("--label", default=None,
                    help="what the motion is; overrides the capture sidecar")
    ap.add_argument("--notes", default=None, help="free-text take notes")
    ap.add_argument("--no-physics", action="store_true",
                    help="skip the dynamics rollout (no qpos_sim, no feasibility)")
    ap.add_argument("--no-egocentric", action="store_true",
                    help="skip the head-camera render")
    args = ap.parse_args()
    process(args.video, args.out, clip_id=args.clip_id,
            subject_height=args.subject_height, subject_id=args.subject_id,
            force=args.force, label=args.label, notes=args.notes,
            physics=not args.no_physics, egocentric=not args.no_egocentric)


if __name__ == "__main__":
    main()
