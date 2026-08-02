"""Dataset v2 clip I/O: one clip = one directory.

    dataset/<clip_id>/
      data.npz         canonical arrays (see REQUIRED_KEYS)
      meta.json        provenance + conventions + quality metrics
      preview.mp4      side-by-side render (source | skeleton | robot)
      intermediate/    raw model outputs, re-runnable provenance
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .conventions import (
    CONVENTIONS,
    H1_JOINT_ORDER,
    H1_VELOCITY_LIMITS,
    QPOS_DIM,
    QVEL_DIM,
    SMPL_NUM_JOINTS,
)

SCHEMA_VERSION = "2.0"

# key -> (expected trailing shape, dtype); None in a shape = any size
# (SMPL body_pose is 69 params, SMPL-X is 63 — the meta records which).
REQUIRED_KEYS = {
    "t": ((), np.float64),
    "qpos": ((QPOS_DIM,), np.float32),
    "qvel": ((QVEL_DIM,), np.float32),
    "smpl_body_pose": ((None,), np.float32),
    "smpl_global_orient": ((3,), np.float32),
    "smpl_transl": ((3,), np.float32),
    "joints_3d": ((None, 3), np.float32),
    "contacts": ((2,), np.bool_),
}
PER_CLIP_KEYS = {"smpl_betas": ((None,), np.float32)}
# Present only when the corresponding stage ran; validated when present.
OPTIONAL_TIMESERIES_KEYS = {"qpos_sim": ((QPOS_DIM,), np.float32)}


def _shape_matches(actual: tuple, expected: tuple) -> bool:
    return len(actual) == len(expected) and all(
        e is None or a == e for a, e in zip(actual, expected)
    )


def save_clip(clip_dir: Path, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    cast = {}
    for key, arr in arrays.items():
        spec = (REQUIRED_KEYS.get(key) or PER_CLIP_KEYS.get(key)
                or OPTIONAL_TIMESERIES_KEYS.get(key))
        cast[key] = np.asarray(arr, dtype=spec[1]) if spec else np.asarray(arr)
    np.savez_compressed(clip_dir / "data.npz", **cast)
    (clip_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_clip(clip_dir: Path):
    data = dict(np.load(clip_dir / "data.npz"))
    meta = json.loads((clip_dir / "meta.json").read_text(encoding="utf-8"))
    return data, meta


def build_meta(
    *,
    clip_id: str,
    source_video: Dict[str, Any],
    subject: Dict[str, Any],
    models: Dict[str, Any],
    processing: Dict[str, Any],
    quality: Dict[str, Any],
    annotation: Dict[str, Any] | None = None,
    renders: list[str] | None = None,
) -> Dict[str, Any]:
    annotation = annotation or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "clip_id": clip_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # The language half of vision-language-action: what the motion IS.
        "annotation": {
            "label": annotation.get("label", ""),
            "notes": annotation.get("notes", ""),
            "recorded_at_unix_ms": annotation.get("recordedAtUnixMs"),
            "session_id": annotation.get("sessionId"),
        },
        "renders": renders or [],
        "source_video": source_video,
        "subject": subject,
        "models": models,
        "robot": {
            "description": "h1_mj_description (MuJoCo Menagerie)",
            "joint_names": H1_JOINT_ORDER,
            "qpos_layout": "free base [x y z, qw qx qy qz] then 19 hinge joints",
            "velocity_limits_rad_s": H1_VELOCITY_LIMITS,
        },
        "conventions": CONVENTIONS,
        "processing": processing,
        "quality": quality,
    }


def validate_clip(clip_dir: Path) -> list[str]:
    """Shape/NaN/velocity/floor checks. Returns a list of problems (empty = ok)."""
    problems: list[str] = []
    try:
        data, meta = load_clip(clip_dir)
    except Exception as e:  # noqa: BLE001
        return [f"unreadable clip: {e!r}"]

    T = None
    for key, (shape, _dtype) in REQUIRED_KEYS.items():
        if key not in data:
            problems.append(f"missing array: {key}")
            continue
        arr = data[key]
        if T is None:
            T = arr.shape[0]
        if arr.shape[0] != T:
            problems.append(f"{key}: length {arr.shape[0]} != {T}")
        if not _shape_matches(tuple(arr.shape[1:]), shape):
            problems.append(f"{key}: shape {arr.shape[1:]} != {shape}")
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            problems.append(f"{key}: contains NaN/inf")

    for key, (shape, _dtype) in PER_CLIP_KEYS.items():
        if key in data and not _shape_matches(tuple(data[key].shape), shape):
            problems.append(f"{key}: shape {data[key].shape} != {shape}")

    for key, (shape, _dtype) in OPTIONAL_TIMESERIES_KEYS.items():
        if key not in data:
            continue
        arr = data[key]
        if T is not None and arr.shape[0] != T:
            problems.append(f"{key}: length {arr.shape[0]} != {T}")
        if not _shape_matches(tuple(arr.shape[1:]), shape):
            problems.append(f"{key}: shape {arr.shape[1:]} != {shape}")
        if not np.isfinite(arr).all():
            problems.append(f"{key}: contains NaN/inf")

    if "t" in data and len(data["t"]) > 1:
        dt = np.diff(data["t"])
        if not np.allclose(dt, dt[0], atol=1e-6):
            problems.append("t: not uniformly sampled")

    if "qpos" in data and "qvel" in data and len(data["qpos"]) > 1:
        quat_norm = np.linalg.norm(data["qpos"][:, 3:7], axis=1)
        if not np.allclose(quat_norm, 1.0, atol=1e-4):
            problems.append("qpos: base quaternion not normalized")
        limits = np.array([H1_VELOCITY_LIMITS[n] for n in H1_JOINT_ORDER])
        max_qvel = np.abs(data["qvel"][:, 6:]).max(axis=0)
        over = max_qvel > limits * 1.001
        if over.any():
            worst = [
                f"{H1_JOINT_ORDER[i]}={max_qvel[i]:.1f}>{limits[i]}"
                for i in np.where(over)[0]
            ]
            problems.append(f"qvel over URDF limits: {', '.join(worst)}")

    if meta.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"meta schema_version != {SCHEMA_VERSION}")

    if not (meta.get("annotation") or {}).get("label"):
        problems.append(
            "no motion label: this clip says nothing about WHAT the motion is, "
            "which a vision-language-action model needs")

    return problems
