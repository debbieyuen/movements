import pickle

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mocap.conventions import (  # noqa: E402
    H1_JOINT_ORDER,
    H1_VELOCITY_LIMITS,
    QPOS_DIM,
    load_h1_model,
)
from mocap.postprocess import postprocess, resample, smooth  # noqa: E402
from mocap.retarget import gmr_to_qpos  # noqa: E402
from mocap.schema import build_meta, load_clip, save_clip, validate_clip  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return load_h1_model()


def _standing_qpos(T=90, fps=30.0, wave=True):
    """Synthetic standing motion with a (deliberately violent) arm wave."""
    qpos = np.zeros((T, QPOS_DIM))
    qpos[:, 2] = 0.98
    qpos[:, 3] = 1.0  # identity wxyz
    if wave:
        elbow = H1_JOINT_ORDER.index("left_elbow") + 7
        tt = np.arange(T) / fps
        qpos[:, elbow] = 1.0 * np.sin(2 * np.pi * 8.0 * tt)  # 8 Hz thrash
        # inject a single-frame spike: raw finite differences would explode
        qpos[T // 2, elbow] += 1.5
    return qpos


def test_model_matches_conventions(model):
    assert model.nq == QPOS_DIM
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    assert [n for n in names if n] == H1_JOINT_ORDER


def test_resample_preserves_duration():
    qpos = _standing_qpos(T=60, wave=False)
    t, out = resample(qpos, src_fps=60.0, dst_fps=30.0)
    assert len(out) == pytest.approx(30, abs=2)
    assert t[-1] == pytest.approx(59 / 60.0, abs=1 / 30.0)
    assert np.allclose(np.linalg.norm(out[:, 3:7], axis=1), 1.0, atol=1e-6)


def test_smooth_keeps_quaternion_valid():
    qpos = _standing_qpos()
    out = smooth(qpos, fps=30.0)
    assert np.allclose(np.linalg.norm(out[:, 3:7], axis=1), 1.0, atol=1e-3)


def test_postprocess_enforces_velocity_limits(model):
    clip = postprocess(_standing_qpos(), src_fps=30.0, model=model)
    limits = np.array([H1_VELOCITY_LIMITS[n] for n in H1_JOINT_ORDER])
    assert (np.abs(clip.qvel[:, 6:]) <= limits[None, :] * 1.01).all(), (
        f"max qvel {np.abs(clip.qvel[:, 6:]).max():.1f} exceeds limits")
    assert clip.quality["max_abs_qvel_rad_s"] <= limits.max() * 1.01


def test_postprocess_floor_alignment(model):
    qpos = _standing_qpos(wave=False)
    qpos[:, 2] += 0.5  # float the robot half a meter up
    clip = postprocess(qpos, src_fps=30.0, model=model)
    # after alignment the standing feet should be at (or just above) z=0
    assert abs(clip.quality["floor_offset_m"] - 0.5) < 0.1
    assert clip.quality["contact_fraction"]["left"] > 0.9


def test_gmr_pickle_reindexes_by_name(tmp_path):
    T = 5
    # GMR order deliberately shuffled vs h1_mj_description order
    gmr_names = list(reversed([f"{n}_joint" for n in H1_JOINT_ORDER]))
    joints = np.tile(np.arange(len(gmr_names), dtype=float), (T, 1))
    pkl = tmp_path / "gmr.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({
            "robot_base_translation": np.zeros((T, 3)),
            "robot_base_rotation": np.tile([1.0, 0, 0, 0], (T, 1)),
            "robot_joint_positions": joints,
            "robot_joint_names": gmr_names,
            "fps": 30.0,
        }, f)
    qpos, fps = gmr_to_qpos(pkl)
    assert fps == 30.0 and qpos.shape == (T, QPOS_DIM)
    # left_hip_yaw is LAST in the shuffled GMR list -> value len-1... check map
    for k, name in enumerate(H1_JOINT_ORDER):
        expected = gmr_names.index(f"{name}_joint")
        assert qpos[0, 7 + k] == expected


def test_gmr_pickle_without_names_is_rejected(tmp_path):
    pkl = tmp_path / "gmr.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({
            "robot_base_translation": np.zeros((2, 3)),
            "robot_base_rotation": np.tile([1.0, 0, 0, 0], (2, 1)),
            "robot_joint_positions": np.zeros((2, 19)),
        }, f)
    with pytest.raises(KeyError):
        gmr_to_qpos(pkl)


def test_save_and_validate_clip(tmp_path, model):
    clip = postprocess(_standing_qpos(), src_fps=30.0, model=model)
    T = len(clip.qpos)
    arrays = {
        "t": clip.t,
        "qpos": clip.qpos,
        "qvel": clip.qvel,
        "contacts": clip.contacts,
        "smpl_body_pose": np.zeros((T, 69)),
        "smpl_global_orient": np.zeros((T, 3)),
        "smpl_transl": np.zeros((T, 3)),
        "joints_3d": np.zeros((T, 24, 3)),
        "smpl_betas": np.zeros(10),
    }
    meta = build_meta(
        clip_id="test", source_video={"filename": "x.mp4"},
        subject={"id": "test", "height_m": 1.57},
        models={}, processing={}, quality=clip.quality,
        annotation={"label": "waving hello", "notes": ""},
    )
    save_clip(tmp_path / "test", arrays, meta)
    assert validate_clip(tmp_path / "test") == []
    _, saved_meta = load_clip(tmp_path / "test")
    assert saved_meta["annotation"]["label"] == "waving hello"


def test_validate_flags_missing_motion_label(tmp_path, model):
    clip = postprocess(_standing_qpos(), src_fps=30.0, model=model)
    T = len(clip.qpos)
    arrays = {
        "t": clip.t, "qpos": clip.qpos, "qvel": clip.qvel,
        "contacts": clip.contacts,
        "smpl_body_pose": np.zeros((T, 69)),
        "smpl_global_orient": np.zeros((T, 3)),
        "smpl_transl": np.zeros((T, 3)),
        "joints_3d": np.zeros((T, 24, 3)),
        "smpl_betas": np.zeros(10),
    }
    meta = build_meta(clip_id="unlabelled", source_video={}, subject={},
                      models={}, processing={}, quality=clip.quality)
    save_clip(tmp_path / "unlabelled", arrays, meta)
    assert any("no motion label" in p for p in validate_clip(tmp_path / "unlabelled"))


def test_validate_catches_bad_velocity(tmp_path, model):
    clip = postprocess(_standing_qpos(), src_fps=30.0, model=model)
    T = len(clip.qpos)
    bad_qvel = clip.qvel.copy()
    bad_qvel[5, 10] = 56.0  # the v1 bug, resurrected
    arrays = {
        "t": clip.t, "qpos": clip.qpos, "qvel": bad_qvel,
        "contacts": clip.contacts,
        "smpl_body_pose": np.zeros((T, 69)),
        "smpl_global_orient": np.zeros((T, 3)),
        "smpl_transl": np.zeros((T, 3)),
        "joints_3d": np.zeros((T, 24, 3)),
        "smpl_betas": np.zeros(10),
    }
    meta = build_meta(clip_id="bad", source_video={}, subject={},
                      models={}, processing={}, quality={})
    save_clip(tmp_path / "bad", arrays, meta)
    problems = validate_clip(tmp_path / "bad")
    assert any("qvel over URDF limits" in p for p in problems)


def test_load_annotation_sidecar(tmp_path):
    from mocap.process import load_annotation

    video = tmp_path / "take1.webm"
    video.write_bytes(b"x")
    assert load_annotation(video) == {}

    video.with_suffix(".annotation.json").write_text(
        '{"label": "squatting", "notes": "third try", "sessionId": "abc"}')
    ann = load_annotation(video)
    assert ann["label"] == "squatting" and ann["notes"] == "third try"


def test_load_annotation_survives_broken_json(tmp_path):
    from mocap.process import load_annotation

    video = tmp_path / "take2.webm"
    video.write_bytes(b"x")
    video.with_suffix(".annotation.json").write_text("{not json")
    assert load_annotation(video) == {}
