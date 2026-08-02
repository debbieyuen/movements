import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mocap.conventions import H1_JOINT_ORDER, QPOS_DIM, load_h1_model  # noqa: E402
from mocap.physics import rollout  # noqa: E402
from mocap.render import EGO_CAM, h1_model_with_ego_camera, render_egocentric  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return load_h1_model()


def _standing_clip(T=45, fps=30.0):
    t = np.arange(T) / fps
    qpos = np.zeros((T, QPOS_DIM))
    qpos[:, 2] = 0.98
    qpos[:, 3] = 1.0
    # slight knee bend so the pose is statically reasonable
    for name, val in [("left_knee", 0.25), ("right_knee", 0.25),
                      ("left_hip_pitch", -0.12), ("right_hip_pitch", -0.12),
                      ("left_ankle", -0.13), ("right_ankle", -0.13)]:
        qpos[:, 7 + H1_JOINT_ORDER.index(name)] = val
    return t, qpos


def test_standing_rollout_survives_with_assist(model):
    t, qpos = _standing_clip()
    res = rollout(model, t, qpos)
    assert res.qpos_sim.shape == qpos.shape
    assert res.metrics["survived"] is True
    assert res.fell_at_s is None
    assert res.metrics["mean_joint_tracking_error_rad"] < 0.15
    assert np.isfinite(res.qpos_sim).all()


def test_unassisted_rollout_reports_honestly(model):
    # Without the harness a plain PD humanoid may or may not stay up while
    # standing, but the result must be well-formed either way and any fall
    # must carry a time.
    t, qpos = _standing_clip(T=30)
    res = rollout(model, t, qpos, assist_stiffness=0.0, assist_damping=0.0)
    assert res.metrics["assisted"] is False
    assert res.metrics["max_assist_force_N"] == 0.0
    if not res.metrics["survived"]:
        assert res.fell_at_s is not None and res.fell_at_s > 0
    assert np.isfinite(res.qpos_sim).all()


def test_impossible_motion_is_flagged(model):
    # Teleport the reference base far sideways instantly: the harness itself
    # gets yanked, and the assist-force metric must reflect the violence even
    # if the robot survives dangling from it.
    t, qpos = _standing_clip(T=40)
    qpos[20:, 0] = 3.0
    res = rollout(model, t, qpos)
    assert (not res.metrics["survived"]
            or res.metrics["max_assist_force_N"] > 200.0)


def test_ego_camera_attaches(model):
    m = h1_model_with_ego_camera()
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, EGO_CAM["name"])
    assert cam_id >= 0
    assert m.nq == model.nq  # same robot, just +1 camera


def test_render_egocentric_produces_video(tmp_path):
    _, qpos = _standing_clip(T=8)
    out = render_egocentric(tmp_path, qpos=qpos, fps=30, width=160, height=120)
    assert out.exists() and out.stat().st_size > 1000
    import imageio.v2 as imageio

    frame = imageio.get_reader(out).get_data(4)
    assert frame.shape[2] == 3
    # An egocentric view of a lit floor world must not be a black frame.
    assert frame.mean() > 10
