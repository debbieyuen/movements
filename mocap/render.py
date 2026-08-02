"""Preview rendering: side-by-side mp4 (source video | SMPL skeleton | H1).

Uses the MuJoCo offscreen renderer (set MUJOCO_GL=egl on headless boxes) and
imageio-ffmpeg for encoding. The skeleton panel is drawn upright on a floor
grid — the exact artifact the team asked for after the sideways videos.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .conventions import load_h1_model

PANEL_W, PANEL_H = 426, 480
FPS_DEFAULT = 30

# SMPL 24-joint kinematic tree (parent of each joint; -1 = root).
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                16, 17, 18, 19, 20, 21]

_EMPTY_WORLD = """
<mujoco>
  <visual><headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/></visual>
  <asset>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.85 0.85 0.85"
             rgb2="0.75 0.75 0.75" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <texture type="skybox" builtin="flat" rgb1="0.9 0.9 0.9" rgb2="0.9 0.9 0.9"
             width="16" height="16"/>
  </asset>
  <worldbody>
    <geom type="plane" size="5 5 0.1" material="grid"/>
    <light pos="0 0 3" dir="0 0 -1"/>
  </worldbody>
</mujoco>
"""


def _lookat_camera(cam: mujoco.MjvCamera, target: np.ndarray) -> None:
    cam.lookat[:] = [target[0], target[1], max(float(target[2]), 0.6)]
    cam.distance = 3.0
    cam.azimuth = 135.0
    cam.elevation = -12.0


def _draw_skeleton(scene: mujoco.MjvScene, joints: np.ndarray) -> None:
    """Append capsule 'bones' + joint spheres to a rendered scene."""
    for j, parent in enumerate(SMPL_PARENTS):
        if scene.ngeom >= scene.maxgeom - 2:
            break
        if parent >= 0:
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                np.zeros(3), np.zeros(3), np.zeros(9),
                                np.array([0.15, 0.5, 0.9, 1.0], dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.02,
                                 joints[parent], joints[j])
            scene.ngeom += 1
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.03, 0.0, 0.0]), joints[j].astype(np.float64),
                            np.eye(3).flatten(),
                            np.array([0.95, 0.5, 0.1, 1.0], dtype=np.float32))
        scene.ngeom += 1


def render_preview(
    clip_dir: Path,
    video: Path | None,
    *,
    t: np.ndarray,
    qpos: np.ndarray,
    joints_3d: np.ndarray,
    fps: int = FPS_DEFAULT,
) -> Path:
    import imageio.v2 as imageio

    out_path = clip_dir / "preview.mp4"

    h1_model = load_h1_model()
    h1_data = mujoco.MjData(h1_model)
    h1_renderer = mujoco.Renderer(h1_model, height=PANEL_H, width=PANEL_W)

    world_model = mujoco.MjModel.from_xml_string(_EMPTY_WORLD)
    world_data = mujoco.MjData(world_model)
    world_renderer = mujoco.Renderer(world_model, height=PANEL_H, width=PANEL_W)
    cam = mujoco.MjvCamera()

    cap = None
    src_frames = 0
    src_fps = fps
    if video is not None and Path(video).exists():
        import cv2

        cap = cv2.VideoCapture(str(video))
        src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                quality=7, macro_block_size=2)
    try:
        for i in range(len(qpos)):
            panels = []

            if cap is not None and src_frames > 0:
                import cv2

                src_idx = min(int(round(t[i] * src_fps)), src_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
                ok, frame = cap.read()
                if ok:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = _fit(frame, PANEL_W, PANEL_H)
                else:
                    frame = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
                panels.append(frame)

            _lookat_camera(cam, joints_3d[i, 0])
            world_renderer.update_scene(world_data, camera=cam)
            _draw_skeleton(world_renderer.scene, joints_3d[i])
            panels.append(world_renderer.render())

            h1_data.qpos[:] = qpos[i]
            mujoco.mj_forward(h1_model, h1_data)
            _lookat_camera(cam, qpos[i, 0:3])
            h1_renderer.update_scene(h1_data, camera=cam)
            panels.append(h1_renderer.render())

            writer.append_data(np.concatenate(panels, axis=1))
    finally:
        writer.close()
        h1_renderer.close()
        world_renderer.close()
        if cap is not None:
            cap.release()

    print(f"[render] wrote {out_path}")
    return out_path


# --------------------------------------------------------------------------
# Egocentric (head camera) rendering
#
# The Menagerie H1 has no separate head body (the head is part of the torso
# mesh) and ships no cameras, so a camera named "ego" is attached to
# torso_link at head height via MjSpec. It looks along the torso's +X
# (forward) axis. MuJoCo cameras look down their -Z with +Y up, so the
# camera frame is x=-Y_body, y=+Z_body.
# --------------------------------------------------------------------------
EGO_CAM = {
    "name": "ego",
    "body": "torso_link",
    "pos": [0.14, 0.0, 0.62],   # just in front of the H1 head shell
    # Camera frame in body coords: x=-Y (image-right), y=+Z (image-up),
    # z=-X (cameras look down -z) -> looks along the torso's forward +X.
    "axes_x": [0.0, -1.0, 0.0],
    "axes_y": [0.0, 0.0, 1.0],
    "fovy": 90.0,               # wide FOV, typical for robot head cameras
}


def _ego_quat_wxyz() -> np.ndarray:
    from scipy.spatial.transform import Rotation

    x = np.array(EGO_CAM["axes_x"], dtype=float)
    y = np.array(EGO_CAM["axes_y"], dtype=float)
    R = np.column_stack([x, y, np.cross(x, y)])
    xyzw = Rotation.from_matrix(R).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def h1_model_with_ego_camera() -> "mujoco.MjModel":
    from .conventions import h1_model_path

    spec = mujoco.MjSpec.from_file(h1_model_path())
    body = spec.body(EGO_CAM["body"])
    if body is None:
        raise RuntimeError(f"body {EGO_CAM['body']!r} not found in the H1 model")
    cam = body.add_camera()
    cam.name = EGO_CAM["name"]
    cam.pos = EGO_CAM["pos"]
    cam.quat = _ego_quat_wxyz()
    cam.fovy = EGO_CAM["fovy"]
    return spec.compile()


def render_egocentric(
    clip_dir: Path,
    *,
    qpos: np.ndarray,
    fps: int = FPS_DEFAULT,
    width: int = 640,
    height: int = 480,
) -> Path:
    """Render what the robot's head camera sees along the given trajectory.

    Pass the PHYSICS trajectory (qpos_sim) when available: observation-action
    pairs should come from physically consistent states. Note the scene is an
    empty floor -- these frames show embodiment and self-motion, not objects.
    """
    import imageio.v2 as imageio

    out_path = clip_dir / "egocentric.mp4"
    model = h1_model_with_ego_camera()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                quality=7, macro_block_size=2)
    try:
        for q in qpos:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=EGO_CAM["name"])
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()
    print(f"[render] wrote {out_path}")
    return out_path


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Letterbox-resize img to (h, w)."""
    import cv2

    scale = min(w / img.shape[1], h / img.shape[0])
    nw, nh = int(img.shape[1] * scale), int(img.shape[0] * scale)
    resized = cv2.resize(img, (nw, nh))
    out = np.zeros((h, w, 3), dtype=np.uint8)
    y, x = (h - nh) // 2, (w - nw) // 2
    out[y:y + nh, x:x + nw] = resized
    return out
