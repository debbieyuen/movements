# Legacy (retired 2026-08)

The original 3-camera pipeline, retired when the project pivoted to
single-camera capture + GVHMR (see the root README). Kept for reference;
none of this is wired into the current app, and some of it depends on the
old wire format, so expect breakage if you run it.

| file | what it was | why retired |
|---|---|---|
| `websocket_pose_server.py` | v1 websockets server | superseded by `server/app.py` (FastAPI, sessions, auth, canonical z-up) |
| `play_latest_h1_mujoco_ik.py` | live H1 IK viewer | superseded by `server/live_viewer_h1.py` (OneEuro, root yaw/height) |
| `body_fuse_3d.py` | 3-camera Kabsch "fusion" | averaged monocular guesses; never gravity-aligned (the sideways-skeleton bug) |
| `detect_clap.py` | wrist-distance clap sync | single camera needs no sync |
| `calibrate_cameras.py` | checkerboard calibration | output was never consumed; GVHMR needs no calibration |
| `extract_frames.py` | stills for calibration | with the above |
| `dataset_builder.py` | v1 dataset exporter | produced unphysical qvel (56 rad/s), pinned root; replaced by `mocap/` |
| `play_fused_h1.py` | fused-3-cam H1 player | 3-cam path retired |
| `play_latest_mujoco.py` / `play_latest_mujoco_ik.py` | Humanoid-v5 players | off-target (H1 is the target) |
| `play_latest_h1_in_maniskill.py` | ManiSkill player | broken: wrote base pose into SAPIEN actuated-joint qpos |
| `live_h1_remapper.py` | heuristic 32-DOF pseudo-H1 | invented joint layout, xyzw quaternion bug |
| `mujoco_humanoid_remapper.py` | H1→Humanoid index remap | dead code (nothing imported it) |
| `h1_joint_remap.py` | offline heuristic retarget CLI | superseded by GMR |
| `list_envs.py` | gymnasium env lister | one-off utility |
| `QuestRecorder.tsx` | WebXR head/controller logger | nothing ever consumed its data |
| `checkerboard_9x6_18mm.svg` | calibration board | with the calibration |
