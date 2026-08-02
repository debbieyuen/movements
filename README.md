# movements

<img width="1160" height="428" alt="Screenshot 2026-06-22 090311" src="https://github.com/user-attachments/assets/38f03302-b4d0-4aa6-9e6b-d1e15f8f48e8" />

Single-camera human motion capture for humanoid robots. Record yourself with
one ordinary camera; get back a physically-plausible **Unitree H1** motion
clip (world-grounded, gravity-aligned, velocity-limited) plus a preview video
— or stream your skeleton to a live H1 in MuJoCo in real time.

```
capture (browser, Mac/phone)          live demo (Mac)
  Next.js + MediaPipe tasks-vision      python -m server.live_viewer_h1
        │ wss/https (one tunnel)          ▲ live/latest_pose.json
        ▼                                 │
  FastAPI server  ── canonical z-up frames, sessions/<sid>/…
        │ video upload
        ▼
offline pipeline (Lambda GPU box)
  python -m mocap.process video.mp4 --subject-height 1.57
    → GVHMR (world SMPL) → GMR (H1 retarget) → postprocess
    → dataset/<clip_id>/{data.npz, meta.json, preview.mp4}
```

## Quickstart (capture + live demo, local)

```bash
# 1. web app
cd my-app && npm install && npm run dev          # http://localhost:3000

# 2. pose server (from the repo root)
python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
uvicorn server.app:app --host 0.0.0.0 --port 8765
#   → prints a token; MOCAP_TOKEN=disabled turns auth off for local use

# 3. live H1 viewer (optional)
python -m server.live_viewer_h1
```

Open a session, pick your camera, **Start preview**, and toggle **Live to
robot** — the H1 in the MuJoCo viewer mirrors you. **Record/Stop** uploads
the clip + keypoints to `sessions/<sid>/`.

### Phones / remote cameras

Expose the server with a Cloudflare quick tunnel:

```bash
cloudflared tunnel --url http://localhost:8765
```

Paste the `https://xxxx.trycloudflare.com` URL and the server token into the
app's **Server URL / Token** fields → Set. **Copy session link** carries both
to other devices.

## Offline processing (GPU box)

One-time setup: [docs/GPU_SETUP.md](docs/GPU_SETUP.md) (GVHMR + GMR under
`external/`, conda env `mocap`).

```bash
# on the box
MUJOCO_GL=egl python -m mocap.process inbox/take1.mp4 --subject-height 1.57

# or from the Mac (rsync up → process → rsync the clip back)
MOCAP_GPU_HOST=ubuntu@<ip> scripts/process_remote.sh take1.mp4 --subject-height 1.57
```

Each clip lands in `dataset/<clip_id>/`:

- `data.npz` — `t (T,)`, `qpos (T,26)`, `qvel (T,25)` (velocity-clamped to
  Unitree URDF limits), `qpos_sim (T,26)` (the physics rollout), `contacts
  (T,2)`, SMPL params + `joints_3d (T,24,3)`
- `meta.json` — the motion label + notes from capture, conventions, model
  repo commits, quality metrics (max |qvel|, foot skate, physics feasibility)
- `preview.mp4` — source | upright skeleton | retargeted H1, side by side.
  **Check it before sharing the clip.**
- `egocentric.mp4` — what the robot's head camera sees along the physics
  trajectory (observation-shaped; empty-floor world for now)
- `gvhmr_*.mp4` — GVHMR's own tracking overlays, kept for review

Physics validation replays every clip under MuJoCo dynamics with a PD tracker
and a base-assist harness; `meta.quality.physics` reports whether the robot
survived, tracking error, and how hard the harness had to pull (large assist
forces = motion is far from feasible on a real H1). Skip with `--no-physics`.

⚠ GVHMR is research-only licensed (and SMPL body models are MPI
research-only). Fine for research; the recovery backend is swappable
(`mocap/recovery.py`) if that ever changes.

## Conventions

Everything downstream of the browser is **right-handed, z-up, gravity −z,
meters, radians, seconds, wxyz quaternions, 30 Hz, world-frame**. The one
place camera coordinates become world coordinates is the server boundary.
Details: [protocol/README.md](protocol/README.md).

## Repo layout

| path | what |
|---|---|
| `my-app/` | Next.js capture app (hooks in `my-app/hooks/`, self-hosted MediaPipe assets) |
| `server/` | FastAPI pose server, protocol, live H1 viewer, OneEuro filter |
| `mocap/` | offline pipeline: recovery, retarget, postprocess, render, CLI |
| `protocol/` | wire/disk schema (JSON Schema + docs) |
| `scripts/` | `process_remote.sh` |
| `docs/` | GPU box setup |
| `legacy/` | retired 3-camera pipeline (see its README) |

## Tests

```bash
python -m pytest            # server protocol/app + mocap postprocess/schema
cd my-app && npm run lint && npm run build
```
