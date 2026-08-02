# GPU box setup (Lambda, Linux + NVIDIA)

One-time setup for the offline pipeline (GVHMR + GMR). Everything lives in a
`mocap` conda env plus two pinned clones under `external/` (gitignored).

## 1. Clone this repo and create the env

```bash
git clone <this-repo> ~/movements && cd ~/movements
conda create -n mocap python=3.10 -y
conda activate mocap
pip install -e ".[dev]"
```

## 2. GVHMR (video → world-grounded SMPL motion)

```bash
mkdir -p external && cd external
git clone https://github.com/zju3dv/GVHMR.git
cd GVHMR
# Follow its docs/INSTALL.md: pinned torch + requirements, then:
pip install -e .
```

Assets GVHMR needs (see its INSTALL.md for links):
- `inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt`
- SMPL/SMPL-X body models under `inputs/checkpoints/body_models/`
  (register at https://smpl.is.tue.mpg.de / https://smpl-x.is.tue.mpg.de)
- `pip install smplx` in the same env if not pulled in already.

Smoke test (their demo, static camera):

```bash
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s
```

⚠ **License**: GVHMR code is research/non-commercial (ZJU); SMPL body models
are MPI research-only. Both are fine for research — flag it to the team, and
the recovery backend is swappable (WHAM is MIT) if it ever matters.

## 3. GMR (SMPL → robot retargeting, MIT)

```bash
cd ~/movements/external
git clone https://github.com/YanjieZe/GMR.git
cd GMR
pip install -e .
# SMPL-X body models also go under assets/body_models (see its README)
```

Smoke test:

```bash
python scripts/gvhmr_to_robot.py \
  --gvhmr_pred_file <hmr4d_results.pt from step 2> \
  --robot unitree_h1 --save_path /tmp/test_h1.pkl
```

After both smoke tests pass, record the pins:

```bash
git -C external/GVHMR rev-parse HEAD
git -C external/GMR rev-parse HEAD
```

(Every processed clip records these commits in its meta.json automatically.)

## 4. Process a clip

On the box:

```bash
MUJOCO_GL=egl python -m mocap.process inbox/take1.mp4 --subject-height 1.57
```

From the Mac (uploads, processes, downloads the finished clip):

```bash
MOCAP_GPU_HOST=ubuntu@<box-ip> scripts/process_remote.sh take1.mp4 --subject-height 1.57
```

If torch pins ever conflict between GVHMR and GMR, split them into two conda
envs and point the wrappers at them with `MOCAP_GVHMR_CONDA_ENV=gvhmr` /
`MOCAP_GMR_CONDA_ENV=gmr` — the wrappers shell out, so nothing else changes.

## Recording the live H1 demo

The live viewer can save what the robot does, not just show it:

```bash
python -m server.live_viewer_h1 --record     # window + mp4 + qpos .npz
python -m server.live_viewer_h1 --headless   # record with no window
```

Output lands in `live/recordings/` as `live_h1_<timestamp>.mp4` plus a matching
`.npz` holding `t`, `qpos` (T×26) and the joint names. Stop it with ctrl-c or
`kill` — both finalize the files cleanly.

On macOS the interactive window requires `mjpython -m server.live_viewer_h1`
(a MuJoCo restriction); `--headless` works under plain `python` everywhere.

Note the frame rate: the server publishes a "latest pose" file at 30 Hz and the
viewer samples it, so the recording captures the poses it actually rendered —
expect somewhat fewer frames than the browser sent.
