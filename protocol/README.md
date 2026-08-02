# Pose protocol v2 — the contract

Authoritative machine-readable schema: [`pose_protocol.schema.json`](pose_protocol.schema.json).
Twins: [`server/protocol.py`](../server/protocol.py) (Python, tested against the
schema in `server/tests/test_protocol.py`) and
[`my-app/lib/protocol.ts`](../my-app/lib/protocol.ts) (TypeScript).

## Coordinate frames

| name | axes | who uses it |
|---|---|---|
| `mp-camera` | MediaPipe world landmarks: x image-right, **y image-DOWN**, z away-from-camera; meters, hip-centered | browser → server wire only |
| `zup-xfwd` | **canonical**: right-handed, +X forward (toward camera), +Y left, +Z up, gravity −z | everything on disk and downstream |

The conversion happens **exactly once, server-side** (`mp_world_to_zup`):

```
X = -s · z_mp      Y = x_mp      Z = -y_mp        (s = meta.depthScale, 0.35)
```

`s` compresses MediaPipe's noisy monocular depth; it is recorded in every
frame's `meta.depthScale` so it can be divided back out. The rotation part is
proper (det = +1) — handedness is preserved. Consumers must check
`coord == "zup-xfwd"` and refuse anything else; nobody re-guesses axes.

## Timestamps

- `tMs` — milliseconds since **this client's stream/recording epoch** (reset
  when recording starts or Live is toggled on). Same meaning on the wire, in
  the downloaded keypoints file, and in server logs.
- `unixMs` — client wall clock; `serverUnixMs` — stamped on receipt.

## Quaternions

**wxyz** (MuJoCo order) everywhere in Python and on disk. No exceptions.

## Offline dataset conventions

See [`mocap/conventions.py`](../mocap/conventions.py): z-up, gravity −z,
floor z=0, meters/radians/seconds, 30 Hz, world-frame (not hip-centered),
H1 joint order + URDF velocity limits. Every clip's `meta.json` restates them.
