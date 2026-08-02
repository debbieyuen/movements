// Pose protocol v2 — TypeScript side.
//
// The authoritative definition is protocol/pose_protocol.schema.json at the
// repo root; server/protocol.py is the Python twin. Keep all three in sync.
//
// The client sends RAW MediaPipe world landmarks ("mp-camera": x image-right,
// y image-DOWN, z away-from-camera, hip-centered meters). The server converts
// to the canonical z-up frame exactly once; the browser never transforms axes.
//
// tMs = milliseconds since this client's stream/recording epoch (near zero at
// the start of a take) — the SAME meaning on the wire and in downloaded files.

export const PROTOCOL_VERSION = 2;

export type Landmark4 = [number, number, number, number]; // x, y, z, visibility

export type PoseFrameWire = {
  v: typeof PROTOCOL_VERSION;
  type: 'pose';
  sessionId: string;
  role: string;
  seq: number;
  tMs: number;
  unixMs: number;
  coord: 'mp-camera';
  source?: string;
  world: Landmark4[];
  norm?: Landmark4[];
};

export type HelloMessage = {
  v: typeof PROTOCOL_VERSION;
  type: 'hello' | 'heartbeat';
  sessionId: string;
  role: string;
  clientUnixMs?: number;
};

export type GoMessage = {
  v: typeof PROTOCOL_VERSION;
  type: 'go';
  sessionId: string;
  seconds: number;
};

export const POSE_SOURCE = 'mediapipe-tasks-vision/pose_landmarker_full';

type MpLandmark = { x: number; y: number; z: number; visibility?: number };

export function toLandmark4(landmarks: MpLandmark[] | undefined): Landmark4[] {
  if (!landmarks) return [];
  return landmarks.map((l) => [
    round4(l.x),
    round4(l.y),
    round4(l.z),
    round4(typeof l.visibility === 'number' ? l.visibility : 1),
  ]);
}

function round4(n: number): number {
  return Math.round(n * 10000) / 10000;
}
