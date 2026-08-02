'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  Landmark,
  NormalizedLandmark,
  PoseLandmarker,
} from '@mediapipe/tasks-vision';

type VideoWithRVFC = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: () => void) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

export type PoseResult = {
  /** ms since the pose loop started for this stream (client epoch) */
  tCaptureMs: number;
  landmarks: NormalizedLandmark[]; // normalized image space
  worldLandmarks: Landmark[]; // meters, hip-centered, MediaPipe axes
};

export type PoseStats = { fps: number; frames: number; detecting: boolean };

const MAX_FPS = 30;

/**
 * MediaPipe tasks-vision PoseLandmarker driven by requestVideoFrameCallback
 * (one inference per delivered camera frame, capped at MAX_FPS).
 *
 * Assets are self-hosted under public/ (see scripts/fetch-assets.mjs) --
 * no runtime CDN dependency. Subscribe to results with onFrame(cb); results
 * flow through refs, NOT React state, so nothing re-renders at 30 Hz.
 */
export function usePose(videoRef: React.RefObject<HTMLVideoElement | null>) {
  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<PoseStats>({
    fps: 0,
    frames: 0,
    detecting: false,
  });

  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const runningRef = useRef(false);
  const rafHandleRef = useRef<number | null>(null);
  const timeoutHandleRef = useRef<number | null>(null);
  const epochRef = useRef<number | null>(null);
  const lastInferMsRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const callbackRef = useRef<((r: PoseResult) => void) | null>(null);
  const frameCountRef = useRef(0);
  const fpsWindowRef = useRef<{ t: number; n: number }>({ t: 0, n: 0 });

  const onFrame = useCallback((cb: ((r: PoseResult) => void) | null) => {
    callbackRef.current = cb;
  }, []);

  const ensureLandmarker = useCallback(async () => {
    if (landmarkerRef.current) return landmarkerRef.current;
    setStatus('loading');
    setError(null);
    try {
      const { FilesetResolver, PoseLandmarker } = await import('@mediapipe/tasks-vision');
      const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
      const landmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: '/models/pose_landmarker_full.task',
          delegate: 'GPU', // falls back to CPU automatically
        },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      landmarkerRef.current = landmarker;
      setStatus('ready');
      return landmarker;
    } catch (err) {
      const msg =
        (err as Error)?.message ??
        'Failed to load the pose model. Re-run: node scripts/fetch-assets.mjs';
      setStatus('error');
      setError(msg);
      throw err;
    }
  }, []);

  const start = useCallback(async () => {
    await ensureLandmarker();
    if (runningRef.current) return;
    runningRef.current = true;
    epochRef.current = null;
    frameCountRef.current = 0;
    lastVideoTimeRef.current = -1;
    fpsWindowRef.current = { t: performance.now(), n: 0 };

    function schedule() {
      if (!runningRef.current) return;
      const video = videoRef.current as VideoWithRVFC | null;
      // Neither requestVideoFrameCallback nor rAF fires while the tab is
      // hidden (backgrounded tab, locked phone). Without a fallback the pose
      // loop would stop dead mid-take and never restart, so fall back to a
      // timer: browsers throttle background timers to ~1 Hz, which keeps the
      // loop alive and lets it resume at full rate the moment you come back.
      if (document.hidden || !video?.requestVideoFrameCallback) {
        timeoutHandleRef.current = window.setTimeout(tick, 1000 / MAX_FPS);
      } else {
        rafHandleRef.current = video.requestVideoFrameCallback(tick);
      }
    }

    function tick() {
      const video = videoRef.current;
      if (!runningRef.current || !video) return;

      const now = performance.now();
      const landmarker = landmarkerRef.current;
      if (
        !landmarker ||
        video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
        now - lastInferMsRef.current < 1000 / MAX_FPS ||
        video.currentTime === lastVideoTimeRef.current
      ) {
        schedule();
        return;
      }
      lastInferMsRef.current = now;
      lastVideoTimeRef.current = video.currentTime;

      try {
        const res = landmarker.detectForVideo(video, now);
        const detected = !!(res.landmarks?.length && res.worldLandmarks?.length);
        if (detected) {
          if (epochRef.current == null) epochRef.current = now;
          frameCountRef.current += 1;
          callbackRef.current?.({
            tCaptureMs: now - epochRef.current,
            landmarks: res.landmarks[0],
            worldLandmarks: res.worldLandmarks[0],
          });
        }

        // fps counts INFERENCES, not detections: an empty frame means nobody
        // is in view, not that the model stalled. `detecting` distinguishes
        // them in the UI. Published at ~2 Hz so nothing re-renders per frame.
        const w = fpsWindowRef.current;
        w.n += 1;
        if (now - w.t >= 500) {
          setStats({
            fps: Math.round((w.n * 1000) / Math.max(now - w.t, 1)),
            frames: frameCountRef.current,
            detecting: detected,
          });
          fpsWindowRef.current = { t: now, n: 0 };
        }
      } catch (err) {
        console.error('Pose inference error:', err);
      }
      schedule();
    }

    tick();
  }, [ensureLandmarker, videoRef]);

  const stop = useCallback(() => {
    runningRef.current = false;
    const video = videoRef.current as VideoWithRVFC | null;
    if (rafHandleRef.current != null) {
      video?.cancelVideoFrameCallback?.(rafHandleRef.current);
      rafHandleRef.current = null;
    }
    if (timeoutHandleRef.current != null) {
      window.clearTimeout(timeoutHandleRef.current);
      timeoutHandleRef.current = null;
    }
  }, [videoRef]);

  /** Reset the tMs epoch (call at recorder.onstart so tMs≈0 at take start). */
  const resetEpoch = useCallback(() => {
    epochRef.current = null;
  }, []);

  useEffect(
    () => () => {
      stop();
      landmarkerRef.current?.close?.();
      landmarkerRef.current = null;
    },
    [stop],
  );

  return { status, error, stats, start, stop, onFrame, resetEpoch };
}
