'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

export type CameraError = {
  kind: 'permission-denied' | 'not-found' | 'in-use' | 'overconstrained' | 'unknown';
  message: string;
};

export type CameraSettings = { width: number; height: number; frameRate: number };

const IDEAL = { width: 1920, height: 1080, frameRate: 30 };

function classifyError(err: unknown): CameraError {
  const e = err as { name?: string; message?: string };
  switch (e?.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return {
        kind: 'permission-denied',
        message: 'Camera permission was denied. Allow camera access in the browser and reload.',
      };
    case 'NotFoundError':
      return { kind: 'not-found', message: 'No camera found on this device.' };
    case 'NotReadableError':
    case 'AbortError':
      return {
        kind: 'in-use',
        message: 'The camera is in use by another app. Close it and try again.',
      };
    case 'OverconstrainedError':
      return {
        kind: 'overconstrained',
        message: 'The selected camera does not support the requested resolution.',
      };
    default:
      return { kind: 'unknown', message: e?.message || 'Could not start the camera.' };
  }
}

/**
 * Camera device selection + stream lifecycle.
 *
 * enumerateDevices() returns blank labels before permission is granted, so
 * start() opens a stream first (saved deviceId if any, else front camera),
 * then re-enumerates to populate the picker with real labels.
 */
export function useCamera() {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceIdState] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [actual, setActual] = useState<CameraSettings | null>(null);
  const [facing, setFacing] = useState<'user' | 'environment' | 'unknown'>('unknown');
  const [error, setError] = useState<CameraError | null>(null);
  // Narration audio: on by default (speech paired with motion is training
  // signal). If the mic is denied, video capture proceeds without it.
  const [audio, setAudioState] = useState(true);
  const [hasAudio, setHasAudio] = useState(false);
  const audioRef = useRef(true);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    try {
      // localStorage is browser-only: initializing after mount avoids a
      // hydration mismatch.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setDeviceIdState(localStorage.getItem('cameraId'));
    } catch {
      // ignore
    }
  }, []);

  const refreshDevices = useCallback(async () => {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      setDevices(all.filter((d) => d.kind === 'videoinput'));
    } catch {
      // enumerateDevices unsupported -> picker just stays empty
    }
  }, []);

  useEffect(() => {
    navigator.mediaDevices?.addEventListener?.('devicechange', refreshDevices);
    return () =>
      navigator.mediaDevices?.removeEventListener?.('devicechange', refreshDevices);
  }, [refreshDevices]);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setStream(null);
    setActual(null);
  }, []);

  const start = useCallback(
    async (overrideDeviceId?: string) => {
      stop();
      setError(null);
      const id = overrideDeviceId ?? deviceId;
      const wantAudio = audioRef.current;

      const videoConstraints = {
        ...(id ? { deviceId: { exact: id } } : { facingMode: { ideal: 'user' as const } }),
        width: { ideal: IDEAL.width },
        height: { ideal: IDEAL.height },
        frameRate: { ideal: IDEAL.frameRate },
      };
      // Echo cancellation etc. off: this is narration into a room mic, not a
      // call, and the processing distorts speech the team may train on.
      const audioConstraints = {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: true,
      };

      let s: MediaStream | null = null;
      // Try video+audio first; on ANY failure retry the exact same video
      // constraints without audio, so a denied mic never blocks capture.
      for (const attempt of [
        { audio: wantAudio ? audioConstraints : false, video: videoConstraints },
        { audio: false as const, video: videoConstraints },
        {
          audio: false as const,
          video: id ? { deviceId: { exact: id } } : { facingMode: { ideal: 'user' as const } },
        },
      ]) {
        try {
          s = await navigator.mediaDevices.getUserMedia(attempt);
          break;
        } catch (err) {
          const classified = classifyError(err);
          if (attempt.audio === false && classified.kind !== 'overconstrained') {
            setError(classified);
            return null;
          }
        }
      }
      if (!s) {
        setError({ kind: 'unknown', message: 'Could not start the camera.' });
        return null;
      }
      setHasAudio(s.getAudioTracks().length > 0);

      streamRef.current = s;
      setStream(s);

      const track = s.getVideoTracks()[0];
      const st = track?.getSettings?.() ?? {};
      setActual({
        width: st.width ?? 0,
        height: st.height ?? 0,
        frameRate: st.frameRate ?? 0,
      });
      setFacing(
        st.facingMode === 'user' || st.facingMode === 'environment'
          ? st.facingMode
          : 'unknown',
      );
      if (track?.getSettings?.().deviceId && !id) {
        setDeviceIdState(track.getSettings().deviceId!);
      }
      await refreshDevices(); // labels are populated now that we have permission
      return s;
    },
    [deviceId, refreshDevices, stop],
  );

  const setDeviceId = useCallback(
    (id: string) => {
      setDeviceIdState(id);
      try {
        localStorage.setItem('cameraId', id);
      } catch {
        // ignore
      }
      if (streamRef.current) void start(id); // hot-swap if already previewing
    },
    [start],
  );

  const setAudio = useCallback(
    (on: boolean) => {
      audioRef.current = on;
      setAudioState(on);
      if (streamRef.current) void start(); // hot-restart to add/drop the mic
    },
    [start],
  );

  useEffect(() => stop, [stop]); // release the camera on unmount

  return {
    devices,
    deviceId,
    setDeviceId,
    stream,
    actual,
    facing,
    error,
    audio,
    hasAudio,
    setAudio,
    start,
    stop,
  };
}
