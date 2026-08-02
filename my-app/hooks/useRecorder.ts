'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

export type RecordingResult = {
  blob: Blob;
  mimeType: string;
  durationMs: number;
};

function pickMimeType(): string {
  if (typeof MediaRecorder === 'undefined') return '';
  const candidates = [
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
    'video/mp4', // iOS Safari
  ];
  return candidates.find((t) => MediaRecorder.isTypeSupported(t)) || '';
}

/**
 * MediaRecorder lifecycle. t=0 is stamped in recorder.onstart (not before
 * start()), and 1 s timeslices keep long takes from building one giant
 * in-memory blob that only materializes at stop. `start` takes the stream at
 * call time so callers that just opened the camera don't race React state.
 */
export function useRecorder(onStart?: () => void) {
  const [isRecording, setIsRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startedAtRef = useRef<number | null>(null);
  const timerRef = useRef<number | null>(null);
  const stopResolveRef = useRef<((r: RecordingResult) => void) | null>(null);
  const onStartRef = useRef(onStart);
  useEffect(() => {
    onStartRef.current = onStart;
  }, [onStart]);

  const start = useCallback((stream: MediaStream) => {
    if (!stream || recorderRef.current) return;
    setError(null);
    chunksRef.current = [];
    setSeconds(0);

    let recorder: MediaRecorder;
    const mimeType = pickMimeType();
    try {
      recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
    } catch (err) {
      setError(`Recording not supported: ${(err as Error).message}`);
      return;
    }

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorder.onstart = () => {
      startedAtRef.current = performance.now();
      setIsRecording(true);
      onStartRef.current?.();
      timerRef.current = window.setInterval(() => setSeconds((s) => s + 1), 1000);
    };
    recorder.onerror = (e) => {
      const err = (e as { error?: { message?: string } }).error;
      setError(`Recording error: ${err?.message ?? 'unknown'}`);
    };
    recorder.onstop = () => {
      const type = recorder.mimeType || mimeType || 'video/webm';
      const durationMs = startedAtRef.current
        ? performance.now() - startedAtRef.current
        : 0;
      const result: RecordingResult = {
        blob: new Blob(chunksRef.current, { type }),
        mimeType: type,
        durationMs,
      };
      chunksRef.current = [];
      startedAtRef.current = null;
      stopResolveRef.current?.(result);
      stopResolveRef.current = null;
    };

    recorder.start(1000); // 1 s timeslices
    recorderRef.current = recorder;
  }, []);

  const stop = useCallback((): Promise<RecordingResult | null> => {
    const recorder = recorderRef.current;
    if (!recorder) return Promise.resolve(null);
    recorderRef.current = null;
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsRecording(false);
    return new Promise((resolve) => {
      stopResolveRef.current = resolve as (r: RecordingResult) => void;
      try {
        recorder.stop();
      } catch {
        resolve(null);
      }
    });
  }, []);

  useEffect(
    () => () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
      try {
        recorderRef.current?.stop();
      } catch {
        // already stopped
      }
    },
    [],
  );

  return { isRecording, seconds, error, start, stop };
}
