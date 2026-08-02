'use client';

import { useCallback, useRef, useState } from 'react';
import { getHttpBase, getToken } from '../components/wsUrl';

// Cloudflare quick tunnels cap request bodies around 100 MB; 16 MB chunks
// stay far under it and retry cheaply on flaky phone connections.
const CHUNK_BYTES = 16 * 1024 * 1024;
const RETRIES = 3;

export type UploadState = 'idle' | 'uploading' | 'done' | 'error';

type UploadArgs = { blob: Blob; filename: string; sessionId: string; role: string };

export function useUploader() {
  const [state, setState] = useState<UploadState>('idle');
  const [progress, setProgress] = useState(0); // 0..1
  const [error, setError] = useState<string | null>(null);
  const lastArgsRef = useRef<UploadArgs | null>(null);

  const upload = useCallback(async (args: UploadArgs) => {
    lastArgsRef.current = args;
    const { blob, filename, sessionId, role } = args;
    setState('uploading');
    setProgress(0);
    setError(null);

    const base = getHttpBase();
    const token = getToken();
    const auth = token ? `&token=${encodeURIComponent(token)}` : '';
    const path = `${base}/upload/${encodeURIComponent(sessionId)}/${encodeURIComponent(
      role,
    )}/${encodeURIComponent(filename)}`;
    const total = Math.max(1, Math.ceil(blob.size / CHUNK_BYTES));

    try {
      for (let i = 0; i < total; i++) {
        const chunk = blob.slice(i * CHUNK_BYTES, (i + 1) * CHUNK_BYTES);
        let lastErr: unknown = null;
        let ok = false;
        for (let attempt = 0; attempt < RETRIES && !ok; attempt++) {
          try {
            const res = await fetch(`${path}?index=${i}&total=${total}${auth}`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/octet-stream' },
              body: chunk,
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            ok = true;
          } catch (err) {
            lastErr = err;
            await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
          }
        }
        if (!ok) throw lastErr ?? new Error('chunk upload failed');
        setProgress((i + 1) / (total + 1));
      }

      const done = await fetch(
        `${path}/complete?size=${blob.size}${auth}`,
        { method: 'POST' },
      );
      if (!done.ok) {
        throw new Error(`finalize failed: HTTP ${done.status}`);
      }
      setProgress(1);
      setState('done');
    } catch (err) {
      setError((err as Error)?.message ?? 'upload failed');
      setState('error');
    }
  }, []);

  const retry = useCallback(() => {
    if (lastArgsRef.current) void upload(lastArgsRef.current);
  }, [upload]);

  return { state, progress, error, upload, retry };
}
