'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { getWsEndpoint } from '../components/wsUrl';
import { PROTOCOL_VERSION, POSE_SOURCE, toLandmark4 } from '../lib/protocol';
import type { PoseResult } from './usePose';

export type SocketStatus = 'disconnected' | 'connecting' | 'open';

/**
 * Streaming socket for pose frames ("Live to robot").
 *
 * Streaming is controlled by the `live` toggle, INDEPENDENT of recording:
 * the old client only opened the socket inside startRecording, so the first
 * ~half second of every take was silently dropped while it connected, and
 * the live demo could not run without recording.
 */
export function usePoseSocket(sessionId: string, role: string) {
  const [status, setStatus] = useState<SocketStatus>('disconnected');
  const [live, setLiveState] = useState(false);
  const [framesSent, setFramesSent] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const liveRef = useRef(false);
  const seqRef = useRef(0);
  const sentRef = useRef(0);
  const roleRef = useRef(role);
  useEffect(() => {
    roleRef.current = role;
  }, [role]);

  const openSocket = useCallback(() => {
    function connect() {
      if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) return;
      setStatus('connecting');
      const ws = new WebSocket(getWsEndpoint());
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus('open');
        ws.send(
          JSON.stringify({
            v: PROTOCOL_VERSION,
            type: 'hello',
            sessionId,
            role: roleRef.current,
            clientUnixMs: Date.now(),
          }),
        );
      };
      ws.onclose = () => {
        if (wsRef.current === ws) wsRef.current = null;
        setStatus('disconnected');
        // Reconnect while live so the robot resumes instead of freezing.
        if (liveRef.current) {
          window.setTimeout(() => {
            if (liveRef.current) connect();
          }, 500);
        }
      };
      ws.onerror = () => {
        try {
          ws.close();
        } catch {
          // ignore
        }
      };
    }
    connect();
  }, [sessionId]);

  const setLive = useCallback(
    (on: boolean) => {
      liveRef.current = on;
      setLiveState(on);
      if (on) {
        seqRef.current = 0;
        openSocket();
      }
    },
    [openSocket],
  );

  const sendFrame = useCallback(
    (r: PoseResult) => {
      const ws = wsRef.current;
      if (!liveRef.current || !ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(
        JSON.stringify({
          v: PROTOCOL_VERSION,
          type: 'pose',
          sessionId,
          role: roleRef.current,
          seq: seqRef.current++,
          tMs: r.tCaptureMs,
          unixMs: Date.now(),
          coord: 'mp-camera',
          source: POSE_SOURCE,
          world: toLandmark4(r.worldLandmarks),
          // `norm` omitted: the overlay is drawn locally; ~45% smaller frames.
        }),
      );
      sentRef.current += 1;
    },
    [sessionId],
  );

  // Publish the sent-counter at 2 Hz instead of per frame.
  useEffect(() => {
    const iv = window.setInterval(() => {
      setFramesSent((prev) => (prev === sentRef.current ? prev : sentRef.current));
    }, 500);
    return () => window.clearInterval(iv);
  }, []);

  useEffect(
    () => () => {
      liveRef.current = false;
      try {
        wsRef.current?.close();
      } catch {
        // ignore
      }
    },
    [],
  );

  return { status, live, setLive, sendFrame, framesSent };
}
