'use client';

import { useEffect, useRef, useState } from 'react';
import CameraRecorder from './CameraRecorder';
import { getToken, getWsEndpoint, getWsUrl, normalizeWsUrl, setToken } from './wsUrl';
import { PROTOCOL_VERSION } from '../lib/protocol';

const ROLE = 'camera';

export default function SessionShell({ sessionId }: { sessionId: string }) {
  // --- Presence: a lightweight socket, separate from the recording socket, so
  // the device announces itself on page load and the host can see it's live
  // BEFORE recording.
  const [connected, setConnected] = useState(false);
  const presenceWsRef = useRef<WebSocket | null>(null);

  // Countdown overlay ('3' | '2' | '1' | 'GO!' | null) + a trigger the
  // recorder watches to auto-start when the host presses Go.
  const [countdown, setCountdown] = useState<number | string | null>(null);
  const [recordSignal, setRecordSignal] = useState(0);

  // Server (tunnel) URL + token, editable at runtime -- no rebuild needed.
  const [wsInput, setWsInput] = useState('');
  const [tokenInput, setTokenInput] = useState('');
  useEffect(() => {
    // localStorage is browser-only: initializing after mount avoids a
    // hydration mismatch on the controlled inputs.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setWsInput(getWsUrl());
     
    setTokenInput(getToken());
  }, []);

  const [sessionUrl, setSessionUrl] = useState('');
  useEffect(() => {
    // window.location is browser-only; see note above.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSessionUrl(`${window.location.origin}/session/${sessionId}`);
  }, [sessionId]);

  // Open the presence socket on load; announce + heartbeat; listen for the
  // countdown broadcast. Reconnects if it drops.
  useEffect(() => {
    let closed = false;

    // Runs when the host presses Go. Start recording FIRST, then show the
    // 3 -> 2 -> 1 -> GO! cue (kept from the multi-camera days -- it is still a
    // nice "get in position" cue even without clap sync).
    const runCountdown = (secs: number) => {
      setRecordSignal((s) => s + 1);
      let n = secs;
      setCountdown(n);
      const iv = window.setInterval(() => {
        n -= 1;
        if (n > 0) setCountdown(n);
        else if (n === 0) setCountdown('GO!');
        else {
          window.clearInterval(iv);
          setCountdown(null);
        }
      }, 1000);
    };

    const connect = () => {
      const ws = new WebSocket(getWsEndpoint());
      presenceWsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        ws.send(
          JSON.stringify({
            v: PROTOCOL_VERSION,
            type: 'hello',
            role: ROLE,
            sessionId,
            clientUnixMs: Date.now(),
          }),
        );
      };
      ws.onmessage = (e) => {
        try {
          const m = JSON.parse(e.data);
          if (m.type === 'countdown') runCountdown(m.seconds || 3);
        } catch {
          // ignore non-JSON messages
        }
      };
      ws.onclose = () => {
        setConnected(false);
        if (!closed) window.setTimeout(connect, 1000);
      };
      ws.onerror = () => {
        try {
          ws.close();
        } catch {
          // ignore
        }
      };
    };

    connect();
    const heartbeat = window.setInterval(() => {
      const ws = presenceWsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(
          JSON.stringify({
            v: PROTOCOL_VERSION,
            type: 'heartbeat',
            role: ROLE,
            sessionId,
          }),
        );
      }
    }, 2000);

    return () => {
      closed = true;
      window.clearInterval(heartbeat);
      presenceWsRef.current?.close();
    };
  }, [sessionId]);

  const copyLink = async () => {
    if (!sessionUrl) return;
    let link = sessionUrl;
    try {
      const params = new URLSearchParams();
      const w = localStorage.getItem('wsUrl');
      if (w) params.set('ws', w);
      const t = getToken();
      if (t) params.set('token', t);
      const qs = params.toString();
      if (qs) link += '?' + qs;
    } catch {
      // ignore
    }
    await navigator.clipboard.writeText(link);
    alert('Session link copied (includes the server URL + token for other devices)');
  };

  // Save the tunnel URL + token and reload so every socket picks them up.
  const setServer = () => {
    const n = normalizeWsUrl(wsInput);
    try {
      localStorage.setItem('wsUrl', n);
    } catch {
      // ignore
    }
    setToken(tokenInput);
    window.location.reload();
  };

  // Host presses Go: server broadcasts a countdown to this session's devices.
  const pressGo = () => {
    const ws = presenceWsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({ v: PROTOCOL_VERSION, type: 'go', sessionId, seconds: 3 }),
      );
    } else {
      alert('Not connected yet — check the server URL and token.');
    }
  };

  return (
    <main className="page">
      {countdown !== null && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'rgba(0, 0, 0, 0.65)',
            zIndex: 1000,
            pointerEvents: 'none',
          }}
        >
          <div
            style={{
              fontSize: countdown === 'GO!' ? '16vh' : '28vh',
              fontWeight: 800,
              color: countdown === 'GO!' ? '#4ade80' : '#ffffff',
              textShadow: '0 4px 24px rgba(0,0,0,0.6)',
            }}
          >
            {countdown === 'GO!' ? '🎬 GO!' : countdown}
          </div>
        </div>
      )}

      <div className="shell">
        <div className="card">
          <h1>Session {sessionId}</h1>
          <p>One camera is all you need — keep your full body in frame.</p>

          <div className="row wrap">
            <button className="secondary" onClick={copyLink}>
              Copy session link
            </button>
            <span title={connected ? 'connected' : 'not connected'}>
              {connected ? '🟢 server connected' : '⚪ server offline'}
            </span>
          </div>

          <div className="row wrap" style={{ marginTop: 12, gap: 8, alignItems: 'center' }}>
            <strong>Server URL:</strong>
            <input
              value={wsInput}
              onChange={(e) => setWsInput(e.target.value)}
              placeholder="https://xxxx.trycloudflare.com"
              style={{ minWidth: 260, padding: '4px 8px' }}
            />
            <strong>Token:</strong>
            <input
              value={tokenInput}
              onChange={(e) => setTokenInput(e.target.value)}
              placeholder="printed by the server at startup"
              style={{ minWidth: 180, padding: '4px 8px' }}
            />
            <button className="secondary" onClick={setServer}>
              Set
            </button>
            <span className="roleHint">
              paste your tunnel URL + token, then Set (no redeploy)
            </span>
          </div>

          <div className="row wrap" style={{ marginTop: 12 }}>
            <button className="primary" onClick={pressGo} disabled={countdown !== null}>
              ▶ Go — countdown + record
            </button>
            <span className="roleHint" style={{ marginLeft: 8 }}>
              3-2-1 cue, then recording starts automatically.
            </span>
          </div>
        </div>

        <CameraRecorder sessionId={sessionId} role={ROLE} recordSignal={recordSignal} />

        <div className="grid2">
          <div className="card">
            <h2>Capture checklist</h2>
            <ul>
              <li>Camera on a tripod or stable surface (static camera)</li>
              <li>Full body in frame the whole take, one person only</li>
              <li>Landscape orientation, bright even lighting</li>
              <li>Keep takes under ~60 seconds</li>
            </ul>
          </div>

          <div className="card">
            <h2>After recording</h2>
            <ul>
              <li>The clip + keypoints upload to the server automatically</li>
              <li>Process on the GPU box: <code>python -m mocap.process &lt;video&gt;</code></li>
              <li>Check the preview.mp4 before sharing the clip</li>
            </ul>
          </div>
        </div>
      </div>
    </main>
  );
}
