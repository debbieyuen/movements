'use client';

// import { useEffect, useState } from 'react';
import { useMemo, useEffect, useRef, useState } from 'react';
import CameraRecorder from './CameraRecorder';
import QuestRecorder from './QuestRecorder';

type Role = 'left-phone' | 'front-camera' | 'right-phone' | 'quest';

const roles: { id: Role; label: string; help: string }[] = [
  { id: 'left-phone', label: 'Left iPhone', help: 'Left-side view' },
  { id: 'front-camera', label: 'Windows / Front Camera', help: 'Front view' },
  { id: 'right-phone', label: 'Right iPhone', help: 'Right-side view' },
  { id: 'quest', label: 'Quest 2', help: 'Head + controller tracking' },
];

export default function SessionShell({ sessionId }: { sessionId: string }) {
  const [role, setRole] = useState<Role>('front-camera');

  // --- Presence: a lightweight socket, separate from the recording socket, so
  // each device announces its role on page load and the host can see which
  // cameras are live BEFORE recording. Purely additive; recording is untouched.
  const [liveRoles, setLiveRoles] = useState<string[]>([]);
  const presenceWsRef = useRef<WebSocket | null>(null);
  const roleRef = useRef<Role>(role); // avoid stale role in the socket callbacks
  roleRef.current = role;

  // Countdown overlay ('3' | '2' | '1' | 'CLAP!' | null) + a trigger the
  // recorder watches to auto-start when the host presses Go.
  const [countdown, setCountdown] = useState<number | string | null>(null);
  const [recordSignal, setRecordSignal] = useState(0);

  // const sessionUrl = useMemo(() => {
  //   if (typeof window === 'undefined') return '';
  //   return `${window.location.origin}/session/${sessionId}`;
  // }, [sessionId]);
  const [sessionUrl, setSessionUrl] = useState('');

  useEffect(() => {
    setSessionUrl(`${window.location.origin}/session/${sessionId}`);
  }, [sessionId]);

  // Open the presence socket on load; announce our role + heartbeat; listen for
  // the server's live-roles broadcast. Reconnects if it drops.
  useEffect(() => {
    let closed = false;
    let heartbeat: number | undefined;

    // Runs on every device when the host presses Go. Start recording FIRST (so
    // the clap is captured), then show 3 -> 2 -> 1 -> CLAP! as the cue.
    const runCountdown = (secs: number) => {
      setRecordSignal((s) => s + 1);
      let n = secs;
      setCountdown(n);
      const iv = window.setInterval(() => {
        n -= 1;
        if (n > 0) setCountdown(n);
        else if (n === 0) setCountdown('CLAP!');
        else {
          window.clearInterval(iv);
          setCountdown(null);
        }
      }, 1000);
    };

    const connect = () => {
      // window.location.hostname works for phones too (they load from the LAN IP).
      const ws = new WebSocket(`ws://${window.location.hostname}:8765`);
      presenceWsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ type: 'hello', role: roleRef.current, sessionId }));
      };
      ws.onmessage = (e) => {
        try {
          const m = JSON.parse(e.data);
          if (m.type === 'presence') setLiveRoles(m.roles || []);
          else if (m.type === 'countdown') runCountdown(m.seconds || 3);
        } catch {
          // ignore non-JSON / non-presence messages
        }
      };
      ws.onclose = () => {
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
    heartbeat = window.setInterval(() => {
      const ws = presenceWsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'heartbeat', role: roleRef.current, sessionId }));
      }
    }, 2000);

    return () => {
      closed = true;
      if (heartbeat) window.clearInterval(heartbeat);
      presenceWsRef.current?.close();
    };
  }, [sessionId]);

  // Re-announce immediately when the role dropdown changes.
  useEffect(() => {
    const ws = presenceWsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'hello', role, sessionId }));
    }
  }, [role, sessionId]);

  const copyLink = async () => {
    if (!sessionUrl) return;
    await navigator.clipboard.writeText(sessionUrl);
    alert('Session link copied');
  };

  // Host presses Go: tell the server to broadcast a countdown to every device.
  const pressGo = () => {
    const ws = presenceWsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'go', seconds: 3 }));
    } else {
      alert('Not connected yet — wait for the cameras to show live.');
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
              fontSize: countdown === 'CLAP!' ? '16vh' : '28vh',
              fontWeight: 800,
              color: countdown === 'CLAP!' ? '#4ade80' : '#ffffff',
              textShadow: '0 4px 24px rgba(0,0,0,0.6)',
            }}
          >
            {countdown === 'CLAP!' ? '👏 CLAP!' : countdown}
          </div>
        </div>
      )}

      <div className="shell">
        <div className="card">
          <h1>Session {sessionId}</h1>
          <p>
            Open this URL on each device. Best to have at least 3 devices (front, left, right)
          </p>

          <div className="row wrap">
            <button className="secondary" onClick={copyLink}>
              Copy session link
            </button>

            <select value={role} onChange={(e) => setRole(e.target.value as Role)}>
              {roles.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.label}
                </option>
              ))}
            </select>
          </div>

          <div className="roleHint">
            {roles.find((r) => r.id === role)?.help}
          </div>

          <div className="row wrap" style={{ marginTop: 12, gap: 16 }}>
            <strong style={{ marginRight: 4 }}>Cameras live:</strong>
            {roles.map((r) => (
              <span key={r.id} title={liveRoles.includes(r.id) ? 'connected' : 'not connected'}>
                {liveRoles.includes(r.id) ? '🟢' : '⚪'} {r.label}
              </span>
            ))}
          </div>

          <div className="row wrap" style={{ marginTop: 12 }}>
            <button className="primary" onClick={pressGo} disabled={countdown !== null}>
              ▶ Go — start all cameras + clap
            </button>
            <span className="roleHint" style={{ marginLeft: 8 }}>
              Starts every connected device recording, then cues a clap to sync them.
            </span>
          </div>
        </div>

        {role === 'quest' ? (
          <QuestRecorder sessionId={sessionId} role={role} />
        ) : (
          <CameraRecorder sessionId={sessionId} role={role} recordSignal={recordSignal} />
        )}

        <div className="grid2">
          <div className="card">
            <h2>Recommended setup</h2>
            <ul>
              <li>iPhone left: side angle</li>
              <li>iPhone front: front angle</li>
              <li>iPhone right or laptop: second side angle</li>
              <li>Quest 2: head and hand tracking</li>
            </ul>
          </div>

          <div className="card">
            <h2>Capture notes</h2>
            <ul>
              <li>Use landscape video on phones</li>
              <li>Keep full body in frame</li>
              <li>Use bright lighting</li>
              <li>Do a clap at the start</li>
            </ul>
          </div>
        </div>
      </div>
    </main>
  );
}