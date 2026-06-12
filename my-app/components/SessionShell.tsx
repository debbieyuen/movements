'use client';

// import { useEffect, useState } from 'react';
import { useMemo, useEffect, useState } from 'react';
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

  // const sessionUrl = useMemo(() => {
  //   if (typeof window === 'undefined') return '';
  //   return `${window.location.origin}/session/${sessionId}`;
  // }, [sessionId]);
  const [sessionUrl, setSessionUrl] = useState('');

  useEffect(() => {
    setSessionUrl(`${window.location.origin}/session/${sessionId}`);
  }, [sessionId]);

  const copyLink = async () => {
    if (!sessionUrl) return;
    await navigator.clipboard.writeText(sessionUrl);
    alert('Session link copied');
  };

  return (
    <main className="page">
      <div className="shell">
        <div className="card">
          <h1>Session {sessionId}</h1>
          <p>
            Open this same URL on each device. Pick a role, start recording, and clap once at
            the beginning for sync.
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
        </div>

        {role === 'quest' ? (
          <QuestRecorder sessionId={sessionId} role={role} />
        ) : (
          <CameraRecorder sessionId={sessionId} role={role} />
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