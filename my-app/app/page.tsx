'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';

export default function HomePage() {
  const router = useRouter();
  const [sessionId, setSessionId] = useState('');

  const createSession = () => {
    const id = crypto.randomUUID().slice(0, 8);
    router.push(`/session/${id}`);
  };

  const joinSession = () => {
    const id = sessionId.trim();
    if (!id) return;
    router.push(`/session/${id}`);
  };

  return (
    <main className="page">
      <div className="shell">
        <div className="hero card">
          <h1>Motion Capture Studio</h1>
          <p>
            Open the same session on your iPhones, Windows computer, and Quest 2.
            Record locally on each device, then use the clips for mocap and retargeting.
          </p>

          <div className="row">
            <button className="primary" onClick={createSession}>
              Create session
            </button>
          </div>

          <div className="joinBox">
            <input
              value={sessionId}
              onChange={(e) => setSessionId(e.target.value)}
              placeholder="Enter session code"
            />
            <button className="secondary" onClick={joinSession}>
              Join session
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}