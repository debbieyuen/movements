'use client';

import { useEffect, useRef, useState } from 'react';

type PoseSample = {
  t: number;
  head?: TransformSample;
  leftController?: TransformSample;
  rightController?: TransformSample;
};

type TransformSample = {
  position: [number, number, number];
  rotation: [number, number, number, number];
};

function toTransform(transform: any): TransformSample | undefined {
  if (!transform) return undefined;

  const p = transform.position;
  const r = transform.orientation;

  if (!p || !r) return undefined;

  return {
    position: [p.x, p.y, p.z],
    rotation: [r.x, r.y, r.z, r.w],
  };
}

export default function QuestRecorder({
  sessionId,
  role,
}: {
  sessionId: string;
  role: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const sessionRef = useRef<XRSession | null>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const refSpaceRef = useRef<XRReferenceSpace | null>(null);
  const samplesRef = useRef<PoseSample[]>([]);
  const rafRef = useRef<number | null>(null);

  const [supported, setSupported] = useState<boolean | null>(null);
  const [recording, setRecording] = useState(false);
  const [sampleCount, setSampleCount] = useState(0);
  const [downloadUrl, setDownloadUrl] = useState('');
  const [downloadName, setDownloadName] = useState('');

  useEffect(() => {
    const check = async () => {
      const ok = !!navigator.xr && (await navigator.xr.isSessionSupported('immersive-vr'));
      setSupported(ok);
    };

    check();
  }, []);

  const stopSession = async () => {
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    if (sessionRef.current) {
      try {
        await sessionRef.current.end();
      } catch {
        // ignore
      }
    }

    sessionRef.current = null;
    refSpaceRef.current = null;
    glRef.current = null;
    setRecording(false);
  };

  const downloadLog = () => {
    const blob = new Blob([JSON.stringify(samplesRef.current, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const name = `${sessionId}_${role}_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    setDownloadUrl(url);
    setDownloadName(name);

    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
  };

  const startSession = async () => {
    if (!navigator.xr) return;

    samplesRef.current = [];
    setSampleCount(0);
    setDownloadUrl('');
    setDownloadName('');

    const session = await navigator.xr.requestSession('immersive-vr', {
      requiredFeatures: ['local-floor'],
      optionalFeatures: ['hand-tracking', 'bounded-floor'],
    });

    sessionRef.current = session;

    const canvas = canvasRef.current || document.createElement('canvas');
    canvasRef.current = canvas;

    const gl = canvas.getContext('webgl', { xrCompatible: true });
    if (!gl) {
      alert('WebGL not available for XR logging.');
      await session.end();
      return;
    }

    glRef.current = gl;
    const layer = new XRWebGLLayer(session, gl);
    session.updateRenderState({ baseLayer: layer });

    const refSpace = await session.requestReferenceSpace('local-floor');
    refSpaceRef.current = refSpace;

    setRecording(true);

    const onFrame = (time: DOMHighResTimeStamp, frame: XRFrame) => {
      const s = sessionRef.current;
      const rs = refSpaceRef.current;
      if (!s || !rs) return;

      const pose = frame.getViewerPose(rs);
      if (pose) {
        const headView = pose.views[0];
        const sample: PoseSample = {
          t: time,
          head: toTransform(headView.transform),
        };

        for (const inputSource of s.inputSources) {
          const handedness = inputSource.handedness;
          const space = inputSource.gripSpace || inputSource.targetRaySpace;
          if (!space) continue;

          const inputPose = frame.getPose(space, rs);
          if (!inputPose) continue;

          const t = toTransform(inputPose.transform);
          if (!t) continue;

          if (handedness === 'left') sample.leftController = t;
          if (handedness === 'right') sample.rightController = t;
        }

        samplesRef.current.push(sample);
        setSampleCount(samplesRef.current.length);
      }

      rafRef.current = session.requestAnimationFrame(onFrame);
    };

    session.requestAnimationFrame(onFrame);

    session.addEventListener('end', () => {
      setRecording(false);
    });
  };

  const toggle = async () => {
    if (recording) {
      await stopSession();
    } else {
      await startSession();
    }
  };

  if (supported === false) {
    return (
      <section className="card">
        <h2>Quest recorder</h2>
        <p>Your browser says WebXR immersive-vr is not supported here.</p>
        <div className="hint">
          Try this in Meta Quest Browser on the headset. The cameras can still be recorded on
          the phones and Windows computer.
        </div>
      </section>
    );
  }

  return (
    <section className="card">
      <h2>Quest 2 recorder</h2>
      <p>
        Use this in Meta Quest Browser. It records head pose and controller poses to JSON.
      </p>

      <div className="row wrap">
        <button className="primary" onClick={toggle}>
          {recording ? 'Stop Quest tracking' : 'Start Quest tracking'}
        </button>

        <button className="secondary" onClick={downloadLog} disabled={sampleCount === 0}>
          Download JSON log
        </button>
      </div>

      <div className="metaRow">
        <span>Status: {recording ? 'Tracking' : 'Idle'}</span>
        <span>Samples: {sampleCount}</span>
      </div>

      <canvas ref={canvasRef} className="hiddenCanvas" />

      {downloadUrl ? (
        <div className="row wrap">
          <a href={downloadUrl} download={downloadName}>
            {downloadName}
          </a>
        </div>
      ) : null}

      <div className="hint">
        This stores tracking only. Your iPhones and Windows camera will still provide the body
        video.
      </div>
    </section>
  );
}