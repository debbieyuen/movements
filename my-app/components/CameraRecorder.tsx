'use client';

import { useEffect, useRef, useState } from 'react';

type Landmark = {
  x: number;
  y: number;
  z: number;
  visibility?: number;
  presence?: number;
};

type PoseFrame = {
  frameIndex: number;
  t: number;
  timeMs: number;
  unixMs: number;
  landmarks: Landmark[];
  worldLandmarks: Landmark[];
};

function pickMimeType() {
  if (typeof window === 'undefined' || typeof MediaRecorder === 'undefined') return '';

  const candidates = [
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
    'video/mp4',
  ];

  return candidates.find((t) => MediaRecorder.isTypeSupported(t)) || '';
}

function serializeLandmarks(landmarks: any[] | undefined): Landmark[] {
  if (!landmarks) return [];

  return landmarks.map((l) => ({
    x: l.x,
    y: l.y,
    z: l.z,
    visibility: typeof l.visibility === 'number' ? l.visibility : undefined,
    presence: typeof l.presence === 'number' ? l.presence : undefined,
  }));
}

function makeFilename(sessionId: string, role: string, ext: string) {
  return `${sessionId}_${role}_${new Date().toISOString().replace(/[:.]/g, '-')}.${ext}`;
}

export default function CameraRecorder({
  sessionId,
  role,
  recordSignal,
}: {
  sessionId: string;
  role: string;
  recordSignal?: number;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);

  const poseRef = useRef<any>(null);
  const drawUtilsRef = useRef<any>(null);
  const poseLoopRef = useRef<number | null>(null);
  const poseBusyRef = useRef(false);
  const poseRunningRef = useRef(false);
  const recordingRef = useRef(false);
  const recordingStartRef = useRef<number | null>(null);
  const poseFramesRef = useRef<PoseFrame[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const liveFrameIndexRef = useRef(0);

  // The pose callback is registered once, so it captures the initial `role`.
  // Mirror the current role into a ref (stable identity) so streamed frames are
  // always tagged with the role currently selected in the dropdown.
  const roleRef = useRef(role);
  roleRef.current = role;

  const [mounted, setMounted] = useState(false);
  const [mimeType, setMimeType] = useState('');
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user');
  const [isPreviewOn, setIsPreviewOn] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [downloadUrl, setDownloadUrl] = useState('');
  const [downloadName, setDownloadName] = useState('');
  const [keypointsUrl, setKeypointsUrl] = useState('');
  const [keypointsName, setKeypointsName] = useState('');
  const [keypointCount, setKeypointCount] = useState(0);
  const [poseReady, setPoseReady] = useState(false);

  useEffect(() => {
    setMounted(true);
    setMimeType(pickMimeType());
  }, []);

  const clearOverlay = () => {
    const canvas = overlayRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const openSocket = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

    // Use the page's own host so recording works from phones on the LAN too
    // (127.0.0.1 on a phone means the phone itself, not the server).
    const ws = new WebSocket(`ws://${window.location.hostname}:8765`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      console.log('WebSocket server:', event.data);
    };

    ws.onerror = (event) => {
      console.error('WebSocket error:', event);
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
      if (wsRef.current === ws) {
        wsRef.current = null;
      }
      // If the socket drops while we're still recording (e.g. the server
      // hiccups on a frame), reconnect so streaming — and the robot — resume
      // instead of silently freezing on the last frame.
      if (recordingRef.current) {
        console.log('Still recording; reconnecting WebSocket in 500ms...');
        window.setTimeout(() => {
          if (recordingRef.current) openSocket();
        }, 500);
      }
    };
  };

  const closeSocket = () => {
    try {
      wsRef.current?.close();
    } catch {
      // ignore
    }
    wsRef.current = null;
  };

  const sendPoseFrame = (results: any) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    const payload = {
      sessionId,
      role: roleRef.current,
      frameIndex: liveFrameIndexRef.current++,
      timeMs: performance.now(),
      unixMs: Date.now(),
      landmarks: serializeLandmarks(results.poseLandmarks),
      worldLandmarks: serializeLandmarks(results.poseWorldLandmarks),
    };

    ws.send(JSON.stringify(payload));
  };

  const ensurePose = async () => {
    if (poseRef.current) return poseRef.current;

    const poseModule: any = await import('@mediapipe/pose');
    const drawingModule: any = await import('@mediapipe/drawing_utils');

    const PoseCtor =
      poseModule.Pose ?? poseModule.default?.Pose ?? poseModule.default ?? null;

    if (!PoseCtor) {
      throw new Error('Could not load MediaPipe Pose constructor.');
    }

    const drawingUtils = drawingModule.default ?? drawingModule;

    const pose = new PoseCtor({
      locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });

    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      smoothSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    pose.onResults((results: any) => {
      const video = videoRef.current;
      const overlay = overlayRef.current;
      if (!video || !overlay) return;

      const ctx = overlay.getContext('2d');
      if (!ctx) return;

      const width = video.videoWidth || 1280;
      const height = video.videoHeight || 720;

      if (overlay.width !== width) overlay.width = width;
      if (overlay.height !== height) overlay.height = height;

      ctx.clearRect(0, 0, width, height);

      if (results.poseLandmarks) {
        const drawing: any = drawUtilsRef.current ?? drawingUtils;

        const drawConnectors =
          drawing.drawConnectors ?? drawing.default?.drawConnectors;
        const drawLandmarks =
          drawing.drawLandmarks ?? drawing.default?.drawLandmarks;
        const POSE_CONNECTIONS =
          drawing.POSE_CONNECTIONS ?? drawing.default?.POSE_CONNECTIONS;

        if (drawConnectors && drawLandmarks && POSE_CONNECTIONS) {
          drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {
            color: '#00ff88',
            lineWidth: 6,
          });

          drawLandmarks(ctx, results.poseLandmarks, {
            color: '#ffcc00',
            lineWidth: 2,
            radius: 4,
          });
        }

        const lm = results.poseLandmarks as any[];
        const labels = [
          { i: 0, name: 'nose' },
          { i: 11, name: 'left shoulder' },
          { i: 12, name: 'right shoulder' },
          { i: 13, name: 'left elbow' },
          { i: 14, name: 'right elbow' },
          { i: 15, name: 'left wrist' },
          { i: 16, name: 'right wrist' },
          { i: 23, name: 'left hip' },
          { i: 24, name: 'right hip' },
          { i: 25, name: 'left knee' },
          { i: 26, name: 'right knee' },
          { i: 27, name: 'left ankle' },
          { i: 28, name: 'right ankle' },
        ];

        ctx.font = '16px Arial';
        ctx.lineWidth = 4;

        for (const item of labels) {
          const p = lm[item.i];
          if (!p) continue;
          if (typeof p.visibility === 'number' && p.visibility < 0.5) continue;

          const x = p.x * width;
          const y = p.y * height;

          ctx.strokeStyle = '#000000';
          ctx.fillStyle = '#ffffff';
          ctx.strokeText(item.name, x + 6, y - 6);
          ctx.fillText(item.name, x + 6, y - 6);
        }
      }

      if (recordingRef.current && results.poseLandmarks) {
        const now = performance.now();
        const start = recordingStartRef.current ?? now;
        const frameIndex = poseFramesRef.current.length;

        const frame: PoseFrame = {
          frameIndex,
          t: now,
          timeMs: now - start,
          unixMs: Date.now(),
          landmarks: serializeLandmarks(results.poseLandmarks),
          worldLandmarks: serializeLandmarks(results.poseWorldLandmarks),
        };

        poseFramesRef.current.push(frame);
        setKeypointCount(poseFramesRef.current.length);
      }

      // sendPoseFrame(results);
      if (recordingRef.current) {
        sendPoseFrame(results);
      }
    });

    poseRef.current = pose;
    drawUtilsRef.current = drawingUtils;
    setPoseReady(true);

    return pose;
  };

  const startPoseLoop = async () => {
    if (!poseRunningRef.current) return;
    if (!poseRef.current || !videoRef.current) return;

    if (
      !poseBusyRef.current &&
      videoRef.current.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
    ) {
      poseBusyRef.current = true;
      try {
        await poseRef.current.send({ image: videoRef.current });
      } catch (err) {
        console.error('Pose detection error:', err);
      } finally {
        poseBusyRef.current = false;
      }
    }

    poseLoopRef.current = window.requestAnimationFrame(startPoseLoop);
  };

  const stopPreview = () => {
    poseRunningRef.current = false;

    if (poseLoopRef.current != null) {
      window.cancelAnimationFrame(poseLoopRef.current);
      poseLoopRef.current = null;
    }

    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;

    if (videoRef.current) videoRef.current.srcObject = null;

    closeSocket();

    setIsPreviewOn(false);
    clearOverlay();
  };

  const startPreview = async () => {
    stopPreview();

    await ensurePose();

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: facingMode },
      },
      audio: false,
    });

    streamRef.current = stream;

    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      try {
        await videoRef.current.play();
      } catch (err) {
        // A rapid stop/start (or the autoPlay attribute) can interrupt an
        // in-flight play() with an AbortError. That is benign — the newer
        // load wins — so swallow it and rethrow anything else.
        if ((err as { name?: string })?.name !== 'AbortError') throw err;
      }
    }

    // openSocket();

    poseRunningRef.current = true;
    poseLoopRef.current = window.requestAnimationFrame(startPoseLoop);
    setIsPreviewOn(true);
  };

  const downloadBlob = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    return url;
  };

  const startRecording = async () => {
    if (!streamRef.current) {
      await startPreview();
    }

    const stream = streamRef.current;
    if (!stream) return;

    openSocket();

    chunksRef.current = [];
    poseFramesRef.current = [];
    setKeypointCount(0);
    setSeconds(0);
    setDownloadUrl('');
    setDownloadName('');
    setKeypointsUrl('');
    setKeypointsName('');
    recordingStartRef.current = performance.now();
    liveFrameIndexRef.current = 0;


    recordingRef.current = true;

    const recorder = mimeType
      ? new MediaRecorder(stream, { mimeType })
      : new MediaRecorder(stream);

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    };

    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, {
        type: mimeType || 'video/webm',
      });

      const ext = blob.type.includes('mp4') ? 'mp4' : 'webm';
      const name = makeFilename(sessionId, role, ext);
      const url = downloadBlob(blob, name);

      setDownloadUrl(url);
      setDownloadName(name);

      const keypointBlob = new Blob([JSON.stringify(poseFramesRef.current, null, 2)], {
        type: 'application/json',
      });
      const keypointName = makeFilename(sessionId, `${role}_keypoints`, 'json');
      const keypointUrl = downloadBlob(keypointBlob, keypointName);

      setKeypointsUrl(keypointUrl);
      setKeypointsName(keypointName);
    };

    recorder.start();
    recorderRef.current = recorder;
    setIsRecording(true);

    timerRef.current = window.setInterval(() => {
      setSeconds((s) => s + 1);
    }, 1000);
  };

  const stopRecording = () => {
    recordingRef.current = false;

    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }

    recorderRef.current?.stop();
    recorderRef.current = null;
    recordingStartRef.current = null;

    closeSocket();
    setIsRecording(false);
  };

  useEffect(() => {
    return () => {
      recordingRef.current = false;
      poseRunningRef.current = false;

      if (timerRef.current) window.clearInterval(timerRef.current);
      if (poseLoopRef.current != null) window.cancelAnimationFrame(poseLoopRef.current);

      streamRef.current?.getTracks().forEach((t) => t.stop());
      stopPreview();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Host's countdown fires: recordSignal increments -> auto-start recording.
  // Ignores the initial value and no-ops if already recording.
  useEffect(() => {
    if (recordSignal && recordSignal > 0 && !recordingRef.current) {
      startRecording();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recordSignal]);

  return (
    <section className="card">
      <h2>Camera recorder</h2>
      <p>
        Use this on the iPhones and the Windows computer. Each device records locally in the
        browser, and a live skeleton overlay shows detected joints.
      </p>

      <div className="row wrap">
        <label>
          Facing mode
          <select
            value={facingMode}
            onChange={(e) => setFacingMode(e.target.value as 'user' | 'environment')}
            disabled={isRecording}
          >
            <option value="user">Front camera</option>
            <option value="environment">Back camera</option>
          </select>
        </label>

        <button className="secondary" onClick={startPreview} disabled={isRecording}>
          Start preview
        </button>

        <button className="primary" onClick={startRecording} disabled={isRecording}>
          Record
        </button>

        <button className="danger" onClick={stopRecording} disabled={!isRecording}>
          Stop
        </button>
      </div>

      <div className="metaRow">
        <span>Status: {isRecording ? 'Recording' : isPreviewOn ? 'Preview ready' : 'Idle'}</span>
        <span>Elapsed: {seconds}s</span>
        <span>Mime: {mounted ? mimeType || 'browser default' : 'loading...'}</span>
        <span>Pose: {poseReady ? 'Ready' : 'Loading...'}</span>
        <span>Frames: {keypointCount}</span>
      </div>

      <div style={{ position: 'relative', width: '100%' }}>
        <video ref={videoRef} className="preview" playsInline muted autoPlay />
        <canvas
          ref={overlayRef}
          className="preview"
          style={{
            position: 'absolute',
            inset: 0,
            pointerEvents: 'none',
            background: 'transparent',
          }}
        />
      </div>

      <div className="row wrap">
        {downloadUrl ? (
          <a className="secondary" href={downloadUrl} download={downloadName}>
            Download clip
          </a>
        ) : null}

        {keypointsUrl ? (
          <a className="secondary" href={keypointsUrl} download={keypointsName}>
            Download keypoints JSON
          </a>
        ) : null}
      </div>

      <div className="hint">
        Record the full dance, then stop and download the clip plus the keypoints JSON. Later you
        can run mocap on the saved videos.
      </div>
    </section>
  );
}