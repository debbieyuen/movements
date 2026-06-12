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
  t: number;
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
}: {
  sessionId: string;
  role: string;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);

  const poseRef = useRef<any>(null);
  const drawingRef = useRef<any>(null);
  const poseLoopRef = useRef<number | null>(null);
  const poseBusyRef = useRef(false);
  const poseRunningRef = useRef(false);
  const recordingRef = useRef(false);
  const poseFramesRef = useRef<PoseFrame[]>([]);

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

  const ensurePose = async () => {
    if (poseRef.current) return poseRef.current;

    const [{ Pose }, drawingUtils] = await Promise.all([
      import('@mediapipe/pose'),
      import('@mediapipe/drawing_utils'),
    ]);

    const pose = new Pose({
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

      const drawing = drawingRef.current;
      if (results.poseLandmarks && drawing) {
        drawing.drawConnectors(ctx, results.poseLandmarks, drawing.POSE_CONNECTIONS, {
          color: '#00D084',
          lineWidth: 4,
        });
        drawing.drawLandmarks(ctx, results.poseLandmarks, {
          color: '#FF5A5F',
          lineWidth: 2,
        });
      }

      if (recordingRef.current && results.poseLandmarks) {
        const frame: PoseFrame = {
          t: performance.now(),
          landmarks: serializeLandmarks(results.poseLandmarks),
          worldLandmarks: serializeLandmarks(results.poseWorldLandmarks),
        };

        poseFramesRef.current.push(frame);
        setKeypointCount(poseFramesRef.current.length);
      }
    });

    poseRef.current = pose;
    drawingRef.current = drawingUtils as any;
    setPoseReady(true);

    return pose;
  };

  const startPoseLoop = async () => {
    if (!poseRunningRef.current) return;
    if (!poseRef.current || !videoRef.current) return;

    if (!poseBusyRef.current && videoRef.current.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
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
      await videoRef.current.play();
    }

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

    chunksRef.current = [];
    poseFramesRef.current = [];
    setKeypointCount(0);
    setSeconds(0);
    setDownloadUrl('');
    setDownloadName('');
    setKeypointsUrl('');
    setKeypointsName('');

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