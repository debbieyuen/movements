'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import ErrorBanner from './ErrorBanner';
import { useCamera } from '../hooks/useCamera';
import { usePose, type PoseResult } from '../hooks/usePose';
import { useRecorder } from '../hooks/useRecorder';
import { usePoseSocket } from '../hooks/usePoseSocket';
import { useUploader } from '../hooks/useUploader';
import { POSE_SOURCE, toLandmark4, type Landmark4 } from '../lib/protocol';

type KeypointFrame = {
  seq: number;
  tMs: number; // ms since recording start -- SAME meaning as the wire
  unixMs: number;
  world: Landmark4[];
  norm: Landmark4[];
};

const OVERLAY_LABELS = [
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
  const drawingRef = useRef<{
    DrawingUtils: typeof import('@mediapipe/tasks-vision').DrawingUtils;
    POSE_CONNECTIONS: (typeof import('@mediapipe/tasks-vision').PoseLandmarker)['POSE_CONNECTIONS'];
  } | null>(null);

  const cam = useCamera();
  const pose = usePose(videoRef);
  const socket = usePoseSocket(sessionId, role);
  const uploader = useUploader();

  const recordingRef = useRef(false);
  const keypointsRef = useRef<KeypointFrame[]>([]);
  const mirrored = cam.facing === 'user';
  const mirroredRef = useRef(mirrored);
  useEffect(() => {
    mirroredRef.current = mirrored;
  }, [mirrored]);

  const [isPreviewOn, setIsPreviewOn] = useState(false);
  const [fallbackDownloads, setFallbackDownloads] = useState<
    { url: string; name: string }[]
  >([]);

  // What the motion IS, in words. A vision-language-action model needs the
  // language half; relabelling a batch of clips after the fact is miserable,
  // so capture it at record time.
  const [label, setLabel] = useState('');
  const [notes, setNotes] = useState('');

  const rec = useRecorder(() => {
    // t=0 for this take: reset the pose epoch the moment recording starts, so
    // tMs in the wire frames and the keypoints file both start near zero.
    pose.resetEpoch();
    keypointsRef.current = [];
    recordingRef.current = true;
  });

  // Load the drawing utilities once (browser-only module).
  useEffect(() => {
    let cancelled = false;
    void import('@mediapipe/tasks-vision').then((m) => {
      if (!cancelled) {
        drawingRef.current = {
          DrawingUtils: m.DrawingUtils,
          POSE_CONNECTIONS: m.PoseLandmarker.POSE_CONNECTIONS,
        };
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const drawOverlay = useCallback((r: PoseResult) => {
    const video = videoRef.current;
    const overlay = overlayRef.current;
    const drawing = drawingRef.current;
    if (!video || !overlay) return;

    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;
    if (overlay.width !== width) overlay.width = width;
    if (overlay.height !== height) overlay.height = height;

    const ctx = overlay.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);

    if (drawing) {
      const du = new drawing.DrawingUtils(ctx);
      du.drawConnectors(r.landmarks, drawing.POSE_CONNECTIONS, {
        color: '#00ff88',
        lineWidth: 6,
      });
      du.drawLandmarks(r.landmarks, { color: '#ffcc00', lineWidth: 2, radius: 4 });
    }

    // Joint labels. The canvas is CSS-mirrored together with the video for
    // front cameras, so text is pre-flipped here to stay readable.
    ctx.font = '16px Arial';
    ctx.lineWidth = 4;
    const flip = mirroredRef.current;
    for (const item of OVERLAY_LABELS) {
      const p = r.landmarks[item.i];
      if (!p) continue;
      if (typeof p.visibility === 'number' && p.visibility < 0.5) continue;
      const x = p.x * width;
      const y = p.y * height;
      ctx.strokeStyle = '#000000';
      ctx.fillStyle = '#ffffff';
      if (flip) {
        ctx.save();
        ctx.scale(-1, 1);
        ctx.strokeText(item.name, -(x - 6), y - 6);
        ctx.fillText(item.name, -(x - 6), y - 6);
        ctx.restore();
      } else {
        ctx.strokeText(item.name, x + 6, y - 6);
        ctx.fillText(item.name, x + 6, y - 6);
      }
    }
  }, []);

  // One subscription handles overlay + local keypoint log + live streaming.
  // Subscribe ONCE: `pose` and `socket` are fresh objects each render, so
  // depending on them would tear down and re-register the callback (and the
  // pose loop's consumer) on every state update. Read them through a ref.
  const frameDepsRef = useRef({ drawOverlay, sendFrame: socket.sendFrame });
  useEffect(() => {
    frameDepsRef.current = { drawOverlay, sendFrame: socket.sendFrame };
  }, [drawOverlay, socket.sendFrame]);

  const { onFrame } = pose;
  useEffect(() => {
    onFrame((r) => {
      const { drawOverlay: draw, sendFrame } = frameDepsRef.current;
      draw(r);
      if (recordingRef.current) {
        keypointsRef.current.push({
          seq: keypointsRef.current.length,
          tMs: r.tCaptureMs,
          unixMs: Date.now(),
          world: toLandmark4(r.worldLandmarks),
          norm: toLandmark4(r.landmarks),
        });
      }
      sendFrame(r);
    });
    return () => onFrame(null);
  }, [onFrame]);

  const startPreview = useCallback(async () => {
    const stream = await cam.start();
    if (!stream) return null; // cam.error is set and shown in the banner
    const video = videoRef.current;
    if (video) {
      video.srcObject = stream;
      try {
        await video.play();
      } catch (err) {
        // A rapid stop/start can interrupt play() with a benign AbortError.
        if ((err as { name?: string })?.name !== 'AbortError') throw err;
      }
    }
    try {
      await pose.start();
    } catch {
      return null; // pose.error is shown in the banner
    }
    setIsPreviewOn(true);
    return stream;
  }, [cam, pose]);

  const stopPreview = useCallback(() => {
    pose.stop();
    cam.stop();
    if (videoRef.current) videoRef.current.srcObject = null;
    const ctx = overlayRef.current?.getContext('2d');
    ctx?.clearRect(0, 0, overlayRef.current!.width, overlayRef.current!.height);
    setIsPreviewOn(false);
  }, [cam, pose]);

  const startRecording = useCallback(async () => {
    if (recordingRef.current) return;
    let stream = cam.stream;
    if (!stream) stream = await startPreview();
    if (!stream) return;
    setFallbackDownloads([]);
    rec.start(stream);
  }, [cam.stream, rec, startPreview]);

  const stopRecording = useCallback(async () => {
    recordingRef.current = false;
    const result = await rec.stop();
    if (!result) return;

    const ext = result.mimeType.includes('mp4') ? 'mp4' : 'webm';
    const videoName = makeFilename(sessionId, role, ext);
    const stem = videoName.slice(0, -(ext.length + 1));
    const keypointsName = makeFilename(sessionId, `${role}_keypoints`, 'json');
    const keypointsBlob = new Blob([JSON.stringify(keypointsRef.current)], {
      type: 'application/json',
    });

    // Sidecar the offline pipeline reads: `<stem>.annotation.json` sits next to
    // the video, and `mocap.process` folds it into the clip's meta.json.
    const annotation = {
      schemaVersion: 1,
      sessionId,
      role,
      video: videoName,
      label: label.trim(),
      notes: notes.trim(),
      recordedAtUnixMs: Date.now(),
      durationMs: Math.round(result.durationMs),
      camera: cam.actual,
      audio: cam.hasAudio,
      poseSource: POSE_SOURCE,
    };
    const annotationName = `${stem}.annotation.json`;
    const annotationBlob = new Blob([JSON.stringify(annotation, null, 2)], {
      type: 'application/json',
    });

    // Primary path: upload to the server (Cloudflare-chunk-safe).
    await uploader.upload({ blob: result.blob, filename: videoName, sessionId, role });
    await uploader.upload({
      blob: keypointsBlob,
      filename: keypointsName,
      sessionId,
      role,
    });
    await uploader.upload({
      blob: annotationBlob,
      filename: annotationName,
      sessionId,
      role,
    });

    // Fallback: offer local downloads (always, so nothing is ever lost).
    setFallbackDownloads([
      { url: URL.createObjectURL(result.blob), name: videoName },
      { url: URL.createObjectURL(keypointsBlob), name: keypointsName },
      { url: URL.createObjectURL(annotationBlob), name: annotationName },
    ]);
  }, [cam.actual, cam.hasAudio, label, notes, rec, role, sessionId, uploader]);

  // Host countdown fires: recordSignal increments -> auto-start recording.
  useEffect(() => {
    if (recordSignal && recordSignal > 0 && !recordingRef.current) {
      void startRecording();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recordSignal]);

  // Release the camera on UNMOUNT ONLY. stopPreview's identity changes across
  // renders (it closes over the hook objects), so depending on it directly
  // would run the cleanup — killing the live camera tracks — after every
  // re-render, and MediaRecorder.start() would then throw NotSupportedError
  // on the dead stream. Route the latest callback through a ref instead.
  const stopPreviewRef = useRef(stopPreview);
  useEffect(() => {
    stopPreviewRef.current = stopPreview;
  }, [stopPreview]);
  useEffect(() => () => stopPreviewRef.current(), []);

  return (
    <section className="card">
      <h2>Camera recorder</h2>
      <p>
        Pick a camera, preview, and record. Recordings upload to the session server
        automatically; use “Live to robot” to stream the skeleton in real time.
      </p>

      <ErrorBanner message={cam.error?.message ?? null} />
      <ErrorBanner message={pose.error} />
      <ErrorBanner message={rec.error} />
      <ErrorBanner
        message={uploader.state === 'error' ? `Upload failed: ${uploader.error}` : null}
      />

      <div className="row wrap" style={{ gap: 8, alignItems: 'center' }}>
        <label style={{ flex: '1 1 260px' }}>
          What is this motion?
          <input
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="e.g. squatting, waving hello, reaching left"
            style={{ width: '100%', padding: '4px 8px' }}
          />
        </label>
        <label style={{ flex: '1 1 200px' }}>
          Notes (optional)
          <input
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="e.g. stumbled at the end"
            style={{ width: '100%', padding: '4px 8px' }}
          />
        </label>
      </div>
      {!label.trim() && (
        <div className="hint">
          Label the motion before recording — the offline pipeline copies it into
          the clip metadata, and unlabelled clips are far less useful for training.
        </div>
      )}

      <div className="row wrap">
        <label>
          Camera
          <select
            value={cam.deviceId ?? ''}
            onChange={(e) => cam.setDeviceId(e.target.value)}
            disabled={rec.isRecording}
          >
            {cam.devices.length === 0 && <option value="">Default camera</option>}
            {cam.devices.map((d, i) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label || `Camera ${i + 1}`}
              </option>
            ))}
          </select>
        </label>

        <button
          className="secondary"
          onClick={() => void startPreview()}
          disabled={rec.isRecording}
        >
          Start preview
        </button>

        <button
          className="primary"
          onClick={() => void startRecording()}
          disabled={rec.isRecording}
        >
          Record
        </button>

        <button
          className="danger"
          onClick={() => void stopRecording()}
          disabled={!rec.isRecording}
        >
          Stop
        </button>

        <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <input
            type="checkbox"
            checked={socket.live}
            onChange={(e) => socket.setLive(e.target.checked)}
          />
          Live to robot
        </label>

        <label
          style={{ display: 'flex', alignItems: 'center', gap: 6 }}
          title="Narrate what you're doing while you move — speech paired with motion is training data"
        >
          <input
            type="checkbox"
            checked={cam.audio}
            onChange={(e) => cam.setAudio(e.target.checked)}
            disabled={rec.isRecording}
          />
          Record narration audio
        </label>
      </div>

      <div className="metaRow">
        <span>
          Status:{' '}
          {rec.isRecording ? 'Recording' : isPreviewOn ? 'Preview ready' : 'Idle'}
        </span>
        <span>Elapsed: {rec.seconds}s</span>
        <span>
          Pose:{' '}
          {pose.status !== 'ready'
            ? pose.status
            : `${pose.stats.fps} fps ${
                pose.stats.detecting ? '· tracking' : '· no person in frame'
              }`}
        </span>
        <span>
          {cam.actual
            ? `${cam.actual.width}×${cam.actual.height}@${Math.round(
                cam.actual.frameRate,
              )}`
            : 'camera off'}
        </span>
        <span>
          Link: {socket.live ? `${socket.status} (${socket.framesSent} sent)` : 'off'}
        </span>
        <span>Mic: {cam.stream ? (cam.hasAudio ? 'on' : 'off') : '—'}</span>
        {uploader.state === 'uploading' && (
          <span>Uploading: {Math.round(uploader.progress * 100)}%</span>
        )}
        {uploader.state === 'done' && <span>Uploaded ✓</span>}
      </div>

      <div
        style={{
          position: 'relative',
          width: '100%',
          transform: mirrored ? 'scaleX(-1)' : undefined,
        }}
      >
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
      {mirrored && (
        <div className="hint">Preview is mirrored; recordings are not.</div>
      )}

      {fallbackDownloads.length > 0 && (
        <div className="row wrap">
          {uploader.state === 'error' && (
            <button className="secondary" onClick={uploader.retry}>
              Retry upload
            </button>
          )}
          {fallbackDownloads.map((f) => (
            <a key={f.name} className="secondary" href={f.url} download={f.name}>
              Download {f.name.includes('keypoints') ? 'keypoints' : 'clip'}
            </a>
          ))}
        </div>
      )}

      <div className="hint">
        Record the full move, then Stop. The clip and keypoints upload to the server
        for offline processing (GVHMR → H1); downloads above are a local backup.
      </div>
    </section>
  );
}
