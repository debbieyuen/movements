# my-app

The Next.js capture app. See the [root README](../README.md) for the full
workflow (pose server, live demo, offline processing).

```bash
npm install   # postinstall also fetches MediaPipe wasm + model into public/
npm run dev   # http://localhost:3000
```

Structure:

- `app/` — routes (home + `/session/[sessionId]`)
- `components/` — `SessionShell` (session + server config), `CameraRecorder`
  (composition + layout), `ErrorBanner`, `wsUrl` (server URL + token resolution)
- `hooks/` — `useCamera`, `usePose`, `useRecorder`, `usePoseSocket`, `useUploader`
- `lib/protocol.ts` — wire format v2 (see [protocol/](../protocol/README.md))
- `scripts/fetch-assets.mjs` — downloads the self-hosted MediaPipe assets
