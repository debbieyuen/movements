// Resolves the pose-server WebSocket URL for both the recording and presence
// sockets, in priority order:
//   1. NEXT_PUBLIC_WS_URL  -> set this to your tunnel, e.g. wss://xxxx.ngrok.io
//      (needed when the page is served over HTTPS from Vercel/a tunnel, since an
//      HTTPS page can only open a secure wss:// socket).
//   2. Otherwise derive from the page: wss:// when the page is HTTPS, else ws://,
//      on the same host at port 8765 (works for local dev + same-LAN http).
export function getWsUrl(): string {
  const configured = process.env.NEXT_PUBLIC_WS_URL;
  if (configured) return configured;

  if (typeof window !== 'undefined') {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${proto}://${window.location.hostname}:8765`;
  }
  return 'ws://127.0.0.1:8765';
}
