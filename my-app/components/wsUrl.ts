// Resolves the pose-server WebSocket URL, in priority order:
//   1. ?ws=... in the page URL  -> lets the shared session link carry the
//      current tunnel URL to every device (also saved to this browser).
//   2. localStorage 'wsUrl'     -> what you last set on this device.
//   3. NEXT_PUBLIC_WS_URL       -> build-time default (optional).
//   4. derived from the page    -> wss:// on HTTPS else ws://, same host :8765.
//
// This means a rotating quick-tunnel URL never requires a Vercel rebuild --
// you paste it into the app (or it rides in on the session link) at runtime.

export function normalizeWsUrl(u: string): string {
  const s = u.trim();
  if (!s) return s;
  if (s.startsWith("https://")) return "wss://" + s.slice("https://".length);
  if (s.startsWith("http://")) return "ws://" + s.slice("http://".length);
  if (s.startsWith("wss://") || s.startsWith("ws://")) return s;
  return "wss://" + s; // bare host -> assume secure
}

export function getWsUrl(): string {
  if (typeof window !== "undefined") {
    try {
      const q = new URLSearchParams(window.location.search).get("ws");
      if (q) {
        const n = normalizeWsUrl(q);
        localStorage.setItem("wsUrl", n);
        return n;
      }
      const saved = localStorage.getItem("wsUrl");
      if (saved) return saved;
    } catch {
      // localStorage blocked -> fall through
    }
  }
  if (process.env.NEXT_PUBLIC_WS_URL) return process.env.NEXT_PUBLIC_WS_URL;
  if (typeof window !== "undefined") {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${window.location.hostname}:8765`;
  }
  return "ws://127.0.0.1:8765";
}
