// Resolves the pose-server base URL and auth token, in priority order:
//   1. ?ws=... / ?token=... in the page URL -> the shared session link carries
//      the current tunnel URL + token to every device (saved to this browser).
//   2. localStorage 'wsUrl' / 'wsToken'     -> what you last set on this device.
//   3. NEXT_PUBLIC_WS_URL                   -> build-time default (optional).
//   4. derived from the page                -> wss:// on HTTPS, same host :8765.
//
// A rotating quick-tunnel URL never requires a rebuild -- paste it into the
// app (or let it ride in on the session link) at runtime.
//
// Server v2 endpoints hang off the base URL:
//   WebSocket: <base>/ws?token=...
//   Uploads:   <http-base>/upload/...

export function normalizeWsUrl(u: string): string {
  const s = u.trim().replace(/\/+$/, '');
  if (!s) return s;
  if (s.startsWith('https://')) return 'wss://' + s.slice('https://'.length);
  if (s.startsWith('http://')) return 'ws://' + s.slice('http://'.length);
  if (s.startsWith('wss://') || s.startsWith('ws://')) return s;
  return 'wss://' + s; // bare host -> assume secure
}

export function getWsUrl(): string {
  if (typeof window !== 'undefined') {
    try {
      const params = new URLSearchParams(window.location.search);
      const q = params.get('ws');
      if (q) {
        const n = normalizeWsUrl(q);
        localStorage.setItem('wsUrl', n);
        const t = params.get('token');
        if (t) localStorage.setItem('wsToken', t);
        return n;
      }
      const saved = localStorage.getItem('wsUrl');
      if (saved) return saved;
    } catch {
      // localStorage blocked -> fall through
    }
  }
  if (process.env.NEXT_PUBLIC_WS_URL) return process.env.NEXT_PUBLIC_WS_URL;
  if (typeof window !== 'undefined') {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${proto}://${window.location.hostname}:8765`;
  }
  return 'ws://127.0.0.1:8765';
}

export function getToken(): string {
  if (typeof window === 'undefined') return '';
  try {
    const q = new URLSearchParams(window.location.search).get('token');
    if (q) {
      localStorage.setItem('wsToken', q);
      return q;
    }
    return localStorage.getItem('wsToken') || '';
  } catch {
    return '';
  }
}

export function setToken(token: string): void {
  try {
    if (token) localStorage.setItem('wsToken', token.trim());
    else localStorage.removeItem('wsToken');
  } catch {
    // ignore
  }
}

/** Full WebSocket endpoint including auth: <base>/ws?token=... */
export function getWsEndpoint(): string {
  const base = getWsUrl();
  const token = getToken();
  return `${base}/ws${token ? `?token=${encodeURIComponent(token)}` : ''}`;
}

/** HTTP(S) base for uploads, derived from the ws(s) base. */
export function getHttpBase(): string {
  const base = getWsUrl();
  if (base.startsWith('wss://')) return 'https://' + base.slice('wss://'.length);
  if (base.startsWith('ws://')) return 'http://' + base.slice('ws://'.length);
  return base;
}
