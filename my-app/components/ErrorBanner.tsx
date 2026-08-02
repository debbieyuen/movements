'use client';

export default function ErrorBanner({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div
      role="alert"
      style={{
        background: '#fee2e2',
        color: '#991b1b',
        border: '1px solid #fca5a5',
        borderRadius: 8,
        padding: '8px 12px',
        margin: '8px 0',
        fontSize: 14,
      }}
    >
      ⚠️ {message}
    </div>
  );
}
