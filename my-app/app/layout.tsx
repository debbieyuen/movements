import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Motion Capture Studio',
  description: 'Multi-device capture for dance and humanoid motion research',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}