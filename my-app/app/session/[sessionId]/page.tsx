import SessionShell from '../../../components/SessionShell';

export default async function SessionPage({
  params,
}: {
  params: Promise<{ sessionId: string }>;
}) {
  const { sessionId } = await params;
  return <SessionShell sessionId={sessionId} />;
}