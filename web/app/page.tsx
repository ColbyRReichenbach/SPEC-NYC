import Link from "next/link";

export default function HomePage() {
  return (
    <main>
      <div className="card">
        <h1>SPEC NYC AVM Dashboard</h1>
        <p>Use the production dashboard routes below.</p>
        <ul>
          <li><Link href="/valuation/single">Single valuation</Link></li>
          <li><Link href="/valuation/batch">Batch valuation</Link></li>
          <li><Link href="/governance">Model governance</Link></li>
          <li><Link href="/monitoring">Monitoring and drift</Link></li>
          <li><Link href="/copilot">AI copilot</Link></li>
        </ul>
      </div>
    </main>
  );
}
