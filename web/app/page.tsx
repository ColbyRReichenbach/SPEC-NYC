import Link from "next/link";

export default function HomePage() {
  return (
    <main className="landing-shell">
      <section className="card fade-in-up">
        <h1>Azuli AVM Dashboard</h1>
        <p className="muted">Canonical, datasource-agnostic valuation workflows for production decisioning.</p>
        <div className="button-row">
          <Link href="/valuation/single" className="primary-btn">
            Open Dashboard
          </Link>
        </div>
      </section>
    </main>
  );
}
