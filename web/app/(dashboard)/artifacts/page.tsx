export default function ArtifactsPage() {
  return (
    <section className="stack-lg fade-in-up">
      <div className="card">
        <h1>Artifacts</h1>
        <p className="muted">Canonical evidence links exposed to the frontend remain provider-agnostic.</p>
      </div>
      <div className="card">
        <h2>Core Paths</h2>
        <ul>
          <li><code>data/processed/hseason001_train_*.csv</code></li>
          <li><code>models/metrics_v1.json</code></li>
          <li><code>reports/arena/proposal_*.json</code></li>
          <li><code>reports/monitoring/performance_latest.json</code></li>
          <li><code>reports/monitoring/drift_latest.json</code></li>
          <li><code>reports/releases/retrain_decision_latest.json</code></li>
        </ul>
      </div>
    </section>
  );
}
