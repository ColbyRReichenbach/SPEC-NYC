import { loadPlatformData } from "@/src/features/platform/data";

export default async function CopilotPage() {
  const data = await loadPlatformData();
  return (
    <div className="view-stack">
      <section className="section-header">
        <span className="eyebrow">Copilot Workspace</span>
        <h1>Grounded artifact assistant state</h1>
        <p>
          Copilot answers are constrained to platform evidence: current package artifacts, governance status,
          monitoring reports, and citations from the backend context pack.
        </p>
      </section>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Context Pack</span>
            <h2>{data.package.id}</h2>
          </div>
        </div>
        <dl className="evidence-list">
          <div><dt>Model metrics</dt><dd><code>{data.package.path}/metrics.json</code></dd></div>
          <div><dt>Governance report</dt><dd><code>{data.release.reportPath}</code></dd></div>
          <div><dt>EDA report</dt><dd><code>{data.eda.reportPath}</code></dd></div>
          <div><dt>Artifact links</dt><dd>{data.eda.artifactLinks.length.toLocaleString()} indexed artifacts</dd></div>
        </dl>
      </section>
    </div>
  );
}
