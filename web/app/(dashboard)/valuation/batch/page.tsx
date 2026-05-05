import { loadPlatformData } from "@/src/features/platform/data";

export default async function BatchValuationPage() {
  const data = await loadPlatformData();
  return (
    <div className="view-stack">
      <section className="section-header">
        <span className="eyebrow">Batch Valuation</span>
        <h1>Filesystem-backed batch scoring contract</h1>
        <p>
          Batch requests post to <code>/api/v1/valuations/batch</code>, score each row through the same model-backed
          contract as single valuation, and persist job artifacts under <code>reports/valuations/batch</code>.
        </p>
      </section>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Active Package</span>
            <h2>{data.package.id}</h2>
          </div>
          <span className={`status-pill ${data.package.status === "approved" ? "pass" : "warn"}`}>{data.package.status}</span>
        </div>
        <dl className="evidence-list">
          <div><dt>Resolver</dt><dd>{data.package.selection?.source ?? "unknown"}</dd></div>
          <div><dt>Dataset</dt><dd><code>{data.package.datasetVersion}</code></dd></div>
          <div><dt>Feature contract</dt><dd><code>{data.package.featureContractVersion}</code></dd></div>
          <div><dt>Batch artifact root</dt><dd><code>reports/valuations/batch</code></dd></div>
        </dl>
      </section>
    </div>
  );
}
