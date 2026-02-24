import { readJsonArtifact } from "@/src/bff/clients/artifactStore";

type MetricsPayload = {
  metadata?: {
    model_version?: string;
  };
};

export default async function ContextStrip() {
  const metrics = await readJsonArtifact<MetricsPayload>("models/metrics_v1.json");
  const modelVersion = metrics?.metadata?.model_version ?? "v1";

  return (
    <div className="context-strip" role="status" aria-live="polite">
      <div>
        <span className="context-label">Environment</span>
        <span className="context-value">Local Demo</span>
      </div>
      <div>
        <span className="context-label">Model Alias</span>
        <span className="context-value">champion</span>
      </div>
      <div>
        <span className="context-label">Model Version</span>
        <span className="context-value">{modelVersion}</span>
      </div>
      <div>
        <span className="context-label">Contract</span>
        <span className="context-value">v1 canonical</span>
      </div>
    </div>
  );
}
