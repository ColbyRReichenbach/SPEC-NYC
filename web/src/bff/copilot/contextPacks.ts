import { buildCanonicalGovernanceStatus } from "@/src/bff/clients/canonicalGovernanceClient";
import { buildCanonicalMonitoringOverview } from "@/src/bff/clients/canonicalMonitoringClient";
import { readJsonArtifact, readTextArtifact } from "@/src/bff/clients/artifactStore";
import type { RoutingDecision } from "@/src/bff/copilot/intentRouter";
import type { CopilotPage } from "@/src/features/copilot/schemas/copilotSchemas";

type EvidenceRef = {
  key: string;
  path: string;
  available: boolean;
};

export type CopilotContextPack = {
  page: CopilotPage;
  summary: string[];
  evidenceRefs: EvidenceRef[];
  missingArtifactKeys: string[];
  contextBundleIds: string[];
};

function toEvidenceRef(key: string, path: string, available: boolean): EvidenceRef {
  return { key, path, available };
}

async function buildValuationPack(): Promise<CopilotContextPack> {
  const metricsPath = "models/metrics_v1.json";
  const runCardPath = "reports/arena/run_card_34e917e198af4e58adb2097b8d9ca229.md";
  const perfPath = "reports/monitoring/performance_latest.json";

  const [metrics, runCardText, performance] = await Promise.all([
    readJsonArtifact<{ overall?: { mdape?: number; ppe10?: number } }>(metricsPath),
    readTextArtifact(runCardPath),
    readJsonArtifact<{ status?: string; metrics?: { overall?: { ppe10?: number; mdape?: number } } }>(perfPath)
  ]);

  const refs: EvidenceRef[] = [
    toEvidenceRef("metrics", metricsPath, Boolean(metrics)),
    toEvidenceRef("run_card", runCardPath, Boolean(runCardText)),
    toEvidenceRef("monitoring_performance", perfPath, Boolean(performance))
  ];
  const missing = refs.filter((ref) => !ref.available).map((ref) => ref.key);

  return {
    page: "valuation",
    summary: [
      `Overall metrics: PPE10 ${metrics?.overall?.ppe10 ?? "n/a"}, MdAPE ${metrics?.overall?.mdape ?? "n/a"}.`,
      `Monitoring status: ${performance?.status ?? "unknown"}.`
    ],
    evidenceRefs: refs,
    missingArtifactKeys: missing,
    contextBundleIds: refs.map((ref) => ref.path)
  };
}

async function buildGovernancePack(): Promise<CopilotContextPack> {
  const governance = await buildCanonicalGovernanceStatus();
  const refs: EvidenceRef[] = [
    toEvidenceRef("governance_status", "reports/arena/proposal_*.json", governance.payload.latest_proposal.proposal_id !== "none")
  ];
  const missing = refs.filter((ref) => !ref.available).map((ref) => ref.key);

  return {
    page: "governance",
    summary: [
      `Proposal status: ${governance.payload.latest_proposal.status}.`,
      `Gate checks: ${governance.payload.gate_results.filter((gate) => gate.status === "pass").length}/${governance.payload.gate_results.length} pass.`
    ],
    evidenceRefs: refs,
    missingArtifactKeys: missing,
    contextBundleIds: [governance.sourceContext.source_id]
  };
}

async function buildMonitoringPack(window: "7d" | "30d" | "90d" | "180d"): Promise<CopilotContextPack> {
  const monitoring = await buildCanonicalMonitoringOverview(window);
  const refs: EvidenceRef[] = [
    toEvidenceRef("drift", "reports/monitoring/drift_latest.json", monitoring.payload.drift_summary.rows > 0),
    toEvidenceRef("performance", "reports/monitoring/performance_latest.json", monitoring.payload.performance_summary.overall.n > 0),
    toEvidenceRef(
      "retrain_decision",
      "reports/releases/retrain_decision_latest.json",
      monitoring.payload.retrain_decision.decision.length > 0
    )
  ];
  const missing = refs.filter((ref) => !ref.available).map((ref) => ref.key);

  return {
    page: "monitoring",
    summary: [
      `Monitoring window: ${window}.`,
      `Drift alerts: ${monitoring.payload.drift_summary.alerts}, warnings: ${monitoring.payload.drift_summary.warnings}.`,
      `Retrain decision: ${monitoring.payload.retrain_decision.decision}.`
    ],
    evidenceRefs: refs,
    missingArtifactKeys: missing,
    contextBundleIds: [monitoring.sourceContext.source_id]
  };
}

export async function buildCopilotContextPack(
  decision: RoutingDecision,
  page: CopilotPage
): Promise<CopilotContextPack> {
  if (page === "governance") return buildGovernancePack();
  if (page === "monitoring") return buildMonitoringPack(decision.entities.window ?? "30d");
  if (page === "valuation") return buildValuationPack();

  const [valuation, governance, monitoring] = await Promise.all([
    buildValuationPack(),
    buildGovernancePack(),
    buildMonitoringPack(decision.entities.window ?? "30d")
  ]);

  const combinedRefs = [...valuation.evidenceRefs, ...governance.evidenceRefs, ...monitoring.evidenceRefs];
  return {
    page: "global",
    summary: [...valuation.summary, ...governance.summary, ...monitoring.summary],
    evidenceRefs: combinedRefs,
    missingArtifactKeys: combinedRefs.filter((ref) => !ref.available).map((ref) => ref.key),
    contextBundleIds: combinedRefs.map((ref) => ref.path)
  };
}
