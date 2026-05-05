import type { SourceContext } from "@/src/bff/types/baseContracts";
import { readJsonArtifact, resolveRepoRoot } from "@/src/bff/clients/artifactStore";
import { resolveDashboardPackageSelection } from "@/src/features/platform/packageResolver";

type PerformancePayload = {
  status?: string;
  metrics?: {
    overall?: { n?: number; ppe10?: number; mdape?: number; r2?: number };
    per_segment?: Record<string, { n?: number; ppe10?: number; mdape?: number; r2?: number }>;
  };
};

type DriftPayload = {
  status?: string;
  alerts?: number;
  warnings?: number;
  rows?: number;
  reference_csv?: string;
  current_csv?: string;
};

type RetrainPayload = {
  should_retrain?: boolean;
  decision?: string;
  reasons?: string[];
  policy?: Record<string, number>;
  signals?: Record<string, string | number | boolean>;
};

export async function buildCanonicalMonitoringOverview(window: string): Promise<{
  payload: {
    window: string;
    drift_summary: {
      status: string;
      alerts: number;
      warnings: number;
      rows: number;
      reference_csv?: string;
      current_csv?: string;
    };
    performance_summary: {
      status: string;
      overall: {
        n: number;
        ppe10: number;
        mdape: number;
        r2: number;
      };
    };
    slice_metrics: Array<{
      slice_key: string;
      n: number;
      ppe10: number;
      mdape: number;
      r2: number;
    }>;
    retrain_decision: {
      should_retrain: boolean;
      decision: string;
      reasons: string[];
      policy: Record<string, number>;
      signals: Record<string, string | number | boolean>;
    };
    degraded: boolean;
    warnings: string[];
  };
  sourceContext: SourceContext;
}> {
  const repoRoot = await resolveRepoRoot();
  const selection = await resolveDashboardPackageSelection(repoRoot);
  const packagePath = selection.packagePath;
  const packageMetricsPath = packagePath ? `${packagePath}/metrics.json` : "missing";
  const packageDriftPath = packagePath ? `${packagePath}/drift_report.json` : "missing";
  const perfPath = "reports/monitoring/performance_latest.json";
  const driftPath = "reports/monitoring/drift_latest.json";
  const retrainPath = "reports/releases/retrain_decision_latest.json";

  const [performance, drift, retrain, packageMetrics, packageDrift] = await Promise.all([
    readJsonArtifact<PerformancePayload>(perfPath),
    readJsonArtifact<DriftPayload>(driftPath),
    readJsonArtifact<RetrainPayload>(retrainPath),
    packagePath ? readJsonArtifact<PerformancePayload>(packageMetricsPath) : Promise.resolve(null),
    packagePath ? readJsonArtifact<DriftPayload>(packageDriftPath) : Promise.resolve(null)
  ]);

  const warnings: string[] = [];
  if (!performance && !packageMetrics) warnings.push("Performance artifact missing; using empty metrics.");
  if (!drift && !packageDrift) warnings.push("Drift artifact missing; using empty drift summary.");
  if (!retrain) warnings.push("Retrain artifact missing; using fallback retrain decision.");

  const metricsPayload = performance ?? packageMetrics;
  const driftPayload = drift ?? packageDrift;
  const overall = metricsPayload?.metrics?.overall ?? (packageMetrics as any)?.overall ?? {};
  const perSegment = metricsPayload?.metrics?.per_segment ?? (packageMetrics as any)?.per_segment ?? {};

  const sliceMetrics = Object.entries(perSegment).map(([segment, rawVals]) => {
    const vals = rawVals as { n?: number; ppe10?: number; mdape?: number; r2?: number };
    return {
      slice_key: segment,
      n: Number(vals.n ?? 0),
      ppe10: Number(vals.ppe10 ?? 0),
      mdape: Number(vals.mdape ?? 0),
      r2: Number(vals.r2 ?? 0)
    };
  });

  return {
    payload: {
      window,
      drift_summary: {
        status: driftPayload?.status ?? "warn",
        alerts: Number(driftPayload?.alerts ?? (packageMetrics as any)?.metadata?.feature_drift_alerts ?? 0),
        warnings: Number(driftPayload?.warnings ?? (packageMetrics as any)?.metadata?.feature_drift_warnings ?? 0),
        rows: Number(driftPayload?.rows ?? (packageMetrics as any)?.metadata?.feature_drift_rows ?? 0),
        reference_csv: driftPayload?.reference_csv,
        current_csv: driftPayload?.current_csv
      },
      performance_summary: {
        status: metricsPayload?.status ?? "package_metrics",
        overall: {
          n: Number(overall.n ?? 0),
          ppe10: Number(overall.ppe10 ?? 0),
          mdape: Number(overall.mdape ?? 0),
          r2: Number(overall.r2 ?? 0)
        }
      },
      slice_metrics: sliceMetrics,
      retrain_decision: {
        should_retrain: Boolean(retrain?.should_retrain),
        decision: retrain?.decision ?? "hold",
        reasons: retrain?.reasons ?? ["Fallback decision due to missing artifact evidence."],
        policy: retrain?.policy ?? {},
        signals: retrain?.signals ?? {}
      },
      degraded: warnings.length > 0,
      warnings
    },
    sourceContext: {
      source_id: [packageMetricsPath, packageDriftPath, perfPath, driftPath, retrainPath].join("|"),
      source_type: "other"
    }
  };
}
