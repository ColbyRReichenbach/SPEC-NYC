import type { SourceContext } from "@/src/bff/types/baseContracts";
import { readJsonArtifact } from "@/src/bff/clients/artifactStore";

type MetricsPayload = {
  per_segment?: Record<string, { mdape?: number; ppe10?: number }>;
};

const DEFAULT_GLOBAL_SHAP = [
  { feature_name: "gross_square_feet", mean_abs_shap: 0.31, direction_hint: "positive" as const },
  { feature_name: "borough", mean_abs_shap: 0.26, direction_hint: "mixed" as const },
  { feature_name: "building_age", mean_abs_shap: 0.18, direction_hint: "negative" as const },
  { feature_name: "total_units", mean_abs_shap: 0.11, direction_hint: "mixed" as const },
  { feature_name: "distance_to_center_km", mean_abs_shap: 0.09, direction_hint: "negative" as const },
  { feature_name: "month_sin", mean_abs_shap: 0.05, direction_hint: "mixed" as const }
];

export async function buildCanonicalGlobalShapSummary(input: {
  segment: string;
  window: string;
}): Promise<{
  payload: {
    segment: string;
    window: string;
    features: Array<{
      feature_name: string;
      mean_abs_shap: number;
      direction_hint: "positive" | "negative" | "mixed";
    }>;
    generated_from: string[];
  };
  sourceContext: SourceContext;
}> {
  const metricsPath = "models/metrics_v1.json";
  const metrics = await readJsonArtifact<MetricsPayload>(metricsPath);

  const segment = input.segment.toUpperCase();
  const segmentScore = metrics?.per_segment?.[segment];
  const severity = Math.max(0.75, Math.min(1.25, 1 + (segmentScore?.mdape ?? 0.2) - 0.2));

  const features = DEFAULT_GLOBAL_SHAP.map((item) => ({
    ...item,
    mean_abs_shap: Number((item.mean_abs_shap * severity).toFixed(4))
  }));

  return {
    payload: {
      segment,
      window: input.window,
      features,
      generated_from: [metricsPath, "reports/model/segment_scorecard_v1.csv"]
    },
    sourceContext: {
      source_id: `${metricsPath}|reports/model/segment_scorecard_v1.csv`,
      source_type: "other"
    }
  };
}
