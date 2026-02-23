import { errorJson, okJson } from "@/src/lib/http";
import {
  singleValuationRequestSchema,
  singleValuationResponseSchema
} from "@/src/features/valuation/schemas/valuationSchemas";

export async function POST(req: Request) {
  const payload = await req.json().catch(() => null);
  const parsed = singleValuationRequestSchema.safeParse(payload);
  if (!parsed.success) {
    return errorJson(`Invalid payload: ${parsed.error.issues[0]?.message ?? "unknown error"}`);
  }

  const alias = parsed.data.context.model_alias;
  const response = singleValuationResponseSchema.parse({
    valuation_id: `val_${crypto.randomUUID().slice(0, 8)}`,
    predicted_price: 1285000,
    prediction_interval: {
      low: 1170000,
      high: 1399000,
      method: "quantile_residual_v1"
    },
    confidence: {
      score: 0.68,
      band: "medium",
      factors: {
        segment_calibration: 0.71,
        support_coverage: 0.62,
        input_completeness: 0.74
      },
      caveats: [
        "Estimate is probabilistic, not an appraisal.",
        "Confidence depends on data quality and comparable coverage."
      ]
    },
    explanation: {
      status: "ready",
      explainer_type: "xgboost_pred_contribs",
      drivers_positive: [
        {
          feature: "gross_square_feet",
          impact: 142000,
          display: "Larger interior size"
        },
        {
          feature: "h3_price_lag",
          impact: 89000,
          display: "Higher local market baseline"
        }
      ],
      drivers_negative: [
        {
          feature: "building_age",
          impact: -47000,
          display: "Older building profile"
        },
        {
          feature: "distance_to_center_km",
          impact: -26000,
          display: "Farther from central demand zones"
        }
      ]
    },
    model: {
      alias,
      run_id: "34e917e198af4e58adb2097b8d9ca229",
      model_version: "1",
      route: parsed.data.property.property_segment
    },
    evidence: {
      run_card_path: "reports/arena/run_card_34e917e198af4e58adb2097b8d9ca229.md",
      metrics_path: "models/metrics_v1.json",
      shap_summary_path: "reports/model/shap_summary_v1.png"
    }
  });

  return okJson(response);
}
