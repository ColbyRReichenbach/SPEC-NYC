import { describe, expect, it } from "vitest";
import {
  propertyExplanationRequestSchema,
  singleValuationRequestSchema,
  singleValuationResponseSchema
} from "@/src/features/valuation/schemas/valuationSchemas";
import { copilotAskRequestSchema, copilotAskResponseSchema } from "@/src/features/copilot/schemas/copilotSchemas";

describe("valuation contracts", () => {
  it("accepts valid single valuation request", () => {
    const parsed = singleValuationRequestSchema.parse({
      property: {
        address: "123 Example St, Brooklyn, NY",
        borough: "BROOKLYN",
        gross_square_feet: 1800,
        year_built: 1930,
        residential_units: 2,
        total_units: 2,
        building_class: "B1",
        property_segment: "SMALL_MULTI",
        sale_date: "2026-02-23"
      },
      context: {
        dataset_version: "ds_hseason001_train_20260223",
        model_alias: "champion"
      }
    });

    expect(parsed.context.model_alias).toBe("champion");
  });

  it("rejects invalid request payload", () => {
    const invalid = singleValuationRequestSchema.safeParse({
      property: {
        address: "x"
      },
      context: {}
    });

    expect(invalid.success).toBe(false);
  });

  it("accepts valid single valuation response", () => {
    const parsed = singleValuationResponseSchema.parse({
      valuation_id: "val_abc123",
      predicted_price: 1285000,
      prediction_interval: { low: 1170000, high: 1399000, method: "quantile_residual_v1" },
      confidence: {
        score: 0.68,
        band: "medium",
        factors: {
          segment_calibration: 0.71,
          support_coverage: 0.62,
          input_completeness: 0.74
        },
        caveats: ["Estimate is probabilistic, not an appraisal."]
      },
      explanation: {
        status: "ready",
        explainer_type: "xgboost_pred_contribs",
        drivers_positive: [{ feature: "gross_square_feet", impact: 142000, display: "Larger interior size" }],
        drivers_negative: [{ feature: "building_age", impact: -47000, display: "Older building profile" }]
      },
      model: {
        alias: "champion",
        run_id: "34e917e198af4e58adb2097b8d9ca229",
        model_version: "1",
        route: "SMALL_MULTI"
      },
      evidence: {
        run_card_path: "reports/arena/run_card_34e917e198af4e58adb2097b8d9ca229.md",
        metrics_path: "models/metrics_v1.json",
        shap_summary_path: "reports/model/shap_summary_v1.png"
      }
    });

    expect(parsed.explanation.drivers_positive.length).toBeGreaterThan(0);
  });

  it("accepts explanation request", () => {
    const parsed = propertyExplanationRequestSchema.parse({ valuation_id: "val_abc123", top_n: 5 });
    expect(parsed.top_n).toBe(5);
  });
});

describe("copilot contracts", () => {
  it("accepts valid copilot ask request", () => {
    const parsed = copilotAskRequestSchema.parse({
      question: "Why this estimate?",
      intent: "why_estimate",
      context: { valuation_id: "val_abc123" }
    });
    expect(parsed.intent).toBe("why_estimate");
  });

  it("accepts valid copilot response", () => {
    const parsed = copilotAskResponseSchema.parse({
      answer: "The estimate is most influenced by interior size and local baseline pricing.",
      citations: ["models/metrics_v1.json"],
      assistant_confidence: 0.83,
      limitations: ["No fresh external listing feed included."],
      safety: {
        guardrail_mode: "strict_grounded",
        fallback_used: false
      }
    });

    expect(parsed.safety.fallback_used).toBe(false);
  });
});
