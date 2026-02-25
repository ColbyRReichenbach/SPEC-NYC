import fs from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  propertyExplanationRequestSchema,
  singleValuationRequestSchema,
  singleValuationResponseSchema
} from "@/src/features/valuation/schemas/valuationSchemas";
import { copilotAskRequestSchema, copilotAskResponseSchema } from "@/src/features/copilot/schemas/copilotSchemas";
import { governanceStatusSchema } from "@/src/features/governance/schemas/governanceSchemas";
import { monitoringOverviewSchema } from "@/src/features/monitoring/schemas/monitoringSchemas";
import {
  propertyDetailResponseSchema,
  propertyNearbyResponseSchema,
  propertySearchResponseSchema
} from "@/src/features/properties/schemas/propertySchemas";
import { globalShapSummaryResponseSchema } from "@/src/features/valuation/schemas/explainabilitySchemas";
import { normalizeMonitoringForUi, normalizeValuationForUi } from "@/src/bff/mappers/uiMappers";
import { buildCopilotFallback } from "@/src/bff/mappers/copilotMapper";

function meta(sourceType: "rdb" | "csv" | "api" | "other" = "other", sourceId = "source_alpha") {
  return {
    contract_version: "v1" as const,
    request_id: crypto.randomUUID(),
    generated_at: new Date().toISOString(),
    source_context: {
      source_id: sourceId,
      source_type: sourceType
    }
  };
}

const valuationPayload = {
  valuation_id: "val_abc123",
  predicted_price: 1285000,
  prediction_interval: { low: 1170000, high: 1399000, method: "quantile_residual_v1" },
  confidence: {
    score: 0.68,
    band: "medium" as const,
    factors: {
      segment_calibration: 0.71,
      support_coverage: 0.62,
      input_completeness: 0.74
    },
    caveats: ["Estimate is probabilistic, not an appraisal."]
  },
  explanation: {
    status: "ready" as const,
    explainer_type: "xgboost_pred_contribs",
    local_accuracy: 0.86,
    drivers_positive: [{ feature: "gross_square_feet", impact: 142000, display: "Larger interior size" }],
    drivers_negative: [{ feature: "building_age", impact: -47000, display: "Older building profile" }]
  },
  model: {
    alias: "champion" as const,
    run_id: "34e917e198af4e58adb2097b8d9ca229",
    model_version: "1",
    route: "SMALL_MULTI"
  },
  evidence: {
    run_card_path: "reports/arena/run_card_34e917e198af4e58adb2097b8d9ca229.md",
    metrics_path: "models/metrics_v1.json",
    shap_summary_path: "reports/model/shap_summary_v1.png"
  }
};

describe("canonical contract hardening", () => {
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
        dataset_version: "unknown_source_v1",
        model_alias: "champion"
      }
    });

    expect(parsed.context.model_alias).toBe("champion");
  });

  it("rejects invalid valuation request payload", () => {
    const invalid = singleValuationRequestSchema.safeParse({
      property: {
        address: "x"
      },
      context: {}
    });

    expect(invalid.success).toBe(false);
  });

  it("accepts canonical valuation response for multiple source types", () => {
    const providers: Array<"rdb" | "csv" | "api"> = ["rdb", "csv", "api"];
    for (const provider of providers) {
      const parsed = singleValuationResponseSchema.parse({
        ...valuationPayload,
        ...meta(provider, `provider_${provider}`)
      });
      expect(parsed.source_context.source_type).toBe(provider);
    }
  });

  it("enforces strict response parsing for missing explanation", () => {
    const invalid = singleValuationResponseSchema.safeParse({
      ...valuationPayload,
      ...meta(),
      explanation: undefined
    });
    expect(invalid.success).toBe(false);
  });

  it("supports degraded explanation path without crash-level schema failure", () => {
    const parsed = singleValuationResponseSchema.parse({
      ...valuationPayload,
      ...meta(),
      explanation: {
        status: "degraded",
        explainer_type: "heuristic_delta_v1",
        local_accuracy: 0.52,
        drivers_positive: [],
        drivers_negative: []
      }
    });

    expect(parsed.explanation.status).toBe("degraded");
  });

  it("accepts explanation request payload", () => {
    const parsed = propertyExplanationRequestSchema.parse({ valuation_id: "val_abc123", top_n: 5 });
    expect(parsed.top_n).toBe(5);
  });
});

describe("provider swap resilience", () => {
  it("normalizes provider A and provider B valuations to identical UI core", () => {
    const providerA = singleValuationResponseSchema.parse({
      ...valuationPayload,
      ...meta("rdb", "client_rdb_1")
    });
    const providerB = singleValuationResponseSchema.parse({
      ...valuationPayload,
      ...meta("csv", "legacy_csv_feed")
    });

    expect(normalizeValuationForUi(providerA)).toEqual(normalizeValuationForUi(providerB));
  });

  it("normalizes monitoring payload independent of source context", () => {
    const monitoringA = monitoringOverviewSchema.parse({
      ...meta("rdb", "client_rdb_1"),
      window: "30d",
      drift_summary: {
        status: "alert",
        alerts: 3,
        warnings: 4,
        rows: 7,
        reference_csv: "reports/monitoring/reference_slice_v1.csv",
        current_csv: "reports/monitoring/current_slice_v1.csv"
      },
      performance_summary: {
        status: "alert",
        overall: { n: 59092, ppe10: 0.3254, mdape: 0.1637, r2: 0.0281 }
      },
      slice_metrics: [
        { slice_key: "ELEVATOR", n: 25785, ppe10: 0.2285, mdape: 0.2224, r2: -0.0931 }
      ],
      retrain_decision: {
        should_retrain: true,
        decision: "retrain",
        reasons: ["performance monitor is alert"],
        policy: { min_ppe10: 0.75 },
        signals: { drift_alerts: 3 }
      },
      degraded: false,
      warnings: []
    });

    const monitoringB = monitoringOverviewSchema.parse({
      ...monitoringA,
      ...meta("api", "future_partner_api")
    });

    expect(normalizeMonitoringForUi(monitoringA)).toEqual(normalizeMonitoringForUi(monitoringB));
  });
});

describe("property explorer + SHAP contracts", () => {
  it("accepts canonical property search response", () => {
    const parsed = propertySearchResponseSchema.parse({
      ...meta("csv", "data/processed/hseason001_train_20260223.csv"),
      query: "beaver",
      total_available: 1,
      total_catalog: 1200,
      items: [
        {
          property_id: "pid_001",
          address: "26 BEAVER STREET",
          borough: "MANHATTAN",
          property_segment: "ELEVATOR",
          lat: 40.7051,
          lng: -74.0104,
          data_quality_status: "ready"
        }
      ]
    });

    expect(parsed.items[0]?.address).toContain("BEAVER");
  });

  it("accepts canonical property nearby response", () => {
    const parsed = propertyNearbyResponseSchema.parse({
      ...meta("csv", "data/processed/hseason001_train_20260223.csv"),
      bbox: { min_lng: -74.2, min_lat: 40.6, max_lng: -73.8, max_lat: 40.82 },
      total_available: 1,
      total_catalog: 1200,
      items: [
        {
          property_id: "pid_001",
          address: "26 BEAVER STREET",
          borough: "MANHATTAN",
          property_segment: "ELEVATOR",
          lat: 40.7051,
          lng: -74.0104,
          data_quality_status: "ready"
        }
      ]
    });

    expect(parsed.bbox.min_lng).toBeLessThan(parsed.bbox.max_lng);
  });

  it("accepts canonical property detail response with inference availability", () => {
    const parsed = propertyDetailResponseSchema.parse({
      ...meta("csv", "data/processed/hseason001_train_20260223.csv"),
      property_id: "pid_001",
      address: "26 BEAVER STREET",
      borough: "MANHATTAN",
      property_segment: "ELEVATOR",
      lat: 40.7051,
      lng: -74.0104,
      data_quality_status: "ready",
      feature_completeness: 1,
      features: {
        gross_square_feet: 1848,
        year_built: 1909,
        residential_units: 2,
        total_units: 2,
        building_class: "D4",
        sale_date: "2021-04-27"
      },
      availability: {
        inference_ready: true,
        missing_required_features: []
      }
    });

    expect(parsed.availability.inference_ready).toBe(true);
  });

  it("accepts global SHAP summary response", () => {
    const parsed = globalShapSummaryResponseSchema.parse({
      ...meta("other", "models/metrics_v1.json|reports/model/segment_scorecard_v1.csv"),
      segment: "SMALL_MULTI",
      window: "180d",
      generated_from: ["models/metrics_v1.json", "reports/model/segment_scorecard_v1.csv"],
      features: [
        { feature_name: "gross_square_feet", mean_abs_shap: 0.31, direction_hint: "positive" },
        { feature_name: "borough", mean_abs_shap: 0.26, direction_hint: "mixed" }
      ]
    });

    expect(parsed.features.length).toBeGreaterThan(0);
  });
});

describe("governance safety + copilot fallback", () => {
  it("enforces read-only governance actions in no-auth mode", () => {
    const parsed = governanceStatusSchema.parse({
      ...meta(),
      registered_model_name: "spec-nyc-avm",
      aliases: {
        champion: { model_version: "1", run_id: "champ_run" },
        challenger: { model_version: null, run_id: null },
        candidate: { model_version: "6", run_id: "cand_run" }
      },
      latest_proposal: {
        proposal_id: "proposal_123",
        status: "no_winner",
        created_at_utc: new Date().toISOString(),
        expires_at_utc: new Date(Date.now() + 3600_000).toISOString(),
        champion: { model_version: "1", run_id: "champ_run" },
        winner: null,
        candidates_ranked: []
      },
      gate_results: [],
      status_reason: "Latest proposal status is 'no_winner'. Governance actions remain read-only in no-auth mode.",
      actions_enabled: false
    });

    expect(parsed.actions_enabled).toBe(false);
  });

  it("returns deterministic fallback copilot output when context is unavailable", () => {
    const fallback = buildCopilotFallback({
      reason: "missing_context_artifacts",
      citations: ["models/metrics_v1.json"]
    });

    const parsed = copilotAskResponseSchema.parse({
      intent_resolved: fallback.intentResolved,
      router_confidence: 0.33,
      answer: fallback.answer,
      citations: fallback.citations,
      assistant_confidence: fallback.assistantConfidence,
      limitations: fallback.limitations,
      actions: fallback.actions,
      safety: fallback.safety,
      ...meta()
    });

    expect(parsed.safety.fallback_used).toBe(true);
    expect(parsed.safety.reason).toBe("missing_context_artifacts");
  });

  it("accepts valid copilot ask request schema", () => {
    const parsed = copilotAskRequestSchema.parse({
      question: "Why this estimate?",
      intent: "auto",
      context: { page: "valuation", valuation_id: "val_abc123", window: "30d" }
    });

    expect(parsed.intent).toBe("auto");
  });
});

describe("accessibility + motion baseline", () => {
  it("includes reduced-motion fallback in global styles", () => {
    const cssPath = path.resolve(
      path.dirname(new URL(import.meta.url).pathname),
      "../../app/globals.css"
    );
    const css = fs.readFileSync(cssPath, "utf-8");
    expect(css.includes("prefers-reduced-motion")).toBe(true);
  });
});
