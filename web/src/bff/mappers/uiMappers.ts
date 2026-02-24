import type { SingleValuationResponse } from "@/src/features/valuation/schemas/valuationSchemas";
import type { GovernanceStatusResponse } from "@/src/features/governance/schemas/governanceSchemas";
import type { z } from "zod";

import { monitoringOverviewSchema } from "@/src/features/monitoring/schemas/monitoringSchemas";

export type MonitoringOverviewResponse = z.infer<typeof monitoringOverviewSchema>;

export function normalizeValuationForUi(payload: SingleValuationResponse) {
  return {
    valuation_id: payload.valuation_id,
    predicted_price: payload.predicted_price,
    confidence_band: payload.confidence.band,
    route: payload.model.route,
    positive_driver_count: payload.explanation.drivers_positive.length,
    negative_driver_count: payload.explanation.drivers_negative.length
  };
}

export function normalizeGovernanceForUi(payload: GovernanceStatusResponse) {
  return {
    proposal_status: payload.latest_proposal.status,
    gate_failures: payload.gate_results.filter((gate) => gate.status === "fail").length,
    actions_enabled: payload.actions_enabled
  };
}

export function normalizeMonitoringForUi(payload: MonitoringOverviewResponse) {
  return {
    drift_status: payload.drift_summary.status,
    overall_ppe10: payload.performance_summary.overall.ppe10,
    retrain_decision: payload.retrain_decision.decision,
    slice_count: payload.slice_metrics.length,
    degraded: payload.degraded
  };
}
