import { errorJson, okJson } from "@/src/lib/http";
import {
  propertyExplanationRequestSchema,
  propertyExplanationResponseSchema
} from "@/src/features/valuation/schemas/valuationSchemas";

export async function POST(req: Request) {
  const payload = await req.json().catch(() => null);
  const parsed = propertyExplanationRequestSchema.safeParse(payload);
  if (!parsed.success) {
    return errorJson(`Invalid payload: ${parsed.error.issues[0]?.message ?? "unknown error"}`);
  }

  const response = propertyExplanationResponseSchema.parse({
    valuation_id: parsed.data.valuation_id,
    status: "ready",
    explainer_type: "xgboost_pred_contribs",
    drivers_positive: [
      { feature: "gross_square_feet", impact: 142000, display: "Larger interior size" },
      { feature: "borough", impact: 63000, display: "Location demand baseline" }
    ],
    drivers_negative: [
      { feature: "building_age", impact: -47000, display: "Older building profile" }
    ],
    contract_version: "v1",
    generated_at: new Date().toISOString(),
    request_id: crypto.randomUUID(),
    source_context: {
      source_id: "models/metrics_v1.json",
      source_type: "other"
    }
  });

  const { contract_version: _cv, generated_at: _ga, request_id: _rid, source_context, ...responseBody } =
    response;

  return okJson(responseBody, 200, { source_context });
}
