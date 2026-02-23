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
      { feature: "gross_square_feet", impact: 142000, display: "Larger interior size" }
    ],
    drivers_negative: [
      { feature: "building_age", impact: -47000, display: "Older building profile" }
    ]
  });

  return okJson(response);
}
