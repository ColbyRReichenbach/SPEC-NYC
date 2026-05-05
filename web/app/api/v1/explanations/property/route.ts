import { errorJson, okJson } from "@/src/lib/http";
import { readStoredValuationExplanation } from "@/src/bff/clients/canonicalValuationClient";
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

  const stored = await readStoredValuationExplanation(parsed.data.valuation_id);
  if (!stored) {
    return errorJson("No persisted valuation explanation exists for this valuation_id.", 404);
  }

  const response = propertyExplanationResponseSchema.parse({
    valuation_id: parsed.data.valuation_id,
    ...stored.explanation,
    contract_version: "v1",
    generated_at: new Date().toISOString(),
    request_id: crypto.randomUUID(),
    source_context: stored.sourceContext
  });

  const { contract_version: _cv, generated_at: _ga, request_id: _rid, source_context, ...responseBody } =
    response;

  return okJson(responseBody, 200, { source_context });
}
