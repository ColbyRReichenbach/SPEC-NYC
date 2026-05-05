import { errorJson, okJson } from "@/src/lib/http";
import { buildCanonicalValuationResponse } from "@/src/bff/clients/canonicalValuationClient";
import {
  singleValuationRequestSchema,
  singleValuationResponseSchema
} from "@/src/features/valuation/schemas/valuationSchemas";

export async function POST(req: Request) {
  const payload = await req.json().catch(() => null);
  const parsedRequest = singleValuationRequestSchema.safeParse(payload);
  if (!parsedRequest.success) {
    return errorJson(`Invalid payload: ${parsedRequest.error.issues[0]?.message ?? "unknown error"}`);
  }

  let canonical: Awaited<ReturnType<typeof buildCanonicalValuationResponse>>;
  try {
    canonical = await buildCanonicalValuationResponse(parsedRequest.data);
  } catch (error) {
    return errorJson(error instanceof Error ? error.message : "Model-backed valuation failed.", 409);
  }
  const parsedResponse = singleValuationResponseSchema.parse({
    ...canonical.payload,
    contract_version: "v1",
    generated_at: new Date().toISOString(),
    request_id: crypto.randomUUID(),
    source_context: canonical.sourceContext
  });

  const { contract_version: _cv, generated_at: _ga, request_id: _rid, source_context: _sc, ...responseBody } =
    parsedResponse;

  return okJson(responseBody, 200, { source_context: canonical.sourceContext });
}
