import { errorJson, okJson } from "@/src/lib/http";
import { buildCanonicalGlobalShapSummary } from "@/src/bff/clients/canonicalShapClient";
import { globalShapSummaryResponseSchema } from "@/src/features/valuation/schemas/explainabilitySchemas";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const segment = searchParams.get("segment") ?? "ALL";
  const window = searchParams.get("window") ?? "180d";

  let canonical: Awaited<ReturnType<typeof buildCanonicalGlobalShapSummary>>;
  try {
    canonical = await buildCanonicalGlobalShapSummary({ segment, window });
  } catch (error) {
    return errorJson(error instanceof Error ? error.message : "Global explainability extraction failed.", 409);
  }
  const parsed = globalShapSummaryResponseSchema.parse({
    ...canonical.payload,
    contract_version: "v1",
    generated_at: new Date().toISOString(),
    request_id: crypto.randomUUID(),
    source_context: canonical.sourceContext
  });

  const { contract_version: _cv, generated_at: _ga, request_id: _rid, source_context, ...responseBody } =
    parsed;

  return okJson(responseBody, 200, { source_context });
}
