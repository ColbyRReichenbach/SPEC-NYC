import { okJson } from "@/src/lib/http";
import { buildCanonicalGlobalShapSummary } from "@/src/bff/clients/canonicalShapClient";
import { globalShapSummaryResponseSchema } from "@/src/features/valuation/schemas/explainabilitySchemas";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const segment = searchParams.get("segment") ?? "ALL";
  const window = searchParams.get("window") ?? "180d";

  const canonical = await buildCanonicalGlobalShapSummary({ segment, window });
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
