import { okJson } from "@/src/lib/http";
import { searchCanonicalProperties } from "@/src/bff/clients/canonicalPropertyClient";
import { propertySearchResponseSchema } from "@/src/features/properties/schemas/propertySchemas";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const q = searchParams.get("q") ?? "";
  const borough = searchParams.get("borough") ?? undefined;
  const segment = searchParams.get("segment") ?? undefined;
  const zipCode = searchParams.get("zip_code") ?? undefined;
  const tier = searchParams.get("tier") ?? undefined;
  const limitRaw = Number(searchParams.get("limit") ?? 30);
  const limit = Number.isFinite(limitRaw) ? Math.min(200, Math.max(1, Math.floor(limitRaw))) : 30;

  const canonical = await searchCanonicalProperties({
    query: q,
    limit,
    borough,
    segment,
    zipCode,
    tier
  });

  const parsed = propertySearchResponseSchema.parse({
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
