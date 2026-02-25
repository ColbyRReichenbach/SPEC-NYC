import { errorJson, okJson } from "@/src/lib/http";
import { getCanonicalPropertyDetail } from "@/src/bff/clients/canonicalPropertyClient";
import { propertyDetailResponseSchema } from "@/src/features/properties/schemas/propertySchemas";

export async function GET(_: Request, { params }: { params: { propertyId: string } }) {
  const propertyId = decodeURIComponent(params.propertyId);
  const canonical = await getCanonicalPropertyDetail(propertyId);

  if (!canonical.payload) {
    return errorJson(`Property not found: ${propertyId}`, 404);
  }

  const parsed = propertyDetailResponseSchema.parse({
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
