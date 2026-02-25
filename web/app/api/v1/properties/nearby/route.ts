import { errorJson, okJson } from "@/src/lib/http";
import { nearbyCanonicalProperties } from "@/src/bff/clients/canonicalPropertyClient";
import { propertyNearbyResponseSchema } from "@/src/features/properties/schemas/propertySchemas";

const DEFAULT_BBOX = {
  min_lng: -74.3,
  min_lat: 40.48,
  max_lng: -73.68,
  max_lat: 40.95
};

function parseBBox(raw: string | null) {
  if (!raw) return DEFAULT_BBOX;
  const parts = raw.split(",").map((token) => Number(token));
  if (parts.length !== 4 || parts.some((value) => !Number.isFinite(value))) return null;
  const [minLng, minLat, maxLng, maxLat] = parts;
  return {
    min_lng: minLng,
    min_lat: minLat,
    max_lng: maxLng,
    max_lat: maxLat
  };
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const scope = searchParams.get("scope") ?? "viewport";
  const borough = searchParams.get("borough") ?? undefined;
  const segment = searchParams.get("segment") ?? undefined;
  const zipCode = searchParams.get("zip_code") ?? undefined;
  const tier = searchParams.get("tier") ?? undefined;
  const bbox = parseBBox(searchParams.get("bbox"));
  if (!bbox) {
    return errorJson("Invalid bbox; expected comma-separated minLng,minLat,maxLng,maxLat", 400);
  }

  const defaultLimit = scope === "all" ? 2_000 : 300;
  const limitRaw = Number(searchParams.get("limit") ?? defaultLimit);
  const limit = Number.isFinite(limitRaw) ? Math.min(2_000, Math.max(1, Math.floor(limitRaw))) : defaultLimit;

  const canonical = await nearbyCanonicalProperties({
    bbox:
      scope === "all"
        ? { min_lng: -180, min_lat: -90, max_lng: 180, max_lat: 90 }
        : bbox,
    limit,
    borough,
    segment,
    zipCode,
    tier
  });
  const parsed = propertyNearbyResponseSchema.parse({
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
