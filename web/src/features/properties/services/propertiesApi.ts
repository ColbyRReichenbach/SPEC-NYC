import {
  propertyDetailResponseSchema,
  propertyNearbyResponseSchema,
  propertySearchResponseSchema
} from "@/src/features/properties/schemas/propertySchemas";

type PropertySearchInput = {
  q: string;
  limit?: number;
  borough?: string;
  segment?: string;
  zipCode?: string;
  tier?: string;
};

type NearbyInput = {
  bbox: {
    minLng: number;
    minLat: number;
    maxLng: number;
    maxLat: number;
  };
  limit?: number;
  scope?: "viewport" | "all";
  borough?: string;
  segment?: string;
  zipCode?: string;
  tier?: string;
};

export async function searchProperties(input: PropertySearchInput) {
  const params = new URLSearchParams();
  params.set("q", input.q);
  if (input.limit) params.set("limit", String(input.limit));
  if (input.borough) params.set("borough", input.borough);
  if (input.segment) params.set("segment", input.segment);
  if (input.zipCode) params.set("zip_code", input.zipCode);
  if (input.tier) params.set("tier", input.tier);

  const response = await fetch(`/api/v1/properties/search?${params.toString()}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Property search failed with status ${response.status}`);
  }

  return propertySearchResponseSchema.parse(await response.json());
}

export async function fetchNearbyProperties(input: NearbyInput) {
  const params = new URLSearchParams();
  params.set(
    "bbox",
    [input.bbox.minLng, input.bbox.minLat, input.bbox.maxLng, input.bbox.maxLat].join(",")
  );
  if (input.limit) params.set("limit", String(input.limit));
  if (input.scope) params.set("scope", input.scope);
  if (input.borough) params.set("borough", input.borough);
  if (input.segment) params.set("segment", input.segment);
  if (input.zipCode) params.set("zip_code", input.zipCode);
  if (input.tier) params.set("tier", input.tier);

  const response = await fetch(`/api/v1/properties/nearby?${params.toString()}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Nearby property lookup failed with status ${response.status}`);
  }

  return propertyNearbyResponseSchema.parse(await response.json());
}

export async function fetchPropertyDetail(propertyId: string) {
  const response = await fetch(`/api/v1/properties/${encodeURIComponent(propertyId)}`, {
    cache: "no-store"
  });
  if (!response.ok) {
    throw new Error(`Property detail request failed with status ${response.status}`);
  }

  return propertyDetailResponseSchema.parse(await response.json());
}
