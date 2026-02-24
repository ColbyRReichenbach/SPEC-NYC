import { parse } from "csv-parse/sync";

import type { SourceContext } from "@/src/bff/types/baseContracts";
import {
  latestArtifactPath,
  readTextArtifact,
  readTextArtifactHead
} from "@/src/bff/clients/artifactStore";

type RawRow = Record<string, string | undefined>;

type CanonicalPropertyRecord = {
  property_id: string;
  address: string;
  borough: string;
  zip_code: string | null;
  property_segment: string;
  price_tier_proxy: string | null;
  lat: number;
  lng: number;
  data_quality_status: "ready" | "partial" | "sparse";
  feature_completeness: number;
  features: {
    gross_square_feet: number | null;
    year_built: number | null;
    residential_units: number | null;
    total_units: number | null;
    building_class: string | null;
    sale_date: string | null;
  };
  availability: {
    inference_ready: boolean;
    missing_required_features: string[];
  };
};

type PropertySearchFilters = {
  query: string;
  limit: number;
  borough?: string;
  segment?: string;
  zipCode?: string;
  tier?: string;
};

type NearbyInput = {
  bbox: {
    min_lng: number;
    min_lat: number;
    max_lng: number;
    max_lat: number;
  };
  limit: number;
  borough?: string;
  segment?: string;
  zipCode?: string;
  tier?: string;
};

let cachedRecords: CanonicalPropertyRecord[] | null = null;
let cachedSourcePath: string | null = null;
let cachedAt = 0;

const CACHE_TTL_MS = 5 * 60 * 1000;
const MAX_PARSED_ROWS = 12_000;

const BOROUGH_MAP: Record<string, string> = {
  "1": "MANHATTAN",
  "2": "BRONX",
  "3": "BROOKLYN",
  "4": "QUEENS",
  "5": "STATEN_ISLAND"
};

function parseNumber(raw: string | undefined): number | null {
  if (!raw) return null;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseIntInRange(raw: string | undefined, min: number, max?: number): number | null {
  const parsed = parseNumber(raw);
  if (parsed === null || !Number.isInteger(parsed)) return null;
  if (parsed < min) return null;
  if (typeof max === "number" && parsed > max) return null;
  return parsed;
}

function normalizeBorough(raw: string | undefined): string {
  if (!raw) return "UNKNOWN";
  const trimmed = raw.trim();
  if (BOROUGH_MAP[trimmed]) return BOROUGH_MAP[trimmed];
  return trimmed.toUpperCase().replace(/\s+/g, "_");
}

function normalizeSegment(raw: string | undefined): string {
  return (raw?.trim() || "UNKNOWN").toUpperCase().replace(/\s+/g, "_");
}

function normalizeZipCode(raw: string | undefined): string | null {
  if (!raw) return null;
  const digits = raw.replace(/\D/g, "");
  if (digits.length < 5) return null;
  return digits.slice(0, 5);
}

function normalizeTier(raw: string | undefined): string | null {
  const value = raw?.trim();
  if (!value) return null;
  return value.toUpperCase().replace(/\s+/g, "_");
}

function computeCompleteness(features: CanonicalPropertyRecord["features"]): number {
  const checks = [
    features.gross_square_feet !== null,
    features.year_built !== null,
    features.total_units !== null,
    features.building_class !== null,
    features.sale_date !== null
  ];
  const complete = checks.filter(Boolean).length;
  return Number((complete / checks.length).toFixed(2));
}

function computeDataQuality(featureCompleteness: number): "ready" | "partial" | "sparse" {
  if (featureCompleteness >= 0.9) return "ready";
  if (featureCompleteness >= 0.6) return "partial";
  return "sparse";
}

function requiredMissing(features: CanonicalPropertyRecord["features"]): string[] {
  const missing: string[] = [];
  if (features.gross_square_feet === null) missing.push("gross_square_feet");
  if (features.year_built === null) missing.push("year_built");
  if (features.total_units === null) missing.push("total_units");
  if (features.building_class === null) missing.push("building_class");
  return missing;
}

async function resolvePropertyDatasetPath(): Promise<string> {
  const latestTrainPath = await latestArtifactPath("data/processed", (name) =>
    /_train_\d{8}\.csv$/i.test(name)
  );
  return latestTrainPath ?? "data/processed/hseason001_train_20260223.csv";
}

function toSourceContext(sourcePath: string): SourceContext {
  return {
    source_id: sourcePath,
    source_type: "csv"
  };
}

async function loadPropertyCatalog(): Promise<{ records: CanonicalPropertyRecord[]; sourcePath: string }> {
  const now = Date.now();
  if (cachedRecords && cachedSourcePath && now - cachedAt < CACHE_TTL_MS) {
    return { records: cachedRecords, sourcePath: cachedSourcePath };
  }

  const sourcePath = await resolvePropertyDatasetPath();
  const rawHead =
    (await readTextArtifactHead(sourcePath, 4_500_000)) ??
    (await readTextArtifact(sourcePath)) ??
    "";

  const cutoff = rawHead.lastIndexOf("\n");
  const parseable = cutoff > 0 ? rawHead.slice(0, cutoff) : rawHead;
  const rows = parse(parseable, {
    columns: true,
    skip_empty_lines: true,
    trim: true
  }) as RawRow[];

  const byId = new Map<string, CanonicalPropertyRecord>();

  for (const [index, row] of rows.entries()) {
    if (index >= MAX_PARSED_ROWS) break;

    const lat = parseNumber(row.latitude);
    const lng = parseNumber(row.longitude);
    if (lat === null || lng === null) continue;
    if (lat < -90 || lat > 90 || lng < -180 || lng > 180) continue;

    const address = row.address?.trim();
    if (!address) continue;

    const propertyId =
      row.property_id?.trim() ||
      `${row.bbl?.trim() || "unknown_bbl"}:${address.toUpperCase()}:${row.sale_date?.trim() || "na"}`;

    if (byId.has(propertyId)) continue;

    const features: CanonicalPropertyRecord["features"] = {
      gross_square_feet: (() => {
        const value = parseNumber(row.gross_square_feet);
        return value !== null && value > 0 ? value : null;
      })(),
      year_built: parseIntInRange(row.year_built, 1800, 2100),
      residential_units: parseIntInRange(row.residential_units, 0),
      total_units: parseIntInRange(row.total_units, 1),
      building_class: row.building_class?.trim() || null,
      sale_date: row.sale_date?.trim() || null
    };

    const featureCompleteness = computeCompleteness(features);
    const missingRequired = requiredMissing(features);

    byId.set(propertyId, {
      property_id: propertyId,
      address,
      borough: normalizeBorough(row.borough),
      zip_code: normalizeZipCode(row.zip_code),
      property_segment: normalizeSegment(row.property_segment),
      price_tier_proxy: normalizeTier(row.price_tier_proxy),
      lat,
      lng,
      data_quality_status: computeDataQuality(featureCompleteness),
      feature_completeness: featureCompleteness,
      features,
      availability: {
        inference_ready: missingRequired.length === 0,
        missing_required_features: missingRequired
      }
    });
  }

  const records = Array.from(byId.values());
  cachedRecords = records;
  cachedSourcePath = sourcePath;
  cachedAt = now;

  return { records, sourcePath };
}

export async function searchCanonicalProperties(input: PropertySearchFilters): Promise<{
  payload: {
    query: string;
    total_available: number;
    total_catalog: number;
    items: Array<Pick<CanonicalPropertyRecord, "property_id" | "address" | "borough" | "zip_code" | "property_segment" | "price_tier_proxy" | "lat" | "lng" | "data_quality_status">>;
  };
  sourceContext: SourceContext;
}> {
  const { records, sourcePath } = await loadPropertyCatalog();
  const query = input.query.trim().toLowerCase();

  const filtered = records.filter((item) => {
    if (input.borough && item.borough !== input.borough.toUpperCase()) return false;
    if (input.segment && item.property_segment !== input.segment.toUpperCase()) return false;
    if (input.zipCode && item.zip_code !== input.zipCode) return false;
    if (input.tier && item.price_tier_proxy !== input.tier.toUpperCase()) return false;
    if (!query) return true;
    return (
      item.address.toLowerCase().includes(query) ||
      item.borough.toLowerCase().includes(query) ||
      item.property_segment.toLowerCase().includes(query)
    );
  });

  return {
    payload: {
      query: input.query,
      total_available: filtered.length,
      total_catalog: records.length,
      items: filtered.slice(0, input.limit).map((item) => ({
        property_id: item.property_id,
        address: item.address,
        borough: item.borough,
        zip_code: item.zip_code,
        property_segment: item.property_segment,
        price_tier_proxy: item.price_tier_proxy,
        lat: item.lat,
        lng: item.lng,
        data_quality_status: item.data_quality_status
      }))
    },
    sourceContext: toSourceContext(sourcePath)
  };
}

export async function nearbyCanonicalProperties(input: NearbyInput): Promise<{
  payload: {
    bbox: NearbyInput["bbox"];
    total_available: number;
    total_catalog: number;
    items: Array<Pick<CanonicalPropertyRecord, "property_id" | "address" | "borough" | "zip_code" | "property_segment" | "price_tier_proxy" | "lat" | "lng" | "data_quality_status">>;
  };
  sourceContext: SourceContext;
}> {
  const { records, sourcePath } = await loadPropertyCatalog();

  const filtered = records.filter(
    (item) =>
      item.lng >= input.bbox.min_lng &&
      item.lng <= input.bbox.max_lng &&
      item.lat >= input.bbox.min_lat &&
      item.lat <= input.bbox.max_lat &&
      (!input.borough || item.borough === input.borough.toUpperCase()) &&
      (!input.segment || item.property_segment === input.segment.toUpperCase()) &&
      (!input.zipCode || item.zip_code === input.zipCode) &&
      (!input.tier || item.price_tier_proxy === input.tier.toUpperCase())
  );

  return {
    payload: {
      bbox: input.bbox,
      total_available: filtered.length,
      total_catalog: records.length,
      items: filtered.slice(0, input.limit).map((item) => ({
        property_id: item.property_id,
        address: item.address,
        borough: item.borough,
        zip_code: item.zip_code,
        property_segment: item.property_segment,
        price_tier_proxy: item.price_tier_proxy,
        lat: item.lat,
        lng: item.lng,
        data_quality_status: item.data_quality_status
      }))
    },
    sourceContext: toSourceContext(sourcePath)
  };
}

export async function getCanonicalPropertyDetail(propertyId: string): Promise<{
  payload: CanonicalPropertyRecord | null;
  sourceContext: SourceContext;
}> {
  const { records, sourcePath } = await loadPropertyCatalog();
  const item = records.find((record) => record.property_id === propertyId) ?? null;

  return {
    payload: item,
    sourceContext: toSourceContext(sourcePath)
  };
}
