import { z } from "zod";

import { apiMetaSchema } from "@/src/bff/types/baseContracts";

export const dataQualityStatusSchema = z.enum(["ready", "partial", "sparse"]);

export const canonicalPropertyPreviewSchema = z
  .object({
    property_id: z.string().min(1),
    address: z.string().min(1),
    borough: z.string().min(1),
    zip_code: z.string().min(1).nullable().optional(),
    property_segment: z.string().min(1),
    price_tier_proxy: z.string().min(1).nullable().optional(),
    lat: z.number().min(-90).max(90),
    lng: z.number().min(-180).max(180),
    data_quality_status: dataQualityStatusSchema
  })
  .strict();

export const canonicalPropertyDetailPayloadSchema = canonicalPropertyPreviewSchema
  .extend({
    feature_completeness: z.number().min(0).max(1),
    features: z
      .object({
        gross_square_feet: z.number().positive().nullable(),
        year_built: z.number().int().min(1800).max(2100).nullable(),
        residential_units: z.number().int().nonnegative().nullable(),
        total_units: z.number().int().positive().nullable(),
        building_class: z.string().min(1).nullable(),
        sale_date: z.string().min(1).nullable()
      })
      .strict(),
    availability: z
      .object({
        inference_ready: z.boolean(),
        missing_required_features: z.array(z.string())
      })
      .strict()
  })
  .strict();

export const propertySearchPayloadSchema = z
  .object({
    query: z.string(),
    total_available: z.number().int().nonnegative(),
    total_catalog: z.number().int().nonnegative(),
    items: z.array(canonicalPropertyPreviewSchema)
  })
  .strict();

export const propertySearchResponseSchema = apiMetaSchema.extend(propertySearchPayloadSchema.shape).strict();

export const propertyNearbyPayloadSchema = z
  .object({
    bbox: z
      .object({
        min_lng: z.number(),
        min_lat: z.number(),
        max_lng: z.number(),
        max_lat: z.number()
      })
      .strict(),
    total_available: z.number().int().nonnegative(),
    total_catalog: z.number().int().nonnegative(),
    items: z.array(canonicalPropertyPreviewSchema)
  })
  .strict();

export const propertyNearbyResponseSchema = apiMetaSchema.extend(propertyNearbyPayloadSchema.shape).strict();

export const propertyDetailResponseSchema = apiMetaSchema
  .extend(canonicalPropertyDetailPayloadSchema.shape)
  .strict();

export type CanonicalPropertyPreview = z.infer<typeof canonicalPropertyPreviewSchema>;
export type CanonicalPropertyDetailResponse = z.infer<typeof propertyDetailResponseSchema>;
