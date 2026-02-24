import { z } from "zod";

import { apiMetaSchema } from "@/src/bff/types/baseContracts";

export const singleValuationRequestSchema = z
  .object({
    property: z
      .object({
        address: z.string().min(3),
        borough: z.string().min(1),
        gross_square_feet: z.number().positive(),
        year_built: z.number().int().min(1800).max(2100),
        residential_units: z.number().int().nonnegative(),
        total_units: z.number().int().positive(),
        building_class: z.string().min(1),
        property_segment: z.string().min(1),
        sale_date: z.string().min(1)
      })
      .strict(),
    context: z
      .object({
        dataset_version: z.string().min(1),
        model_alias: z.enum(["champion", "challenger", "candidate"]).default("champion"),
        property_id: z.string().min(1).optional()
      })
      .strict()
  })
  .strict();

export const driverSchema = z
  .object({
    feature: z.string(),
    impact: z.number(),
    display: z.string()
  })
  .strict();

export const singleValuationPayloadSchema = z
  .object({
    valuation_id: z.string(),
    predicted_price: z.number(),
    prediction_interval: z
      .object({
        low: z.number(),
        high: z.number(),
        method: z.string()
      })
      .strict(),
    confidence: z
      .object({
        score: z.number().min(0).max(1),
        band: z.enum(["high", "medium", "low"]),
        factors: z
          .object({
            segment_calibration: z.number().min(0).max(1),
            support_coverage: z.number().min(0).max(1),
            input_completeness: z.number().min(0).max(1)
          })
          .strict(),
        caveats: z.array(z.string())
      })
      .strict(),
    explanation: z
      .object({
        status: z.enum(["ready", "degraded", "unavailable"]),
        explainer_type: z.string(),
        local_accuracy: z.number().min(0).max(1).optional(),
        drivers_positive: z.array(driverSchema),
        drivers_negative: z.array(driverSchema)
      })
      .strict(),
    model: z
      .object({
        alias: z.enum(["champion", "challenger", "candidate"]),
        run_id: z.string(),
        model_version: z.string(),
        route: z.string()
      })
      .strict(),
    evidence: z
      .object({
        run_card_path: z.string(),
        metrics_path: z.string(),
        shap_summary_path: z.string()
      })
      .strict()
  })
  .strict();

export const singleValuationResponseSchema = apiMetaSchema.extend(singleValuationPayloadSchema.shape).strict();

export const propertyExplanationRequestSchema = z
  .object({
    valuation_id: z.string().min(1),
    top_n: z.number().int().min(1).max(10).default(5)
  })
  .strict();

export const propertyExplanationPayloadSchema = z
  .object({
    valuation_id: z.string(),
    status: z.enum(["ready", "degraded", "unavailable"]),
    explainer_type: z.string(),
    drivers_positive: z.array(driverSchema),
    drivers_negative: z.array(driverSchema)
  })
  .strict();

export const propertyExplanationResponseSchema = apiMetaSchema
  .extend(propertyExplanationPayloadSchema.shape)
  .strict();

export type SingleValuationRequest = z.infer<typeof singleValuationRequestSchema>;
export type SingleValuationResponse = z.infer<typeof singleValuationResponseSchema>;
