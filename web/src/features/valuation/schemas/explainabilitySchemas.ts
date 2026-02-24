import { z } from "zod";

import { apiMetaSchema } from "@/src/bff/types/baseContracts";

export const shapDirectionHintSchema = z.enum(["positive", "negative", "mixed"]);

export const globalShapFeatureSchema = z
  .object({
    feature_name: z.string().min(1),
    mean_abs_shap: z.number().nonnegative(),
    direction_hint: shapDirectionHintSchema
  })
  .strict();

export const globalShapSummaryPayloadSchema = z
  .object({
    segment: z.string().min(1),
    window: z.string().min(1),
    features: z.array(globalShapFeatureSchema),
    generated_from: z.array(z.string())
  })
  .strict();

export const globalShapSummaryResponseSchema = apiMetaSchema
  .extend(globalShapSummaryPayloadSchema.shape)
  .strict();

export type GlobalShapSummaryResponse = z.infer<typeof globalShapSummaryResponseSchema>;
