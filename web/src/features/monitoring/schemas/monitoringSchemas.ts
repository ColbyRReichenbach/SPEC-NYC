import { z } from "zod";

import { apiMetaSchema } from "@/src/bff/types/baseContracts";

const driftSummarySchema = z
  .object({
    status: z.string(),
    alerts: z.number().int().nonnegative(),
    warnings: z.number().int().nonnegative(),
    rows: z.number().int().nonnegative(),
    reference_csv: z.string().optional(),
    current_csv: z.string().optional()
  })
  .strict();

const overallMetricsSchema = z
  .object({
    n: z.number().int().nonnegative(),
    ppe10: z.number(),
    mdape: z.number(),
    r2: z.number()
  })
  .strict();

const sliceMetricSchema = z
  .object({
    slice_key: z.string(),
    n: z.number().int().nonnegative(),
    ppe10: z.number(),
    mdape: z.number(),
    r2: z.number()
  })
  .strict();

const retrainDecisionSchema = z
  .object({
    should_retrain: z.boolean(),
    decision: z.string(),
    reasons: z.array(z.string()),
    policy: z.record(z.string(), z.number()),
    signals: z.record(z.string(), z.union([z.number(), z.string(), z.boolean()]))
  })
  .strict();

export const monitoringOverviewPayloadSchema = z
  .object({
    window: z.string(),
    drift_summary: driftSummarySchema,
    performance_summary: z
      .object({
        status: z.string(),
        overall: overallMetricsSchema
      })
      .strict(),
    slice_metrics: z.array(sliceMetricSchema),
    retrain_decision: retrainDecisionSchema,
    degraded: z.boolean(),
    warnings: z.array(z.string())
  })
  .strict();

export const monitoringOverviewSchema = apiMetaSchema.extend(monitoringOverviewPayloadSchema.shape).strict();
