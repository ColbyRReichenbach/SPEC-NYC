import { z } from "zod";

import { apiMetaSchema } from "@/src/bff/types/baseContracts";

const aliasEntrySchema = z
  .object({
    model_version: z.string().nullable(),
    run_id: z.string().nullable()
  })
  .strict();

const candidateRankSchema = z
  .object({
    run_id: z.string(),
    model_version: z.string(),
    gate_pass: z.boolean(),
    weighted_segment_mdape_improvement: z.number(),
    overall_ppe10_lift: z.number(),
    max_major_segment_ppe10_drop: z.number(),
    min_major_segment_ppe10: z.number(),
    drift_alert_delta: z.number(),
    fairness_alert_delta: z.number()
  })
  .strict();

const gateResultSchema = z
  .object({
    gate_key: z.string(),
    label: z.string(),
    status: z.enum(["pass", "fail"]),
    threshold: z.string(),
    actual: z.number()
  })
  .strict();

export const governanceStatusPayloadSchema = z
  .object({
    registered_model_name: z.string(),
    aliases: z
      .object({
        champion: aliasEntrySchema,
        challenger: aliasEntrySchema,
        candidate: aliasEntrySchema
      })
      .strict(),
    latest_proposal: z
      .object({
        proposal_id: z.string(),
        status: z.string(),
        created_at_utc: z.string().datetime(),
        expires_at_utc: z.string().datetime(),
        champion: aliasEntrySchema,
        winner: z
          .object({
            run_id: z.string(),
            model_version: z.string()
          })
          .nullable(),
        candidates_ranked: z.array(candidateRankSchema)
      })
      .strict(),
    gate_results: z.array(gateResultSchema),
    status_reason: z.string(),
    actions_enabled: z.boolean()
  })
  .strict();

export const governanceStatusSchema = apiMetaSchema.extend(governanceStatusPayloadSchema.shape).strict();

export const proposalActionRequestSchema = z
  .object({
    reason: z.string().min(3).optional(),
    actor: z.string().min(1)
  })
  .strict();

export type GovernanceStatusResponse = z.infer<typeof governanceStatusSchema>;
