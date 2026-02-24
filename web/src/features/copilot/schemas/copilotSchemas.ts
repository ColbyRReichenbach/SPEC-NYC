import { z } from "zod";

import { apiMetaSchema } from "@/src/bff/types/baseContracts";

export const copilotPageSchema = z.enum(["valuation", "governance", "monitoring", "global"]);
export const copilotWindowSchema = z.enum(["7d", "30d", "90d", "180d"]);

export const copilotIntentRequestedSchema = z.enum([
  "auto",
  "why_estimate",
  "what_changed_recently",
  "improve_confidence",
  "promotion_status",
  "monitoring_remediation"
]);

export const copilotIntentResolvedSchema = z.enum([
  "why_estimate",
  "what_changed_recently",
  "improve_confidence",
  "promotion_status",
  "monitoring_remediation",
  "unknown"
]);

export const copilotActionSchema = z
  .object({
    title: z.string().min(1),
    owner: z.string().min(1),
    command: z.string().min(1).optional(),
    artifact: z.string().min(1).optional(),
    priority: z.enum(["p0", "p1", "p2"])
  })
  .strict();

export const copilotAskRequestSchema = z
  .object({
    question: z.string().min(3),
    intent: copilotIntentRequestedSchema.default("auto"),
    context: z
      .object({
        page: copilotPageSchema.default("global"),
        property_id: z.string().min(1).optional(),
        valuation_id: z.string().min(1).optional(),
        proposal_id: z.string().min(1).optional(),
        window: copilotWindowSchema.optional(),
        segment: z.string().min(1).optional(),
        tier: z.string().min(1).optional(),
        session_id: z.string().min(1).max(80).optional(),
        simulate_unavailable: z.boolean().optional(),
        debug_trace: z.boolean().optional()
      })
      .default({})
  })
  .strict();

export const copilotAskPayloadSchema = z
  .object({
    intent_resolved: copilotIntentResolvedSchema,
    router_confidence: z.number().min(0).max(1),
    answer: z.string(),
    citations: z.array(z.string()),
    assistant_confidence: z.number().min(0).max(1),
    limitations: z.array(z.string()),
    actions: z.array(copilotActionSchema).optional(),
    safety: z
      .object({
        guardrail_mode: z.string(),
        fallback_used: z.boolean(),
        reason: z.string().optional(),
        prompt_injection_blocked: z.boolean().optional()
      })
      .strict(),
    trace: z
      .object({
        intent_requested: copilotIntentRequestedSchema,
        context_bundle_ids: z.array(z.string()),
        missing_artifact_keys: z.array(z.string()),
        token_budget_exceeded: z.boolean()
      })
      .strict()
      .optional()
  })
  .strict();

export const copilotAskResponseSchema = apiMetaSchema.extend(copilotAskPayloadSchema.shape).strict();

export type CopilotAskRequest = z.infer<typeof copilotAskRequestSchema>;
export type CopilotIntentRequested = z.infer<typeof copilotIntentRequestedSchema>;
export type CopilotIntentResolved = z.infer<typeof copilotIntentResolvedSchema>;
export type CopilotPage = z.infer<typeof copilotPageSchema>;
export type CopilotAskResponse = z.infer<typeof copilotAskResponseSchema>;
export type CopilotAction = z.infer<typeof copilotActionSchema>;
