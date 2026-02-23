import { z } from "zod";

export const copilotAskRequestSchema = z.object({
  question: z.string().min(3),
  intent: z.enum(["why_estimate", "what_changed_recently", "improve_confidence"]),
  context: z
    .object({
      valuation_id: z.string().optional(),
      proposal_id: z.string().optional(),
      window: z.string().optional()
    })
    .default({})
});

export const copilotAskResponseSchema = z.object({
  answer: z.string(),
  citations: z.array(z.string()),
  assistant_confidence: z.number().min(0).max(1),
  limitations: z.array(z.string()),
  safety: z.object({
    guardrail_mode: z.string(),
    fallback_used: z.boolean(),
    reason: z.string().optional()
  })
});
