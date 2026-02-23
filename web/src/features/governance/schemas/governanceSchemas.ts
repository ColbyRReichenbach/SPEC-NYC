import { z } from "zod";

export const governanceStatusSchema = z.object({
  registered_model_name: z.string(),
  aliases: z.object({
    champion: z.object({ model_version: z.string().nullable(), run_id: z.string().nullable() }),
    challenger: z.object({ model_version: z.string().nullable(), run_id: z.string().nullable() }),
    candidate: z.object({ model_version: z.string().nullable(), run_id: z.string().nullable() })
  })
});

export const proposalActionRequestSchema = z.object({
  reason: z.string().min(3).optional(),
  actor: z.string().min(1)
});
