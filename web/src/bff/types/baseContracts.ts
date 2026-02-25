import { z } from "zod";

export const sourceTypeSchema = z.enum(["rdb", "csv", "api", "other"]);

export const sourceContextSchema = z
  .object({
    source_id: z.string().min(1),
    source_type: sourceTypeSchema
  })
  .strict();

export const apiMetaSchema = z
  .object({
    contract_version: z.literal("v1"),
    request_id: z.string().uuid(),
    generated_at: z.string().datetime(),
    source_context: sourceContextSchema
  })
  .strict();

export type SourceContext = z.infer<typeof sourceContextSchema>;
export type ApiMeta = z.infer<typeof apiMetaSchema>;
