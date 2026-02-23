import { okJson } from "@/src/lib/http";
import { governanceStatusSchema } from "@/src/features/governance/schemas/governanceSchemas";

export async function GET() {
  const payload = governanceStatusSchema.parse({
    registered_model_name: "spec-nyc-avm",
    aliases: {
      champion: { model_version: "1", run_id: "34e917e198af4e58adb2097b8d9ca229" },
      challenger: { model_version: null, run_id: null },
      candidate: { model_version: "6", run_id: "879ab7838c214d3a907e34a687978264" }
    }
  });

  return okJson(payload);
}
