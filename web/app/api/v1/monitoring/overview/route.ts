import { okJson } from "@/src/lib/http";
import { buildCanonicalMonitoringOverview } from "@/src/bff/clients/canonicalMonitoringClient";
import { monitoringOverviewSchema } from "@/src/features/monitoring/schemas/monitoringSchemas";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const window = searchParams.get("window") ?? "30d";

  const canonical = await buildCanonicalMonitoringOverview(window);
  const parsed = monitoringOverviewSchema.parse({
    ...canonical.payload,
    contract_version: "v1",
    generated_at: new Date().toISOString(),
    request_id: crypto.randomUUID(),
    source_context: canonical.sourceContext
  });

  const { contract_version: _cv, generated_at: _ga, request_id: _rid, source_context, ...responseBody } = parsed;

  return okJson(responseBody, 200, { source_context });
}
