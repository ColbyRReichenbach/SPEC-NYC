import { okJson } from "@/src/lib/http";
import { buildCanonicalMonitoringOverview } from "@/src/bff/clients/canonicalMonitoringClient";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const window = searchParams.get("window") ?? "30d";
  const canonical = await buildCanonicalMonitoringOverview(window);

  return okJson(
    {
      window,
      performance_summary: canonical.payload.performance_summary,
      slice_metrics: canonical.payload.slice_metrics,
      degraded: canonical.payload.degraded,
      warnings: canonical.payload.warnings
    },
    200,
    { source_context: canonical.sourceContext }
  );
}
