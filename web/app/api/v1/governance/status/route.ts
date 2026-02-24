import { okJson } from "@/src/lib/http";
import { buildCanonicalGovernanceStatus } from "@/src/bff/clients/canonicalGovernanceClient";
import { governanceStatusSchema } from "@/src/features/governance/schemas/governanceSchemas";

export async function GET() {
  const canonical = await buildCanonicalGovernanceStatus();
  const parsed = governanceStatusSchema.parse({
    ...canonical.payload,
    contract_version: "v1",
    generated_at: new Date().toISOString(),
    request_id: crypto.randomUUID(),
    source_context: canonical.sourceContext
  });

  const { contract_version: _cv, generated_at: _ga, request_id: _rid, source_context, ...responseBody } = parsed;

  return okJson(responseBody, 200, { source_context });
}
