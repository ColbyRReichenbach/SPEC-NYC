import { governanceStatusSchema, type GovernanceStatusResponse } from "@/src/features/governance/schemas/governanceSchemas";

export async function fetchGovernanceStatus(): Promise<GovernanceStatusResponse> {
  const response = await fetch("/api/v1/governance/status", { cache: "no-store" });

  if (!response.ok) {
    throw new Error(`Governance status request failed with status ${response.status}`);
  }

  const data = await response.json();
  return governanceStatusSchema.parse(data);
}
