import { NextResponse } from "next/server";

import { resolveRepoRoot } from "@/src/bff/clients/artifactStore";
import { listExperimentRuns } from "@/src/features/platform/experimentRegistry";

export async function GET() {
  const repoRoot = await resolveRepoRoot();
  const experiments = await listExperimentRuns(repoRoot);
  return NextResponse.json({ experiments });
}
