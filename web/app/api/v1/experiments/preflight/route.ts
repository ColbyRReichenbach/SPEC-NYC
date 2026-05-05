import { NextResponse } from "next/server";

import { resolveRepoRoot } from "@/src/bff/clients/artifactStore";
import { loadPlatformData } from "@/src/features/platform/data";
import {
  buildExperimentRunBundle,
  listExperimentRuns,
  validateExperimentRequest,
  writeExperimentRunBundle,
  type ExperimentRequest
} from "@/src/features/platform/experimentRegistry";

export async function GET() {
  const repoRoot = await resolveRepoRoot();
  const experiments = await listExperimentRuns(repoRoot);
  return NextResponse.json({ experiments });
}

export async function POST(request: Request) {
  const payload = (await request.json()) as ExperimentRequest;
  const validationError = validateExperimentRequest(payload);

  if (validationError) {
    return NextResponse.json({ error: validationError }, { status: 400 });
  }

  const repoRoot = await resolveRepoRoot();
  const data = await loadPlatformData();
  const result = buildExperimentRunBundle({ payload, data });
  await writeExperimentRunBundle(repoRoot, result);

  return NextResponse.json({ experiment: result.bundle });
}
