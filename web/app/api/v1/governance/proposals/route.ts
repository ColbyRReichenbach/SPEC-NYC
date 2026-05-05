import { NextResponse } from "next/server";

import { resolveRepoRoot } from "@/src/bff/clients/artifactStore";
import {
  createReleaseProposal,
  listGovernanceState,
  ReleaseLifecycleError
} from "@/src/features/platform/releaseRegistry";

export const dynamic = "force-dynamic";

export async function GET() {
  const repoRoot = await resolveRepoRoot();
  const state = await listGovernanceState(repoRoot);
  return NextResponse.json(state);
}

export async function POST(request: Request) {
  try {
    const payload = (await request.json()) as { experimentId?: string };
    if (!payload.experimentId) {
      return NextResponse.json({ error: "experimentId is required." }, { status: 400 });
    }

    const repoRoot = await resolveRepoRoot();
    const proposal = await createReleaseProposal(repoRoot, payload.experimentId);
    const state = await listGovernanceState(repoRoot);
    return NextResponse.json({ proposal, ...state });
  } catch (error) {
    return lifecycleErrorResponse(error, "Release proposal creation failed.");
  }
}

function lifecycleErrorResponse(error: unknown, fallback: string) {
  if (error instanceof ReleaseLifecycleError) {
    return NextResponse.json({ error: error.message }, { status: error.status });
  }

  return NextResponse.json(
    { error: error instanceof Error ? error.message : fallback },
    { status: 500 }
  );
}
