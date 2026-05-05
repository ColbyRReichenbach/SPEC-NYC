import { NextResponse } from "next/server";

import { resolveRepoRoot } from "@/src/bff/clients/artifactStore";
import {
  listGovernanceState,
  rejectReleaseProposal,
  ReleaseLifecycleError,
  type ProposalDecisionPayload
} from "@/src/features/platform/releaseRegistry";

export const dynamic = "force-dynamic";

export async function POST(request: Request, { params }: { params: { proposalId: string } }) {
  try {
    const payload = (await request.json()) as ProposalDecisionPayload;
    const repoRoot = await resolveRepoRoot();
    const proposal = await rejectReleaseProposal(repoRoot, params.proposalId, payload);
    const state = await listGovernanceState(repoRoot);
    return NextResponse.json({ proposal, ...state });
  } catch (error) {
    return lifecycleErrorResponse(error, "Release proposal rejection failed.");
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
