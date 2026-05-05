import { NextResponse } from "next/server";

import { resolveRepoRoot } from "@/src/bff/clients/artifactStore";
import { LifecycleError, recordExperimentReviewDecision, type ReviewDecisionPayload } from "@/src/features/platform/experimentRegistry";

export async function POST(request: Request, { params }: { params: { experimentId: string } }) {
  try {
    const repoRoot = await resolveRepoRoot();
    const payload = (await request.json()) as ReviewDecisionPayload;
    const experiment = await recordExperimentReviewDecision(repoRoot, params.experimentId, payload);
    return NextResponse.json({ experiment });
  } catch (error) {
    return lifecycleErrorResponse(error);
  }
}

function lifecycleErrorResponse(error: unknown) {
  if (error instanceof LifecycleError) {
    return NextResponse.json({ error: error.message }, { status: error.status });
  }

  return NextResponse.json(
    { error: error instanceof Error ? error.message : "Experiment review decision failed." },
    { status: 500 }
  );
}
