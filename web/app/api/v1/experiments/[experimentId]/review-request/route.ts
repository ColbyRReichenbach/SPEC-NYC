import { NextResponse } from "next/server";

import { resolveRepoRoot } from "@/src/bff/clients/artifactStore";
import { LifecycleError, requestExperimentReview, type ReviewRequestPayload } from "@/src/features/platform/experimentRegistry";

export async function POST(request: Request, { params }: { params: { experimentId: string } }) {
  try {
    const repoRoot = await resolveRepoRoot();
    const payload = (await request.json().catch(() => ({}))) as ReviewRequestPayload;
    const experiment = await requestExperimentReview(repoRoot, params.experimentId, payload);
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
    { error: error instanceof Error ? error.message : "Experiment review request failed." },
    { status: 500 }
  );
}
