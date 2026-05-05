import { spawn } from "node:child_process";

import { NextResponse } from "next/server";

import { resolveRepoRoot } from "@/src/bff/clients/artifactStore";
import { LifecycleError, readExperimentRun } from "@/src/features/platform/experimentRegistry";

export async function POST(_request: Request, { params }: { params: { experimentId: string } }) {
  try {
    const repoRoot = await resolveRepoRoot();
    const experiment = await readExperimentRun(repoRoot, params.experimentId);

    if (!experiment) {
      return NextResponse.json({ error: `Experiment not found: ${params.experimentId}.` }, { status: 404 });
    }

    if (experiment.status !== "queued") {
      return NextResponse.json(
        { error: `Only queued experiments can start a worker. Current status is ${experiment.status}.` },
        { status: 409 }
      );
    }

    const child = spawn(
      "python3",
      ["-m", "src.experiments.worker", "--repo-root", repoRoot, "--experiment-id", params.experimentId, "--once"],
      {
        cwd: repoRoot,
        detached: true,
        env: process.env,
        stdio: "ignore"
      }
    );
    child.unref();

    return NextResponse.json({
      experiment,
      worker: {
        status: "started",
        pid: child.pid,
        command: `python3 -m src.experiments.worker --repo-root ${repoRoot} --experiment-id ${params.experimentId} --once`
      }
    });
  } catch (error) {
    if (error instanceof LifecycleError) {
      return NextResponse.json({ error: error.message }, { status: error.status });
    }

    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Experiment worker start failed." },
      { status: 500 }
    );
  }
}
