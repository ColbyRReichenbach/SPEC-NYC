import { readBatchValuationJob } from "@/src/bff/clients/batchValuationClient";
import { errorJson, okJson } from "@/src/lib/http";

export async function GET(_: Request, { params }: { params: { jobId: string } }) {
  let job;
  try {
    job = await readBatchValuationJob(params.jobId);
  } catch (error) {
    return errorJson(error instanceof Error ? error.message : "Invalid batch job id.", 400);
  }

  if (!job) {
    return errorJson(`Batch valuation job not found: ${params.jobId}.`, 404);
  }

  return okJson(job, 200, {
    source_context: {
      source_id: `reports/valuations/batch/${job.job_id}.json`,
      source_type: "other"
    }
  });
}
