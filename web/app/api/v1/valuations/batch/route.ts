import { runBatchValuation } from "@/src/bff/clients/batchValuationClient";
import { errorJson, okJson } from "@/src/lib/http";

export async function POST(req: Request) {
  const payload = await req.json().catch(() => null);
  try {
    const job = await runBatchValuation(payload);
    return okJson(job, 201, {
      source_context: {
        source_id: `reports/valuations/batch/${job.job_id}.json`,
        source_type: "other"
      }
    });
  } catch (error) {
    return errorJson(error instanceof Error ? error.message : "Batch valuation failed.", 400);
  }
}
