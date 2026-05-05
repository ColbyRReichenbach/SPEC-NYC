import { promises as fs } from "node:fs";
import path from "node:path";

import { resolveRepoRoot } from "@/src/bff/clients/artifactStore";
import { buildCanonicalValuationResponse } from "@/src/bff/clients/canonicalValuationClient";
import {
  singleValuationRequestSchema,
  type SingleValuationRequest
} from "@/src/features/valuation/schemas/valuationSchemas";

export type BatchValuationJob = {
  job_id: string;
  status: "completed" | "failed";
  submitted_at: string;
  completed_at: string;
  processed_rows: number;
  total_rows: number;
  success_rows: number;
  error_rows: number;
  results: Array<{
    row_index: number;
    valuation_id?: string;
    predicted_price?: number;
    error?: string;
  }>;
};

export async function runBatchValuation(payload: unknown): Promise<BatchValuationJob> {
  const requests = parseBatchRequests(payload);
  const submittedAt = new Date().toISOString();
  const jobId = `job_${crypto.randomUUID().slice(0, 12)}`;
  const results: BatchValuationJob["results"] = [];

  for (const [index, request] of requests.entries()) {
    try {
      const response = await buildCanonicalValuationResponse(request);
      results.push({
        row_index: index,
        valuation_id: response.payload.valuation_id,
        predicted_price: response.payload.predicted_price
      });
    } catch (error) {
      results.push({
        row_index: index,
        error: error instanceof Error ? error.message : "Valuation failed."
      });
    }
  }

  const errorRows = results.filter((row) => row.error).length;
  const job: BatchValuationJob = {
    job_id: jobId,
    status: errorRows === requests.length ? "failed" : "completed",
    submitted_at: submittedAt,
    completed_at: new Date().toISOString(),
    processed_rows: requests.length,
    total_rows: requests.length,
    success_rows: requests.length - errorRows,
    error_rows: errorRows,
    results
  };

  await writeBatchJob(job);
  return job;
}

export async function readBatchValuationJob(jobId: string): Promise<BatchValuationJob | null> {
  const safeJobId = assertJobId(jobId);
  const repoRoot = await resolveRepoRoot();
  try {
    return JSON.parse(
      await fs.readFile(path.join(repoRoot, "reports/valuations/batch", `${safeJobId}.json`), "utf-8")
    ) as BatchValuationJob;
  } catch {
    return null;
  }
}

function parseBatchRequests(payload: unknown): SingleValuationRequest[] {
  const input = payload as { requests?: unknown };
  if (!Array.isArray(input?.requests) || input.requests.length === 0) {
    throw new Error("Batch payload must include a non-empty requests array.");
  }
  if (input.requests.length > 25) {
    throw new Error("Local batch API accepts at most 25 valuation requests.");
  }
  return input.requests.map((item) => singleValuationRequestSchema.parse(item));
}

async function writeBatchJob(job: BatchValuationJob) {
  const repoRoot = await resolveRepoRoot();
  const dir = path.join(repoRoot, "reports/valuations/batch");
  await fs.mkdir(dir, { recursive: true });
  await fs.writeFile(path.join(dir, `${job.job_id}.json`), `${JSON.stringify(job, null, 2)}\n`);
}

function assertJobId(value: string) {
  if (!/^job_[a-f0-9-]{8,36}$/i.test(value)) {
    throw new Error("Invalid batch job id.");
  }
  return value;
}
