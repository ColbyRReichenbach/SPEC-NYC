import { okJson } from "@/src/lib/http";

export async function GET(_: Request, { params }: { params: { jobId: string } }) {
  return okJson({
    job_id: params.jobId,
    status: "running",
    submitted_at: new Date(Date.now() - 15000).toISOString(),
    processed_rows: 4200,
    total_rows: 12000,
    success_rows: 4125,
    error_rows: 75
  });
}
