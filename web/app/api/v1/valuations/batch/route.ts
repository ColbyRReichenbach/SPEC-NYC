import { okJson } from "@/src/lib/http";

export async function POST() {
  return okJson({
    job_id: `job_${crypto.randomUUID().slice(0, 8)}`,
    status: "queued",
    submitted_at: new Date().toISOString(),
    processed_rows: 0,
    total_rows: 0,
    success_rows: 0,
    error_rows: 0
  }, 202);
}
