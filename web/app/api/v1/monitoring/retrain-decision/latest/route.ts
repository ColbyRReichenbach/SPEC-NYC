import { okJson } from "@/src/lib/http";

export async function GET() {
  return okJson({
    decision: "hold",
    generated_at_utc: new Date().toISOString(),
    artifact_path: "reports/releases/retrain_decision_latest.json"
  });
}
