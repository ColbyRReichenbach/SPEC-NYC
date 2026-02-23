import { okJson } from "@/src/lib/http";

export async function GET() {
  return okJson({
    window: "30d",
    drift_alerts: 0,
    ppe10: 0.3254,
    mdape: 0.1637,
    retrain_recommended: false
  });
}
