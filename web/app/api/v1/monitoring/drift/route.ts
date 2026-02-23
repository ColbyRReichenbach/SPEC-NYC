import { okJson } from "@/src/lib/http";

export async function GET() {
  return okJson({
    window: "30d",
    alerts: [],
    summary: "No drift alerts detected"
  });
}
