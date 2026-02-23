import { okJson } from "@/src/lib/http";

export async function GET() {
  return okJson({
    window: "30d",
    metrics: {
      ppe10: 0.3254,
      mdape: 0.1637,
      r2: 0.0281
    }
  });
}
