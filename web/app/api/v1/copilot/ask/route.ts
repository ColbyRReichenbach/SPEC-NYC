import { errorJson, okJson } from "@/src/lib/http";
import {
  copilotAskRequestSchema,
  copilotAskResponseSchema
} from "@/src/features/copilot/schemas/copilotSchemas";

const ANSWERS: Record<string, string> = {
  why_estimate:
    "The estimate is most influenced by interior size and local baseline pricing, with downward pressure from building age.",
  what_changed_recently:
    "Recent monitoring shows stable drift status and no newly approved proposal winner against champion.",
  improve_confidence:
    "Provide complete and current property details and verify values for key structural features to improve confidence."
};

export async function POST(req: Request) {
  const payload = await req.json().catch(() => null);
  const parsed = copilotAskRequestSchema.safeParse(payload);
  if (!parsed.success) {
    return errorJson(`Invalid payload: ${parsed.error.issues[0]?.message ?? "unknown error"}`);
  }

  const response = copilotAskResponseSchema.parse({
    answer: ANSWERS[parsed.data.intent],
    citations: [
      "models/metrics_v1.json",
      "reports/arena/run_card_34e917e198af4e58adb2097b8d9ca229.md"
    ],
    assistant_confidence: 0.83,
    limitations: ["No fresh external listing feed is included in this scaffold response."],
    safety: {
      guardrail_mode: "strict_grounded",
      fallback_used: false
    }
  });

  return okJson(response);
}
