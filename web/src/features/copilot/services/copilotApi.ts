import {
  copilotAskRequestSchema,
  copilotAskResponseSchema,
  type CopilotAskRequest,
  type CopilotAskResponse
} from "@/src/features/copilot/schemas/copilotSchemas";

const REQUEST_TIMEOUT_MS = 16_000;

export async function askCopilot(payload: CopilotAskRequest): Promise<CopilotAskResponse> {
  const parsedPayload = copilotAskRequestSchema.parse(payload);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch("/api/v1/copilot/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(parsedPayload),
      signal: controller.signal
    });

    if (!response.ok) {
      throw new Error(`Copilot request failed with status ${response.status}`);
    }

    const data = await response.json();
    return copilotAskResponseSchema.parse(data);
  } finally {
    clearTimeout(timeout);
  }
}
