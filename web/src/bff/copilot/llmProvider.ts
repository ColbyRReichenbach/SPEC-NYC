import type { CopilotAction, CopilotIntentResolved } from "@/src/features/copilot/schemas/copilotSchemas";
import type { CopilotContextPack } from "@/src/bff/copilot/contextPacks";
import fs from "node:fs";
import path from "node:path";

type LlmDraft = {
  answer: string;
  limitations: string[];
  actions: CopilotAction[];
  assistant_confidence: number;
};

type LlmInput = {
  question: string;
  intent: CopilotIntentResolved;
  page: string;
  contextPack: CopilotContextPack;
};

let dotenvCache: Record<string, string> | null = null;

function parseDotEnv(content: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of content.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const idx = trimmed.indexOf("=");
    if (idx <= 0) continue;
    const key = trimmed.slice(0, idx).trim();
    const rawValue = trimmed.slice(idx + 1).trim();
    const value = rawValue.replace(/^["']|["']$/g, "");
    out[key] = value;
  }
  return out;
}

function getDotEnvCache(): Record<string, string> {
  if (dotenvCache) return dotenvCache;

  const candidates = [
    path.resolve(process.cwd(), ".env.local"),
    path.resolve(process.cwd(), ".env"),
    path.resolve(process.cwd(), "..", ".env.local"),
    path.resolve(process.cwd(), "..", ".env")
  ];

  const merged: Record<string, string> = {};
  for (const candidate of candidates) {
    try {
      if (!fs.existsSync(candidate)) continue;
      const content = fs.readFileSync(candidate, "utf-8");
      Object.assign(merged, parseDotEnv(content));
    } catch {
      // ignore parse/read failures
    }
  }
  dotenvCache = merged;
  return merged;
}

function getRuntimeEnv(key: string): string | undefined {
  const direct = process.env[key];
  if (direct && direct.length > 0) return direct;
  const cache = getDotEnvCache();
  return cache[key];
}

function buildPrompt(input: LlmInput): string {
  return [
    "You are an AVM copilot for internal ops.",
    "Rules:",
    "- Ground strictly on provided context.",
    "- Do not invent metrics, files, or policies.",
    "- No legal/lending/compliance determinations.",
    "- Keep concise and operational.",
    "",
    `Intent: ${input.intent}`,
    `Page: ${input.page}`,
    `Question: ${input.question}`,
    "Context summary:",
    ...input.contextPack.summary.map((line) => `- ${line}`),
    "Evidence paths:",
    ...input.contextPack.evidenceRefs.map((ref) => `- ${ref.path} (available=${ref.available})`),
    "",
    "Return STRICT JSON with keys:",
    "{",
    '  "answer": "string",',
    '  "limitations": ["string"],',
    '  "actions": [{"title":"string","owner":"string","command":"string?","artifact":"string?","priority":"p0|p1|p2"}],',
    '  "assistant_confidence": 0.0',
    "}"
  ].join("\n");
}

function parseJsonObject(text: string): Record<string, unknown> | null {
  const trimmed = text.trim();
  try {
    return JSON.parse(trimmed) as Record<string, unknown>;
  } catch {
    const match = trimmed.match(/\{[\s\S]*\}/);
    if (!match) return null;
    try {
      return JSON.parse(match[0]) as Record<string, unknown>;
    } catch {
      return null;
    }
  }
}

function normalizeDraft(raw: Record<string, unknown>): LlmDraft | null {
  if (typeof raw.answer !== "string" || raw.answer.trim().length < 3) return null;
  const limitations = Array.isArray(raw.limitations)
    ? raw.limitations.filter((item): item is string => typeof item === "string" && item.length > 0)
    : [];

  const actions = Array.isArray(raw.actions)
    ? raw.actions
        .map((item) => {
          if (!item || typeof item !== "object") return null;
          const rec = item as Record<string, unknown>;
          if (typeof rec.title !== "string" || typeof rec.owner !== "string") return null;
          const priority = rec.priority === "p0" || rec.priority === "p1" || rec.priority === "p2" ? rec.priority : "p1";
          return {
            title: rec.title,
            owner: rec.owner,
            command: typeof rec.command === "string" ? rec.command : undefined,
            artifact: typeof rec.artifact === "string" ? rec.artifact : undefined,
            priority
          } as CopilotAction;
        })
        .filter((item): item is CopilotAction => Boolean(item))
    : [];

  const confRaw = typeof raw.assistant_confidence === "number" ? raw.assistant_confidence : 0.72;
  const assistantConfidence = Math.min(0.95, Math.max(0.1, confRaw));

  return {
    answer: raw.answer,
    limitations,
    actions,
    assistant_confidence: assistantConfidence
  };
}

export async function generateGroundedCopilotDraft(input: LlmInput): Promise<LlmDraft | null> {
  const apiKey = getRuntimeEnv("OPENAI_API_KEY");
  if (!apiKey || apiKey === "your_openai_api_key_here") {
    console.info(JSON.stringify({ event: "copilot_llm_skip", reason: "missing_api_key" }));
    return null;
  }

  const model = getRuntimeEnv("OPENAI_MODEL") || "gpt-4o-mini";
  const maxTokensRaw = Number(getRuntimeEnv("AI_MAX_OUTPUT_TOKENS") ?? "700");
  const maxTokens = Number.isFinite(maxTokensRaw) ? Math.min(1200, Math.max(200, Math.floor(maxTokensRaw))) : 700;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12000);

  try {
    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`
      },
      signal: controller.signal,
      body: JSON.stringify({
        model,
        temperature: 0.2,
        max_tokens: maxTokens,
        messages: [
          { role: "system", content: "Return only valid JSON. No markdown." },
          { role: "user", content: buildPrompt(input) }
        ]
      })
    });

    if (!response.ok) {
      const body = await response.text().catch(() => "");
      let errorType = "unknown";
      try {
        const parsed = JSON.parse(body) as { error?: { type?: string } };
        errorType = parsed.error?.type ?? "unknown";
      } catch {
        // ignore parse failure
      }
      console.warn(
        JSON.stringify({
          event: "copilot_llm_error",
          reason: "non_ok_status",
          status: response.status,
          error_type: errorType
        })
      );
      return null;
    }
    const body = (await response.json()) as {
      choices?: Array<{ message?: { content?: string } }>;
    };
    const content = body.choices?.[0]?.message?.content;
    if (!content) {
      console.warn(JSON.stringify({ event: "copilot_llm_error", reason: "empty_content" }));
      return null;
    }
    const parsed = parseJsonObject(content);
    if (!parsed) {
      console.warn(JSON.stringify({ event: "copilot_llm_error", reason: "invalid_json_response" }));
      return null;
    }
    const normalized = normalizeDraft(parsed);
    if (!normalized) {
      console.warn(JSON.stringify({ event: "copilot_llm_error", reason: "schema_normalization_failed" }));
      return null;
    }
    console.info(JSON.stringify({ event: "copilot_llm_used", model }));
    return normalized;
  } catch (error) {
    console.warn(
      JSON.stringify({
        event: "copilot_llm_error",
        reason: "request_exception",
        message: error instanceof Error ? error.message : "unknown_error"
      })
    );
    return null;
  } finally {
    clearTimeout(timeout);
  }
}
