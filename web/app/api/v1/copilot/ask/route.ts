import { errorJson, okJson } from "@/src/lib/http";
import { buildCopilotContextPack } from "@/src/bff/copilot/contextPacks";
import { generateGroundedCopilotDraft } from "@/src/bff/copilot/llmProvider";
import { routeCopilotIntent } from "@/src/bff/copilot/intentRouter";
import { buildCopilotResponse } from "@/src/bff/copilot/responseBuilder";
import { appendSessionTurn, getSessionTurns } from "@/src/bff/copilot/sessionMemory";
import { evaluateCopilotSafety, recordCopilotAudit } from "@/src/bff/copilot/safetyBridge";
import { emitCopilotTelemetry } from "@/src/bff/copilot/telemetry";
import {
  copilotAskRequestSchema,
  copilotAskResponseSchema,
  type CopilotPage
} from "@/src/features/copilot/schemas/copilotSchemas";

export async function POST(req: Request) {
  const startedAt = Date.now();
  const payload = await req.json().catch(() => null);
  const parsedRequest = copilotAskRequestSchema.safeParse(payload);
  if (!parsedRequest.success) {
    return errorJson(`Invalid payload: ${parsedRequest.error.issues[0]?.message ?? "unknown error"}`);
  }

  const page: CopilotPage = parsedRequest.data.context.page ?? "global";
  const decision = routeCopilotIntent(parsedRequest.data);
  const sessionId = parsedRequest.data.context.session_id ?? "anon";
  const sessionKey = `${page}:${sessionId}`;
  const history = getSessionTurns(sessionKey);

  const safetyEval = evaluateCopilotSafety(parsedRequest.data.question);
  const contextPack = await buildCopilotContextPack(decision, page);

  const responsePayload = buildCopilotResponse({
    decision,
    contextPack,
    simulateUnavailable: parsedRequest.data.context.simulate_unavailable === true,
    safetyBlockedReason: safetyEval.blocked ? safetyEval.reason : undefined,
    promptInjectionBlocked: safetyEval.promptInjectionBlocked
  });

  // Citations are mandatory for non-fallback answers.
  const citationsForResponse = responsePayload.citations.filter(Boolean);
  const fallbackForMissingCitations =
    !responsePayload.safety.fallback_used && citationsForResponse.length === 0;

  const finalPayload = fallbackForMissingCitations
    ? buildCopilotResponse({
        decision: { ...decision, lowConfidence: true },
        contextPack,
        simulateUnavailable: false,
        safetyBlockedReason: "missing_context_artifacts",
        promptInjectionBlocked: false
      })
    : responsePayload;

  let llmPayload = finalPayload;
  if (!llmPayload.safety.fallback_used) {
    const llmDraft = await generateGroundedCopilotDraft({
      question: parsedRequest.data.question,
      intent: llmPayload.intentResolved,
      page,
      contextPack
    });
    if (llmDraft) {
      llmPayload = {
        ...llmPayload,
        answer: llmDraft.answer,
        limitations: llmDraft.limitations.length ? llmDraft.limitations : llmPayload.limitations,
        actions: llmDraft.actions.length ? llmDraft.actions : llmPayload.actions,
        assistantConfidence: llmDraft.assistant_confidence,
        safety: {
          ...llmPayload.safety,
          guardrail_mode: "strict_grounded_llm"
        }
      };
    }
  }

  appendSessionTurn(sessionKey, {
    ts: Date.now(),
    question: parsedRequest.data.question,
    answer: llmPayload.answer,
    intent: llmPayload.intentResolved
  });

  recordCopilotAudit({
    promptHash: safetyEval.promptHash,
    reason: llmPayload.safety.reason,
    page,
    intentRequested: decision.requestedIntent,
    intentResolved: llmPayload.intentResolved,
    fallbackUsed: llmPayload.safety.fallback_used
  });

  const latencyMs = Date.now() - startedAt;
  emitCopilotTelemetry({
    page,
    intent_requested: decision.requestedIntent,
    intent_resolved: llmPayload.intentResolved,
    router_confidence: decision.confidence,
    latency_ms: latencyMs,
    fallback_used: finalPayload.safety.fallback_used,
    fallback_reason: finalPayload.safety.reason,
    guardrail_mode: finalPayload.safety.guardrail_mode,
    citation_count: finalPayload.citations.length,
    out_of_scope: finalPayload.intentResolved === "unknown"
  });

  const parsedResponse = copilotAskResponseSchema.parse({
    intent_resolved: finalPayload.intentResolved,
    router_confidence: decision.confidence,
    answer: llmPayload.answer,
    citations: llmPayload.citations,
    assistant_confidence: llmPayload.assistantConfidence,
    limitations: history.length
      ? [...llmPayload.limitations, `Session context considered from ${history.length} prior turn(s).`]
      : llmPayload.limitations,
    actions: llmPayload.actions,
    safety: llmPayload.safety,
    trace: parsedRequest.data.context.debug_trace
      ? {
          intent_requested: decision.requestedIntent,
          context_bundle_ids: contextPack.contextBundleIds,
          missing_artifact_keys: contextPack.missingArtifactKeys,
          token_budget_exceeded: safetyEval.tokenBudgetExceeded
        }
      : undefined,
    contract_version: "v1",
    generated_at: new Date().toISOString(),
    request_id: crypto.randomUUID(),
    source_context: llmPayload.sourceContext
  });

  const {
    contract_version: _cv,
    generated_at: _ga,
    request_id: _rid,
    source_context,
    ...responseBody
  } = parsedResponse;

  return okJson(responseBody, 200, { source_context });
}
