import type { CopilotContextPack } from "@/src/bff/copilot/contextPacks";
import type { RoutingDecision } from "@/src/bff/copilot/intentRouter";
import { buildCopilotFallback, buildGroundedCopilotAnswer } from "@/src/bff/mappers/copilotMapper";

export function buildCopilotResponse(input: {
  decision: RoutingDecision;
  contextPack: CopilotContextPack;
  simulateUnavailable: boolean;
  safetyBlockedReason?: "missing_context_artifacts" | "budget_exceeded" | "safety_blocked" | "upstream_timeout" | "intent_unresolved";
  promptInjectionBlocked?: boolean;
}) {
  const citations = input.contextPack.evidenceRefs
    .filter((ref) => ref.available)
    .map((ref) => ref.path);

  if (input.safetyBlockedReason) {
    return buildCopilotFallback({
      reason: input.safetyBlockedReason,
      promptInjectionBlocked: input.promptInjectionBlocked,
      citations
    });
  }

  if (input.simulateUnavailable) {
    return buildCopilotFallback({
      reason: "upstream_timeout",
      citations
    });
  }

  if (input.decision.lowConfidence) {
    return buildCopilotFallback({
      reason: "intent_unresolved",
      citations
    });
  }

  if (input.contextPack.missingArtifactKeys.length > 0 || citations.length === 0) {
    return buildCopilotFallback({
      reason: "missing_context_artifacts",
      citations
    });
  }

  return buildGroundedCopilotAnswer({
    intent: input.decision.resolvedIntent,
    summary: input.contextPack.summary,
    citations,
    lowConfidenceRouting: input.decision.lowConfidence
  });
}
