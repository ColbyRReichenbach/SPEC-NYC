import crypto from "node:crypto";

const INJECTION_PATTERNS = [
  /ignore\s+previous\s+instructions/i,
  /reveal\s+(system|hidden)\s+prompt/i,
  /<system>/i,
  /developer\s+message/i,
  /tool\s+call/i
];

const MAX_CHARS = 5_500;
const TOKEN_ESTIMATE_DIVISOR = 4;
const TOKEN_BUDGET = 1_600;

export type SafetyEvaluation = {
  sanitizedQuestion: string;
  blocked: boolean;
  reason?: "safety_blocked" | "budget_exceeded";
  promptInjectionBlocked: boolean;
  tokenBudgetExceeded: boolean;
  promptHash: string;
};

function sanitizeQuestion(question: string): string {
  return question.replace(/\s+/g, " ").trim().slice(0, MAX_CHARS);
}

function looksLikeInjection(question: string): boolean {
  return INJECTION_PATTERNS.some((pattern) => pattern.test(question));
}

function exceedsBudget(question: string): boolean {
  const estimatedTokens = Math.ceil(question.length / TOKEN_ESTIMATE_DIVISOR);
  return estimatedTokens > TOKEN_BUDGET;
}

export function evaluateCopilotSafety(question: string): SafetyEvaluation {
  const sanitizedQuestion = sanitizeQuestion(question);
  const promptInjectionBlocked = looksLikeInjection(sanitizedQuestion);
  const tokenBudgetExceeded = exceedsBudget(question) || exceedsBudget(sanitizedQuestion);

  if (promptInjectionBlocked) {
    return {
      sanitizedQuestion,
      blocked: true,
      reason: "safety_blocked",
      promptInjectionBlocked: true,
      tokenBudgetExceeded,
      promptHash: crypto.createHash("sha256").update(sanitizedQuestion).digest("hex").slice(0, 16)
    };
  }

  if (tokenBudgetExceeded) {
    return {
      sanitizedQuestion,
      blocked: true,
      reason: "budget_exceeded",
      promptInjectionBlocked: false,
      tokenBudgetExceeded: true,
      promptHash: crypto.createHash("sha256").update(sanitizedQuestion).digest("hex").slice(0, 16)
    };
  }

  return {
    sanitizedQuestion,
    blocked: false,
    promptInjectionBlocked: false,
    tokenBudgetExceeded: false,
    promptHash: crypto.createHash("sha256").update(sanitizedQuestion).digest("hex").slice(0, 16)
  };
}

export function recordCopilotAudit(input: {
  promptHash: string;
  reason?: string;
  page: string;
  intentRequested: string;
  intentResolved: string;
  fallbackUsed: boolean;
}): void {
  // v1 structured audit log; compatible with later backend audit sink integration.
  console.info(
    JSON.stringify({
      event: "copilot_audit",
      ts: new Date().toISOString(),
      prompt_hash: input.promptHash,
      reason: input.reason ?? null,
      page: input.page,
      intent_requested: input.intentRequested,
      intent_resolved: input.intentResolved,
      fallback_used: input.fallbackUsed
    })
  );
}
