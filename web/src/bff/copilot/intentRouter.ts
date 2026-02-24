import type {
  CopilotAskRequest,
  CopilotIntentRequested,
  CopilotIntentResolved,
  CopilotPage
} from "@/src/features/copilot/schemas/copilotSchemas";

export type ExtractedEntities = {
  property_id?: string;
  valuation_id?: string;
  proposal_id?: string;
  window?: "7d" | "30d" | "90d" | "180d";
  segment?: string;
  tier?: string;
};

export type RoutingDecision = {
  requestedIntent: CopilotIntentRequested;
  resolvedIntent: CopilotIntentResolved;
  confidence: number;
  entities: ExtractedEntities;
  downgradedForPage: boolean;
  lowConfidence: boolean;
};

const SUPPORTED_FOR_PAGE: Record<CopilotPage, CopilotIntentResolved[]> = {
  valuation: ["why_estimate", "improve_confidence", "what_changed_recently", "unknown"],
  governance: ["what_changed_recently", "promotion_status", "monitoring_remediation", "unknown"],
  monitoring: ["what_changed_recently", "monitoring_remediation", "improve_confidence", "unknown"],
  global: [
    "why_estimate",
    "what_changed_recently",
    "improve_confidence",
    "promotion_status",
    "monitoring_remediation",
    "unknown"
  ]
};

function scoreIntent(question: string): Record<CopilotIntentResolved, number> {
  const text = question.toLowerCase();
  const score: Record<CopilotIntentResolved, number> = {
    why_estimate: 0.2,
    what_changed_recently: 0.2,
    improve_confidence: 0.2,
    promotion_status: 0.2,
    monitoring_remediation: 0.2,
    unknown: 0.1
  };

  if (/why|because|driver|estimate|valuation|price/i.test(text)) score.why_estimate += 0.6;
  if (/changed|recent|last|trend|drift|month|week/i.test(text)) score.what_changed_recently += 0.55;
  if (/improve|confidence|uncertain|uncertainty|better data|increase/i.test(text)) score.improve_confidence += 0.58;
  if (/promote|promotion|approve|reject|gate|policy|winner|champion/i.test(text)) score.promotion_status += 0.62;
  if (/remediate|fix|ops|monitoring|alert|retrain|incident|action/i.test(text)) {
    score.monitoring_remediation += 0.62;
  }
  if (/is this ok|help|what now|unclear|not sure/i.test(text)) score.unknown += 0.4;

  return score;
}

function extractWindow(question: string): ExtractedEntities["window"] | undefined {
  const match = question.toLowerCase().match(/\b(7|30|90|180)d\b/);
  if (!match) return undefined;
  const raw = `${match[1]}d`;
  return raw === "7d" || raw === "30d" || raw === "90d" || raw === "180d" ? raw : undefined;
}

export function routeCopilotIntent(request: CopilotAskRequest): RoutingDecision {
  const page = request.context.page ?? "global";
  const requestedIntent = request.intent ?? "auto";

  const entities: ExtractedEntities = {
    property_id: request.context.property_id,
    valuation_id: request.context.valuation_id ?? request.question.match(/\bval_[a-z0-9_]+\b/i)?.[0],
    proposal_id: request.context.proposal_id ?? request.question.match(/\bproposal_[a-z0-9_]+\b/i)?.[0],
    window: request.context.window ?? extractWindow(request.question),
    segment: request.context.segment,
    tier: request.context.tier
  };

  let resolvedIntent: CopilotIntentResolved;
  let confidence = 0.82;
  let lowConfidence = false;

  if (requestedIntent !== "auto") {
    resolvedIntent = requestedIntent;
  } else {
    const scores = scoreIntent(request.question);
    const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
    resolvedIntent = (ranked[0]?.[0] as CopilotIntentResolved | undefined) ?? "unknown";
    confidence = Math.min(0.95, Math.max(0.2, ranked[0]?.[1] ?? 0.25));
    lowConfidence = confidence < 0.45 || resolvedIntent === "unknown";
  }

  const allowedIntents = SUPPORTED_FOR_PAGE[page];
  let downgradedForPage = false;
  if (!allowedIntents.includes(resolvedIntent)) {
    downgradedForPage = true;
    if (page === "valuation") {
      resolvedIntent = "why_estimate";
      confidence = Math.min(confidence, 0.6);
    } else if (page === "governance") {
      resolvedIntent = "promotion_status";
      confidence = Math.min(confidence, 0.6);
    } else if (page === "monitoring") {
      resolvedIntent = "monitoring_remediation";
      confidence = Math.min(confidence, 0.6);
    } else {
      resolvedIntent = "unknown";
      confidence = Math.min(confidence, 0.5);
    }
  }

  return {
    requestedIntent,
    resolvedIntent,
    confidence: Number(confidence.toFixed(2)),
    entities,
    downgradedForPage,
    lowConfidence
  };
}
