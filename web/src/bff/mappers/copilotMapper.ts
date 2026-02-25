import type { SourceContext } from "@/src/bff/types/baseContracts";
import type { CopilotIntentResolved } from "@/src/features/copilot/schemas/copilotSchemas";

export type CopilotAction = {
  title: string;
  owner: string;
  command?: string;
  artifact?: string;
  priority: "p0" | "p1" | "p2";
};

export function buildCopilotFallback(params: {
  reason:
    | "missing_context_artifacts"
    | "budget_exceeded"
    | "safety_blocked"
    | "upstream_timeout"
    | "intent_unresolved";
  promptInjectionBlocked?: boolean;
  citations?: string[];
}): {
  intentResolved: CopilotIntentResolved;
  answer: string;
  citations: string[];
  assistantConfidence: number;
  limitations: string[];
  actions: CopilotAction[];
  safety: {
    guardrail_mode: string;
    fallback_used: boolean;
    reason: string;
    prompt_injection_blocked?: boolean;
  };
  sourceContext: SourceContext;
} {
  return {
    intentResolved: "unknown",
    answer: "AI copilot unavailable; showing evidence summary from latest artifacts.",
    citations: params.citations?.length ? params.citations : ["models/metrics_v1.json"],
    assistantConfidence: 0.4,
    limitations: ["Generated from deterministic fallback template."],
    actions: [
      {
        title: "Verify context artifacts are present and fresh",
        owner: "MLOps",
        artifact: "reports/monitoring/performance_latest.json",
        priority: "p1"
      }
    ],
    safety: {
      guardrail_mode: "fallback_template",
      fallback_used: true,
      reason: params.reason,
      prompt_injection_blocked: params.promptInjectionBlocked
    },
    sourceContext: {
      source_id: "models/metrics_v1.json|reports/arena/proposal_*.json|reports/monitoring/performance_latest.json",
      source_type: "other"
    }
  };
}

export function buildGroundedCopilotAnswer(input: {
  intent: CopilotIntentResolved;
  summary: string[];
  citations: string[];
  lowConfidenceRouting: boolean;
}): {
  intentResolved: CopilotIntentResolved;
  answer: string;
  citations: string[];
  assistantConfidence: number;
  limitations: string[];
  actions: CopilotAction[];
  safety: {
    guardrail_mode: string;
    fallback_used: boolean;
    reason?: string;
    prompt_injection_blocked?: boolean;
  };
  sourceContext: SourceContext;
} {
  const baseContext = input.summary.join(" ");
  const citations = input.citations;

  if (input.intent === "why_estimate") {
    return {
      intentResolved: input.intent,
      answer:
        `Estimate movement is primarily explained by mapped feature drivers and route context. ${baseContext}`.trim(),
      citations,
      assistantConfidence: input.lowConfidenceRouting ? 0.62 : 0.84,
      limitations: ["Grounded to available valuation and monitoring artifacts only."],
      actions: [
        {
          title: "Validate selected property feature completeness before final decision",
          owner: "Data",
          priority: "p1"
        }
      ],
      safety: { guardrail_mode: "strict_grounded", fallback_used: false },
      sourceContext: {
        source_id: citations.join("|"),
        source_type: "other"
      }
    };
  }

  if (input.intent === "what_changed_recently") {
    return {
      intentResolved: input.intent,
      answer: `Recent artifact state summary: ${baseContext}`,
      citations,
      assistantConfidence: input.lowConfidenceRouting ? 0.58 : 0.81,
      limitations: ["Change summary does not include external market feeds unless present in artifacts."],
      actions: [
        {
          title: "Review drift/performance deltas for largest affected slices",
          owner: "ML",
          artifact: "reports/monitoring/performance_latest.json",
          priority: "p0"
        }
      ],
      safety: { guardrail_mode: "strict_grounded", fallback_used: false },
      sourceContext: {
        source_id: citations.join("|"),
        source_type: "other"
      }
    };
  }

  if (input.intent === "promotion_status") {
    return {
      intentResolved: input.intent,
      answer: `Promotion readiness is determined by policy gates and proposal status. ${baseContext}`,
      citations,
      assistantConfidence: input.lowConfidenceRouting ? 0.57 : 0.8,
      limitations: ["No promotion recommendation is made without complete gate evidence."],
      actions: [
        {
          title: "Resolve failing policy gates before proposing promotion",
          owner: "Release",
          command: "python3 -m src.validate_release --mode smoke --contract-profile canonical",
          priority: "p0"
        }
      ],
      safety: { guardrail_mode: "strict_grounded", fallback_used: false },
      sourceContext: {
        source_id: citations.join("|"),
        source_type: "other"
      }
    };
  }

  if (input.intent === "monitoring_remediation") {
    return {
      intentResolved: input.intent,
      answer: `Monitoring remediation should prioritize data and segment stability issues surfaced by artifacts. ${baseContext}`,
      citations,
      assistantConfidence: input.lowConfidenceRouting ? 0.56 : 0.78,
      limitations: ["Remediation actions are operator guidance and must be validated in release checks."],
      actions: [
        {
          title: "Run release smoke and inspect failing gates",
          owner: "MLOps",
          command: "python3 -m src.validate_release --mode smoke --contract-profile canonical",
          priority: "p0"
        },
        {
          title: "Rebuild context artifacts if missing",
          owner: "Data",
          priority: "p1"
        }
      ],
      safety: { guardrail_mode: "strict_grounded", fallback_used: false },
      sourceContext: {
        source_id: citations.join("|"),
        source_type: "other"
      }
    };
  }

  return {
    intentResolved: "improve_confidence",
    answer:
      `Confidence can be improved by tightening feature completeness and comparable coverage. ${baseContext}`.trim(),
    citations,
    assistantConfidence: input.lowConfidenceRouting ? 0.55 : 0.79,
    limitations: ["Improvement guidance is heuristic and should be validated with new model runs."],
    actions: [
      {
        title: "Prioritize missing structural features for targeted records",
        owner: "Data",
        priority: "p1"
      }
    ],
    safety: { guardrail_mode: "strict_grounded", fallback_used: false },
    sourceContext: {
      source_id: citations.join("|"),
      source_type: "other"
    }
  };
}
