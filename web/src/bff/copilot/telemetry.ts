export function emitCopilotTelemetry(event: {
  page: string;
  intent_requested: string;
  intent_resolved: string;
  router_confidence: number;
  latency_ms: number;
  fallback_used: boolean;
  fallback_reason?: string;
  guardrail_mode: string;
  citation_count: number;
  out_of_scope: boolean;
}) {
  console.info(
    JSON.stringify({
      event: "copilot_telemetry",
      ts: new Date().toISOString(),
      ...event
    })
  );
}
