"use client";

import { useMemo, useState } from "react";
import { usePathname } from "next/navigation";

import { askCopilot } from "@/src/features/copilot/services/copilotApi";
import type {
  CopilotAction,
  CopilotIntentRequested,
  CopilotPage
} from "@/src/features/copilot/schemas/copilotSchemas";

const INTENT_LABELS: Record<Exclude<CopilotIntentRequested, "auto">, string> = {
  why_estimate: "Why this estimate?",
  what_changed_recently: "What changed recently?",
  improve_confidence: "What improves confidence?",
  promotion_status: "Can we promote now?",
  monitoring_remediation: "What should ops do now?"
};

function inferPage(pathname: string): CopilotPage {
  if (pathname.includes("/valuation/")) return "valuation";
  if (pathname.includes("/governance")) return "governance";
  if (pathname.includes("/monitoring")) return "monitoring";
  return "global";
}

function getSessionId(): string {
  if (typeof window === "undefined") return "ssr";
  const key = "spec_copilot_session_id";
  const existing = window.localStorage.getItem(key);
  if (existing) return existing;
  const created = `sess_${Math.random().toString(36).slice(2, 10)}`;
  window.localStorage.setItem(key, created);
  return created;
}

export default function CopilotDrawer() {
  const pathname = usePathname();
  const page = inferPage(pathname);

  const [open, setOpen] = useState(false);
  const [pending, setPending] = useState(false);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<string>("Ask Copilot for grounded guidance with artifact citations.");
  const [citations, setCitations] = useState<string[]>([]);
  const [limitations, setLimitations] = useState<string[]>([]);
  const [actions, setActions] = useState<CopilotAction[]>([]);
  const [safetyMode, setSafetyMode] = useState("strict_grounded");
  const [responseSource, setResponseSource] = useState<"llm" | "deterministic" | "fallback">(
    "deterministic"
  );
  const [intentResolved, setIntentResolved] = useState<string>("-");
  const [routerConfidence, setRouterConfidence] = useState<number | null>(null);

  const defaultIntent = useMemo<Exclude<CopilotIntentRequested, "auto">>(() => {
    if (page === "governance") return "promotion_status";
    if (page === "monitoring") return "monitoring_remediation";
    return "why_estimate";
  }, [page]);

  async function submit(intent: CopilotIntentRequested, promptText: string) {
    const trimmed = promptText.trim();
    if (!trimmed) return;

    try {
      setPending(true);
      const response = await askCopilot({
        question: trimmed,
        intent,
        context: {
          page,
          window: "30d",
          session_id: getSessionId()
        }
      });

      setAnswer(response.answer);
      setCitations(response.citations);
      setLimitations(response.limitations);
      setActions(response.actions ?? []);
      setSafetyMode(response.safety.guardrail_mode);
      if (response.safety.fallback_used) {
        setResponseSource("fallback");
      } else if (response.safety.guardrail_mode === "strict_grounded_llm") {
        setResponseSource("llm");
      } else {
        setResponseSource("deterministic");
      }
      setIntentResolved(response.intent_resolved);
      setRouterConfidence(response.router_confidence);
      setQuestion("");
    } catch (error) {
      setAnswer(`Copilot request failed: ${error instanceof Error ? error.message : "unknown error"}`);
      setCitations([]);
      setLimitations(["Artifact-grounded Copilot evidence is unavailable while connection issues are resolved."]);
      setActions([]);
      setSafetyMode("fallback_template");
      setResponseSource("fallback");
      setIntentResolved("unknown");
      setRouterConfidence(null);
    } finally {
      setPending(false);
    }
  }

  return (
    <>
      <button
        className="copilot-trigger"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
        aria-controls="copilot-drawer"
      >
        {open ? "Close Copilot" : "Open Copilot"}
      </button>
      <aside id="copilot-drawer" className={`copilot-drawer ${open ? "open" : ""}`} aria-hidden={!open}>
        <h3>AI Copilot</h3>
        <p className="muted">Contextual, citation-grounded assistance for this page.</p>

        <label className="context-label" htmlFor="copilot-question">Ask a question</label>
        <textarea
          id="copilot-question"
          rows={3}
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Type your question..."
          disabled={pending}
        />
        <div className="button-row">
          <button
            className="primary-btn"
            disabled={pending || question.trim().length < 3}
            onClick={() => submit("auto", question)}
          >
            {pending ? "Thinking..." : "Send"}
          </button>
        </div>

        <div className="chip-row" role="group" aria-label="Copilot quick intents">
          <button disabled={pending} onClick={() => submit(defaultIntent, INTENT_LABELS[defaultIntent])}>
            {INTENT_LABELS[defaultIntent]}
          </button>
          <button disabled={pending} onClick={() => submit("what_changed_recently", INTENT_LABELS.what_changed_recently)}>
            {INTENT_LABELS.what_changed_recently}
          </button>
          <button disabled={pending} onClick={() => submit("improve_confidence", INTENT_LABELS.improve_confidence)}>
            {INTENT_LABELS.improve_confidence}
          </button>
        </div>

        <div className="copilot-answer card" aria-busy={pending}>
          <h4>Answer</h4>
          <p>{answer}</p>
          <p className="muted">
            Intent: <code>{intentResolved}</code> · Router confidence:{" "}
            <code>{routerConfidence === null ? "-" : routerConfidence.toFixed(2)}</code>
          </p>
          <p className="muted">
            Safety mode: <code>{safetyMode}</code>
          </p>
          <p className="muted">
            Response source:{" "}
            <span className={`copilot-source-badge ${responseSource}`}>
              {responseSource === "llm"
                ? "LLM"
                : responseSource === "fallback"
                  ? "Fallback Template"
                  : "Deterministic Grounded"}
            </span>
          </p>

          {actions.length > 0 ? (
            <>
              <h5>Actions</h5>
              <ul>
                {actions.map((item) => (
                  <li key={`${item.title}:${item.priority}`}>
                    <strong>{item.title}</strong> ({item.priority}, owner: {item.owner})
                    {item.command ? <div><code>{item.command}</code></div> : null}
                    {item.artifact ? <div><code>{item.artifact}</code></div> : null}
                  </li>
                ))}
              </ul>
            </>
          ) : null}

          {citations.length > 0 ? (
            <>
              <h5>Citations</h5>
              <ul>
                {citations.map((citation) => (
                  <li key={citation}>
                    <code>{citation}</code>
                  </li>
                ))}
              </ul>
            </>
          ) : null}

          {limitations.length > 0 ? (
            <>
              <h5>Limitations</h5>
              <ul>
                {limitations.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </>
          ) : null}
        </div>
      </aside>
    </>
  );
}
