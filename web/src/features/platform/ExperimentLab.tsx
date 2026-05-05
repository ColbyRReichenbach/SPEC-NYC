"use client";

import { useCallback, useEffect, useState, type FormEvent } from "react";
import {
  AlertCircle,
  Beaker,
  CheckCircle2,
  Clock3,
  FileJson,
  FlaskConical,
  GitCompareArrows,
  LockKeyhole,
  Play,
  ShieldCheck
} from "lucide-react";

import type { PlatformData } from "@/src/features/platform/data";
import type { ExperimentRunBundle } from "@/src/features/platform/experimentRegistry";

type ExperimentForm = {
  hypothesis: string;
  expectedEffect: string;
  segment: string;
  primaryMetric: string;
  modelFamily: string;
  validationPlan: string;
  trialBudget: number;
  riskReview: boolean;
};

const initialForm: ExperimentForm = {
  hypothesis: "Train-fitted H3 price lag will reduce MdAPE for Brooklyn small multifamily properties.",
  expectedEffect: "Lower MdAPE by 3-5% without worsening PPE10.",
  segment: "SMALL_MULTI",
  primaryMetric: "MdAPE",
  modelFamily: "XGBoost segment specialist",
  validationPlan: "Time split + borough/segment slices",
  trialBudget: 12,
  riskReview: true
};

const presets = [
  {
    label: "Geo lag ablation",
    hypothesis: "Removing H3 lag features will expose whether spatial memory is carrying predictive lift.",
    expectedEffect: "Quantify MdAPE degradation and feature reliance.",
    segment: "ALL",
    modelFamily: "Ablation study",
    trialBudget: 4
  },
  {
    label: "Segment specialist",
    hypothesis: "A segment-specific model can improve single-family PPE10 without masking condo error.",
    expectedEffect: "Improve single-family PPE10 while preserving global guardrails.",
    segment: "SINGLE_FAMILY",
    modelFamily: "XGBoost segment specialist",
    trialBudget: 16
  },
  {
    label: "Temporal robustness",
    hypothesis: "Explicit market regime features will stabilize late-window valuation residuals.",
    expectedEffect: "Reduce holdout drift and quarter-level MdAPE variance.",
    segment: "ALL",
    modelFamily: "Temporal residual model",
    trialBudget: 8
  }
];

export function ExperimentLabView({ data }: { data: PlatformData }) {
  const [form, setForm] = useState<ExperimentForm>(initialForm);
  const [experiments, setExperiments] = useState<ExperimentRunBundle[]>([]);
  const [activeExperiment, setActiveExperiment] = useState<ExperimentRunBundle | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "running" | "complete" | "error">("loading");
  const [error, setError] = useState<string | null>(null);
  const [actionStatus, setActionStatus] = useState<string | null>(null);

  const refreshExperiments = useCallback(async () => {
    const response = await fetch("/api/v1/experiments/registry");
    const payload = (await response.json()) as { experiments?: ExperimentRunBundle[] };
    const nextExperiments = payload.experiments ?? [];
    setExperiments(nextExperiments);
    setActiveExperiment((current) => {
      if (!current) {
        return null;
      }
      return nextExperiments.find((experiment) => experiment.id === current.id) ?? current;
    });
    setStatus("idle");
    return nextExperiments;
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    async function loadExperiments() {
      try {
        const response = await fetch("/api/v1/experiments/registry", { signal: controller.signal });
        const payload = (await response.json()) as { experiments?: ExperimentRunBundle[] };
        setExperiments(payload.experiments ?? []);
        setStatus("idle");
      } catch (requestError) {
        if (!controller.signal.aborted) {
          setError(requestError instanceof Error ? requestError.message : "Unable to load experiments.");
          setStatus("error");
        }
      }
    }

    void loadExperiments();
    return () => controller.abort();
  }, []);

  const reviewQueue = experiments.filter((experiment) => experiment.status === "review_requested");
  const experimentQueue = experiments.filter((experiment) => experiment.status === "queued" || experiment.status === "running");
  const trackedExperiments = experiments.filter((experiment) => experiment.status === "completed" || experiment.status === "failed");

  async function runExperiment(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setStatus("running");
    setError(null);

    let payload: { experiment?: ExperimentRunBundle; error?: string };
    let response: Response;

    try {
      response = await fetch("/api/v1/experiments/preflight", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form)
      });
      payload = (await response.json()) as { experiment?: ExperimentRunBundle; error?: string };
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Experiment preflight failed.");
      setStatus("error");
      return;
    }

    if (!response.ok || !payload.experiment) {
      setError(payload.error ?? "Experiment preflight failed.");
      setStatus("error");
      return;
    }

    setActiveExperiment(payload.experiment);
    setExperiments((current) => [payload.experiment as ExperimentRunBundle, ...current].slice(0, 12));
    setStatus("complete");
  }

  async function runExperimentAction({
    experiment,
    endpoint,
    body,
    successMessage
  }: {
    experiment: ExperimentRunBundle;
    endpoint: string;
    body?: Record<string, unknown>;
    successMessage: string;
  }) {
    setStatus("running");
    setActionStatus(null);
    setError(null);

    let response: Response;
    let payload: { experiment?: ExperimentRunBundle; error?: string; worker?: { pid?: number; command?: string } };

    try {
      response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body ?? {})
      });
      payload = (await response.json()) as { experiment?: ExperimentRunBundle; error?: string; worker?: { pid?: number; command?: string } };
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Experiment action failed.");
      setStatus("error");
      return;
    }

    if (!response.ok || !payload.experiment) {
      setError(payload.error ?? "Experiment action failed.");
      setStatus("error");
      return;
    }

    const updatedExperiment = payload.experiment;
    setExperiments((current) =>
      [updatedExperiment, ...current.filter((candidate) => candidate.id !== updatedExperiment.id)]
        .sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))
        .slice(0, 20)
    );
    setActiveExperiment(updatedExperiment);
    setActionStatus(payload.worker?.pid ? `${successMessage} Worker pid ${payload.worker.pid}.` : successMessage);
    setStatus("complete");

    if (payload.worker?.pid) {
      window.setTimeout(() => {
        void refreshExperiments();
      }, 1500);
    }
  }

  function applyPreset(preset: (typeof presets)[number]) {
    setForm((current) => ({
      ...current,
      hypothesis: preset.hypothesis,
      expectedEffect: preset.expectedEffect,
      segment: preset.segment,
      modelFamily: preset.modelFamily,
      trialBudget: preset.trialBudget
    }));
  }

  return (
    <div className="view-stack">
      <section className="experiment-hero">
        <div>
          <span className="eyebrow">Experiment Control Room</span>
          <h1>Hypothesis-led model development with auditable preflight runs</h1>
          <p>
            Log research intent, bind it to the current package, move it through review, queue a real trainer-backed
            job, and keep every decision tied to filesystem artifacts.
          </p>
        </div>
        <div className="experiment-hero-card">
          <span>Current baseline</span>
          <strong>{formatPercent(data.package.mdape)} MdAPE</strong>
          <code>{data.package.id}</code>
        </div>
      </section>

      <div className="experiment-layout">
        <form className="panel experiment-form" onSubmit={runExperiment}>
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Hypothesis</span>
              <h2>Run governed preflight</h2>
            </div>
            <span className="status-pill pass">
              <ShieldCheck size={14} aria-hidden="true" />
              audit logged
            </span>
          </div>

          <div className="preset-row" aria-label="Experiment presets">
            {presets.map((preset) => (
              <button key={preset.label} type="button" onClick={() => applyPreset(preset)}>
                <Beaker size={14} aria-hidden="true" />
                {preset.label}
              </button>
            ))}
          </div>

          <div className="locked-spec-grid" aria-label="Locked spec controls">
            <SpecControl label="Dataset rows" value="Locked snapshot" />
            <SpecControl label="Feature contract" value={data.package.featureContractVersion} />
            <SpecControl label="Split policy" value="time_ordered_split" />
            <SpecControl label="Arena rule" value="same rows" />
          </div>

          <label className="field-stack">
            Hypothesis
            <textarea
              value={form.hypothesis}
              onChange={(event) => setForm({ ...form, hypothesis: event.target.value })}
              spellCheck={false}
              suppressHydrationWarning
              rows={4}
            />
          </label>
          <label className="field-stack">
            Expected effect
            <input value={form.expectedEffect} onChange={(event) => setForm({ ...form, expectedEffect: event.target.value })} />
          </label>

          <div className="experiment-fields">
            <label className="field-stack">
              Segment
              <select value={form.segment} onChange={(event) => setForm({ ...form, segment: event.target.value })}>
                <option>ALL</option>
                <option>SINGLE_FAMILY</option>
                <option>SMALL_MULTI</option>
                <option>WALKUP</option>
                <option>ELEVATOR</option>
              </select>
            </label>
            <label className="field-stack">
              Primary metric
              <select value={form.primaryMetric} onChange={(event) => setForm({ ...form, primaryMetric: event.target.value })}>
                <option>MdAPE</option>
                <option>PPE10</option>
                <option>R2</option>
                <option>Slice parity</option>
              </select>
            </label>
            <label className="field-stack">
              Model family
              <select value={form.modelFamily} onChange={(event) => setForm({ ...form, modelFamily: event.target.value })}>
                <option>XGBoost segment specialist</option>
                <option>Global XGBoost baseline</option>
                <option>Ablation study</option>
                <option>Temporal residual model</option>
                <option>Neural tabular prototype</option>
              </select>
            </label>
            <label className="field-stack">
              Validation plan
              <select value={form.validationPlan} onChange={(event) => setForm({ ...form, validationPlan: event.target.value })}>
                <option>Time split + borough/segment slices</option>
                <option>Backtest by sale quarter</option>
                <option>Ablation against v2 baseline</option>
                <option>Stress test high-missingness slices</option>
              </select>
            </label>
          </div>

          <div className="experiment-run-row">
            <label className="field-stack compact">
              Trial budget
              <input
                type="number"
                min={1}
                max={100}
                value={form.trialBudget}
                onChange={(event) => setForm({ ...form, trialBudget: Number(event.target.value) })}
              />
            </label>
            <label className="toggle-row">
              <input
                type="checkbox"
                checked={form.riskReview}
                onChange={(event) => setForm({ ...form, riskReview: event.target.checked })}
              />
              <span>Require model-risk review before promotion</span>
            </label>
          </div>

          <div className="comparison-contract-callout">
            <GitCompareArrows size={17} aria-hidden="true" />
            <span>
              Challenger and champion comparisons are blocked unless both score the locked dataset snapshot and split
              signature generated by this preflight.
            </span>
          </div>

          <button className="command-button experiment-run-button" type="submit" disabled={status === "running"}>
            <Play size={16} aria-hidden="true" />
            {status === "running" ? "Working" : "Run Experiment Preflight"}
          </button>
          {error ? (
            <p className="form-error">
              <AlertCircle size={15} aria-hidden="true" />
              {error}
            </p>
          ) : null}
        </form>

        <section className="panel experiment-output">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Run Output</span>
              <h2>{activeExperiment ? activeExperiment.id : "No active run"}</h2>
            </div>
            <RunStatus status={status} />
          </div>
          {activeExperiment ? (
            <>
              <div className="manifest-card">
                <FileJson size={18} aria-hidden="true" />
                <div>
                  <span>Run manifest</span>
                  <code>{activeExperiment.manifest_path}</code>
                </div>
              </div>
              <div className="locked-artifact-grid">
                <ArtifactRef label="Spec hash" value={shortHash(activeExperiment.spec_hash)} />
                <ArtifactRef label="Run dir" value={activeExperiment.run_dir} />
                <ArtifactRef label="Dataset snapshot" value={activeExperiment.artifact_paths.dataset_snapshot} />
                <ArtifactRef label="Review decision" value={activeExperiment.artifact_paths.review_decision ?? "not_created"} />
                <ArtifactRef label="Job manifest" value={activeExperiment.artifact_paths.job_manifest ?? "not_created"} />
                <ArtifactRef label="Comparison report" value={activeExperiment.artifact_paths.comparison_report} />
              </div>
              <LifecycleRail experiment={activeExperiment} />
              <ExperimentActions
                experiment={activeExperiment}
                disabled={status === "running"}
                onRequestReview={(experiment) =>
                  runExperimentAction({
                    experiment,
                    endpoint: `/api/v1/experiments/${experiment.id}/review-request`,
                    body: { reason: "Spec is ready for human review." },
                    successMessage: "Review request written."
                  })
                }
                onApprove={(experiment) =>
                  runExperimentAction({
                    experiment,
                    endpoint: `/api/v1/experiments/${experiment.id}/review`,
                    body: {
                      decision: "approved",
                      reviewer: "local_reviewer",
                      reason: "Approved for controlled challenger training with locked data contract."
                    },
                    successMessage: "Review approval written."
                  })
                }
                onReject={(experiment) =>
                  runExperimentAction({
                    experiment,
                    endpoint: `/api/v1/experiments/${experiment.id}/review`,
                    body: {
                      decision: "rejected",
                      reviewer: "local_reviewer",
                      reason: "Rejected from dashboard review."
                    },
                    successMessage: "Review rejection written."
                  })
                }
                onQueue={(experiment) =>
                  runExperimentAction({
                    experiment,
                    endpoint: `/api/v1/experiments/${experiment.id}/queue`,
                    body: { reason: "Approved experiment queued from dashboard." },
                    successMessage: "Training job queued."
                  })
                }
                onStartWorker={(experiment) =>
                  runExperimentAction({
                    experiment,
                    endpoint: `/api/v1/experiments/${experiment.id}/worker`,
                    successMessage: "Worker started."
                  })
                }
                onRefresh={() => {
                  setActionStatus("Registry refreshed.");
                  void refreshExperiments();
                }}
              />
              {actionStatus ? <p className="form-success">{actionStatus}</p> : null}
              <div className="gate-stack">
                {activeExperiment.gates.map((gate) => (
                  <div key={gate.name}>
                    <span className={`status-dot ${gate.status}`} />
                    <strong>{gate.name}</strong>
                    <p>{gate.detail}</p>
                  </div>
                ))}
              </div>
              <pre className="manifest-preview">{JSON.stringify(activeExperiment, null, 2)}</pre>
            </>
          ) : (
            <div className="empty-state">
              <FlaskConical size={24} aria-hidden="true" />
              <strong>Preflight artifacts will appear here.</strong>
              <p>The app records run intent and gate state before model training can be promoted.</p>
            </div>
          )}
        </section>
      </div>

      <section className="queue-board" aria-label="Governed experiment queues">
        <QueueColumn title="Review Queue" experiments={reviewQueue} emptyLabel="No specs waiting for review." onSelect={setActiveExperiment} />
        <QueueColumn title="Experiment Queue" experiments={experimentQueue} emptyLabel="No training jobs queued." onSelect={setActiveExperiment} />
        <QueueColumn title="Tracked Experiments" experiments={trackedExperiments} emptyLabel="No completed worker runs yet." onSelect={setActiveExperiment} />
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Previous Runs</span>
            <h2>{experiments.length} locked experiment specs</h2>
          </div>
        </div>
        <div className="experiment-log-grid">
          {experiments.length === 0 ? (
            <p className="quiet">No logged experiment manifests yet.</p>
          ) : (
            experiments.map((experiment) => (
              <article key={experiment.id} className="experiment-run-card">
                <div>
                  <Clock3 size={15} aria-hidden="true" />
                  <time>{formatTimestamp(experiment.created_at)}</time>
                </div>
                <span className="run-card-status">
                  <LockKeyhole size={14} aria-hidden="true" />
                  {formatExperimentStatus(experiment.status)}
                </span>
                <strong>{experiment.hypothesis.statement}</strong>
                <span>{experiment.run_plan.model_family} · {experiment.hypothesis.primary_metric}</span>
                <dl>
                  <div><dt>Spec</dt><dd><code>{shortHash(experiment.spec_hash)}</code></dd></div>
                  <div><dt>Dataset</dt><dd><code>{shortHash(experiment.dataset_snapshot.split_signature_sha256)}</code></dd></div>
                  <div><dt>Compare</dt><dd>{experiment.comparison.same_dataset_required ? "same rows required" : "not enforced"}</dd></div>
                </dl>
                <button type="button" className="text-command" onClick={() => setActiveExperiment(experiment)}>
                  Inspect run
                </button>
              </article>
            ))
          )}
        </div>
      </section>
    </div>
  );
}

function LifecycleRail({ experiment }: { experiment: ExperimentRunBundle }) {
  const steps = [
    { key: "spec_locked_preflight", label: "Spec" },
    { key: "review_requested", label: "Review" },
    { key: "review_approved", label: "Approved" },
    { key: "queued", label: "Queued" },
    { key: "running", label: "Running" },
    { key: "completed", label: "Compared" }
  ];
  const activeIndex = Math.max(0, steps.findIndex((step) => step.key === experiment.status));

  return (
    <div className="lifecycle-rail" aria-label="Experiment lifecycle">
      {steps.map((step, index) => (
        <div
          key={step.key}
          className={index <= activeIndex && experiment.status !== "review_rejected" && experiment.status !== "failed" ? "is-complete" : ""}
        >
          <span>{index + 1}</span>
          <strong>{step.label}</strong>
        </div>
      ))}
    </div>
  );
}

function ExperimentActions({
  experiment,
  disabled,
  onRequestReview,
  onApprove,
  onReject,
  onQueue,
  onStartWorker,
  onRefresh
}: {
  experiment: ExperimentRunBundle;
  disabled: boolean;
  onRequestReview: (experiment: ExperimentRunBundle) => void;
  onApprove: (experiment: ExperimentRunBundle) => void;
  onReject: (experiment: ExperimentRunBundle) => void;
  onQueue: (experiment: ExperimentRunBundle) => void;
  onStartWorker: (experiment: ExperimentRunBundle) => void;
  onRefresh: () => void;
}) {
  return (
    <div className="experiment-action-row" aria-label="Experiment lifecycle actions">
      {experiment.status === "spec_locked_preflight" ? (
        <button className="command-button secondary" type="button" disabled={disabled} onClick={() => onRequestReview(experiment)}>
          <ShieldCheck size={15} aria-hidden="true" />
          Queue for Review
        </button>
      ) : null}
      {experiment.status === "review_requested" ? (
        <>
          <button className="command-button secondary" type="button" disabled={disabled} onClick={() => onApprove(experiment)}>
            <CheckCircle2 size={15} aria-hidden="true" />
            Approve Review
          </button>
          <button className="command-button danger" type="button" disabled={disabled} onClick={() => onReject(experiment)}>
            <AlertCircle size={15} aria-hidden="true" />
            Reject
          </button>
        </>
      ) : null}
      {experiment.status === "review_approved" ? (
        <button className="command-button secondary" type="button" disabled={disabled} onClick={() => onQueue(experiment)}>
          <Play size={15} aria-hidden="true" />
          Queue Training
        </button>
      ) : null}
      {experiment.status === "queued" ? (
        <button className="command-button secondary" type="button" disabled={disabled} onClick={() => onStartWorker(experiment)}>
          <Play size={15} aria-hidden="true" />
          Start Worker
        </button>
      ) : null}
      <button className="text-command" type="button" disabled={disabled} onClick={onRefresh}>
        Refresh Registry
      </button>
    </div>
  );
}

function QueueColumn({
  title,
  experiments,
  emptyLabel,
  onSelect
}: {
  title: string;
  experiments: ExperimentRunBundle[];
  emptyLabel: string;
  onSelect: (experiment: ExperimentRunBundle) => void;
}) {
  return (
    <section className="panel queue-column">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Governed Queue</span>
          <h2>{title}</h2>
        </div>
        <span className="status-pill warn">{experiments.length}</span>
      </div>
      <div className="queue-stack">
        {experiments.length === 0 ? (
          <p className="quiet">{emptyLabel}</p>
        ) : (
          experiments.map((experiment) => (
            <button key={experiment.id} type="button" className="queue-item" onClick={() => onSelect(experiment)}>
              <span>{formatExperimentStatus(experiment.status)}</span>
              <strong>{experiment.hypothesis.segment}</strong>
              <small>{experiment.hypothesis.statement}</small>
              <code>{shortHash(experiment.spec_hash)}</code>
            </button>
          ))
        )}
      </div>
    </section>
  );
}

function SpecControl({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ArtifactRef({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      <code>{value}</code>
    </div>
  );
}

function RunStatus({ status }: { status: "idle" | "loading" | "running" | "complete" | "error" }) {
  if (status === "complete") {
    return (
      <span className="status-pill pass">
        <CheckCircle2 size={14} aria-hidden="true" />
        complete
      </span>
    );
  }

  if (status === "error") {
    return (
      <span className="status-pill fail">
        <AlertCircle size={14} aria-hidden="true" />
        error
      </span>
    );
  }

  return <span className="status-pill warn">{status === "running" ? "running" : "ready"}</span>;
}

function formatPercent(value: number) {
  return `${Math.round(value * 1000) / 10}%`;
}

function formatTimestamp(value: string) {
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit"
  }).format(new Date(value));
}

function shortHash(value: string | undefined) {
  if (!value) {
    return "not_set";
  }
  return value.length > 14 ? `${value.slice(0, 14)}...` : value;
}

function formatExperimentStatus(status: ExperimentRunBundle["status"]) {
  return status.replaceAll("_", " ");
}
