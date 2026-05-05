"use client";

import Link from "next/link";
import type { CSSProperties } from "react";
import { useState } from "react";
import {
  ArrowRight,
  BarChart3,
  CheckCircle2,
  Database,
  ExternalLink,
  FileText,
  FlaskConical,
  GitCompareArrows,
  LineChart,
  LockKeyhole,
  SearchCheck,
  ShieldCheck,
  Sparkles,
  TriangleAlert,
  type LucideIcon
} from "lucide-react";

import type { EdaDisplayTable, EdaHeroMetric, EdaHypothesis, EdaSummaryCard, PlatformData } from "@/src/features/platform/data";
import type { ExperimentRunBundle } from "@/src/features/platform/experimentRegistry";

type CreationState = {
  hypothesisId: string;
  status: "idle" | "running" | "complete" | "error";
  message: string;
  experiment: ExperimentRunBundle | null;
};

const summaryIcons: Record<EdaSummaryCard["icon"], LucideIcon> = {
  database: Database,
  warning: TriangleAlert,
  chart: BarChart3,
  shield: ShieldCheck
};

const tableIcons: Record<EdaDisplayTable["icon"], LucideIcon> = {
  search: SearchCheck,
  compare: GitCompareArrows,
  database: Database,
  line: LineChart
};

export function EdaHypothesisLabView({ data }: { data: PlatformData }) {
  const [creation, setCreation] = useState<CreationState>({
    hypothesisId: "",
    status: "idle",
    message: "",
    experiment: null
  });

  async function lockHypothesisSpec(hypothesis: EdaHypothesis) {
    setCreation({
      hypothesisId: hypothesis.id,
      status: "running",
      message: "Locking governed spec.",
      experiment: null
    });

    let response: Response;
    let payload: { experiment?: ExperimentRunBundle; error?: string };
    try {
      response = await fetch("/api/v1/experiments/preflight", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          hypothesis: `[EDA ${hypothesis.sourceRunTag}] ${hypothesis.statement}`,
          expectedEffect: hypothesis.expectedEffect,
          segment: hypothesis.segment,
          primaryMetric: hypothesis.primaryMetric,
          modelFamily: hypothesis.modelFamily,
          validationPlan: hypothesis.validationPlan,
          trialBudget: hypothesis.trialBudget,
          riskReview: hypothesis.riskReview
        })
      });
      payload = (await response.json()) as { experiment?: ExperimentRunBundle; error?: string };
    } catch (error) {
      setCreation({
        hypothesisId: hypothesis.id,
        status: "error",
        message: error instanceof Error ? error.message : "Unable to create experiment preflight.",
        experiment: null
      });
      return;
    }

    if (!response.ok || !payload.experiment) {
      setCreation({
        hypothesisId: hypothesis.id,
        status: "error",
        message: payload.error ?? "Unable to create experiment preflight.",
        experiment: null
      });
      return;
    }

    setCreation({
      hypothesisId: hypothesis.id,
      status: "complete",
      message: "Locked spec written to experiment registry.",
      experiment: payload.experiment
    });
  }

  return (
    <div className="view-stack">
      <section className="eda-hero">
        <div>
          <span className="eyebrow">EDA & Hypothesis Lab</span>
          <h1>{data.eda.pageTitle}</h1>
          <p>{data.eda.pageDescription}</p>
        </div>
        <div className="eda-hero-metrics" aria-label="Latest EDA summary">
          {data.eda.heroMetrics.map((metric) => (
            <MetricTile key={metric.id} metric={metric} />
          ))}
        </div>
      </section>

      <div className="eda-summary-grid">
        {data.eda.summaryCards.map((card) => (
          <EvidenceCard
            key={card.id}
            icon={summaryIcons[card.icon]}
            label={card.label}
            value={card.value}
            detail={card.detail}
          />
        ))}
      </div>

      <div className="eda-workspace">
        <section className="panel eda-primary-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Hypothesis Backlog</span>
              <h2>{data.eda.hypotheses.length} governed candidates</h2>
            </div>
            <span className="status-pill pass">
              <LockKeyhole size={14} aria-hidden="true" />
              source linked
            </span>
          </div>
          <div className="hypothesis-grid">
            {data.eda.hypotheses.length === 0 ? (
              <div className="empty-state compact">
                <FlaskConical size={22} aria-hidden="true" />
                <strong>No EDA hypotheses found.</strong>
                <p>Run the EDA generator to populate `reports/eda/hypothesis_backlog_*.md`.</p>
              </div>
            ) : (
              data.eda.hypotheses.slice(0, 10).map((hypothesis) => {
                const isActive = creation.hypothesisId === hypothesis.id;
                return (
                  <article key={hypothesis.id} className="hypothesis-card">
                    <div className="hypothesis-card-header">
                      <span>{hypothesis.category}</span>
                      <code>{hypothesis.sourceRunTag}</code>
                    </div>
                    <strong>{hypothesis.statement}</strong>
                    <p>{hypothesis.expectedEffect}</p>
                    <dl>
                      <div><dt>Segment</dt><dd>{hypothesis.segment}</dd></div>
                      <div><dt>Metric</dt><dd>{hypothesis.primaryMetric}</dd></div>
                      <div><dt>Model</dt><dd>{hypothesis.modelFamily}</dd></div>
                      <div><dt>Trials</dt><dd>{hypothesis.trialBudget}</dd></div>
                    </dl>
                    <button
                      className="command-button secondary"
                      type="button"
                      disabled={creation.status === "running"}
                      onClick={() => lockHypothesisSpec(hypothesis)}
                    >
                      <FlaskConical size={15} aria-hidden="true" />
                      {isActive && creation.status === "running" ? "Locking Spec" : "Lock Hypothesis Spec"}
                    </button>
                    {isActive && creation.status !== "idle" ? (
                      <div className={`hypothesis-action-status ${creation.status}`}>
                        {creation.status === "complete" ? <CheckCircle2 size={15} aria-hidden="true" /> : <Sparkles size={15} aria-hidden="true" />}
                        <span>{creation.message}</span>
                        {creation.experiment ? (
                          <Link href="/experiments">
                            Open run <ArrowRight size={13} aria-hidden="true" />
                          </Link>
                        ) : null}
                      </div>
                    ) : null}
                  </article>
                );
              })
            )}
          </div>
        </section>

        <aside className="eda-side-rail">
          <section className="panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">Latest EDA Run</span>
                <h2>{data.eda.latestRunTag}</h2>
              </div>
              <FileText size={20} aria-hidden="true" />
            </div>
            <dl className="evidence-list">
              <div><dt>Generated</dt><dd>{formatTimestamp(data.eda.generatedAt)}</dd></div>
              <div><dt>Report</dt><dd><code>{data.eda.reportPath}</code></dd></div>
              <div><dt>Input</dt><dd><code>{data.eda.inputCsv}</code></dd></div>
              <div><dt>Predictions</dt><dd><code>{data.eda.predictionsCsv}</code></dd></div>
              <div><dt>Command</dt><dd><code>{data.eda.command}</code></dd></div>
            </dl>
          </section>

          <section className="panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">Artifact Index</span>
                <h2>{data.eda.artifactLinks.length} linked artifacts</h2>
              </div>
            </div>
            <div className="eda-artifact-list">
              {data.eda.artifactLinks.length === 0 ? (
                <p className="quiet">No EDA artifacts are linked by the latest manifest.</p>
              ) : (
                data.eda.artifactLinks.map((artifact) => (
                  <Link key={artifact.id} href={artifact.href} className="eda-artifact-link">
                    <div>
                      <strong>{artifact.label}</strong>
                      <span>{artifact.kind} · {formatBytes(artifact.sizeBytes)}</span>
                    </div>
                    <code>{artifact.path}</code>
                    <ExternalLink size={14} aria-hidden="true" />
                  </Link>
                ))
              )}
            </div>
          </section>

          <section className="panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">Run History</span>
                <h2>{data.eda.runs.length} EDA manifests</h2>
              </div>
            </div>
            <div className="eda-run-list">
              {data.eda.runs.map((run) => (
                <div key={run.manifestPath}>
                  <strong>{run.runTag}</strong>
                  <span>{formatTimestamp(run.generatedAt)} · {run.artifactCount} artifacts</span>
                  <code>{run.manifestPath}</code>
                </div>
              ))}
            </div>
          </section>
        </aside>
      </div>

      {chunkTables(data.eda.tables).map((tables, index) => (
        <div className="eda-two-column" key={index}>
          {tables.map((table) => (
            <DataTablePanel key={table.id} table={table} />
          ))}
        </div>
      ))}

      <section className="panel eda-report-panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Report Preview</span>
            <h2>{data.eda.reportPath}</h2>
          </div>
        </div>
        <div className="eda-report-grid">
          {data.eda.reportSections.map((section) => (
            <article key={section.title}>
              <h3>{section.title}</h3>
              <pre>{section.body}</pre>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

function EvidenceCard({
  icon: Icon,
  label,
  value,
  detail
}: {
  icon: LucideIcon;
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <section className="panel eda-evidence-card">
      <Icon size={19} aria-hidden="true" />
      <span>{label}</span>
      <strong>{value}</strong>
      <p>{detail}</p>
    </section>
  );
}

function DataTablePanel({ table }: { table: EdaDisplayTable }) {
  const Icon = tableIcons[table.icon];
  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">{table.eyebrow}</span>
          <h2>{table.title}</h2>
        </div>
        <Icon size={20} aria-hidden="true" />
      </div>
      <div className="eda-table" style={{ "--eda-column-count": table.columns.length } as CSSProperties}>
        <div className="eda-table-row header">
          {table.columns.map((column) => (
            <span key={column}>{column}</span>
          ))}
        </div>
        {table.rows.length === 0 ? (
          <div className="eda-table-row empty">
            <span>{table.emptyLabel}</span>
          </div>
        ) : (
          table.rows.map((row, index) => (
            <div className="eda-table-row" key={index}>
              {row.map((value, valueIndex) => (
                <span key={`${table.columns[valueIndex] ?? "value"}-${valueIndex}`}>{value}</span>
              ))}
            </div>
          ))
        )}
      </div>
    </section>
  );
}

function MetricTile({ metric }: { metric: EdaHeroMetric }) {
  return (
    <div>
      <span>{metric.label}</span>
      <strong>{metric.value}</strong>
    </div>
  );
}

function formatTimestamp(value: string) {
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) {
    return value;
  }
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit"
  }).format(parsed);
}

function formatBytes(value: number) {
  if (!Number.isFinite(value) || value <= 0) {
    return "missing";
  }
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${Math.round(value / 102.4) / 10} KB`;
  }
  return `${Math.round(value / 104857.6) / 10} MB`;
}

function chunkTables(tables: EdaDisplayTable[]) {
  const chunks: EdaDisplayTable[][] = [];
  for (let index = 0; index < tables.length; index += 2) {
    chunks.push(tables.slice(index, index + 2));
  }
  return chunks;
}
