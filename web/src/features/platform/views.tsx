import {
  AlertTriangle,
  CheckCircle2,
  Database,
  FileText,
  GitBranch,
  LockKeyhole,
  PackageCheck,
  ShieldAlert,
  type LucideIcon
} from "lucide-react";

import type { GateSummary, PlatformData, SliceMetric } from "@/src/features/platform/data";
import { ReleaseGovernancePanel } from "@/src/features/platform/ReleaseGovernancePanel";
import { ValuationIntake } from "@/src/features/platform/ValuationIntake";

export function WorkbenchView({ data }: { data: PlatformData }) {
  return (
    <div className="view-stack">
      <HeroBand data={data} />
      <div className="two-column-grid">
        <ValuationIntake data={data} />
        <EvidenceRail data={data} />
      </div>
      <SlicePerformance title="Segment Performance" rows={data.package.segmentMetrics} />
    </div>
  );
}

export function GovernanceView({ data }: { data: PlatformData }) {
  return (
    <div className="view-stack">
      <SectionHeader
        eyebrow="Governance"
        title="Release gates, approval state, and model-risk evidence"
        detail="The platform separates candidate evidence from production approval, so a model can be inspectable without being deployable."
      />
      <div className="gate-grid">
        {data.release.gates.map((gate) => (
          <GateCard key={gate.name} gate={gate} />
        ))}
      </div>
      <div className="two-column-grid">
        <section className="panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Release Decision</span>
              <h2>{data.package.decision}</h2>
            </div>
            <span className={`status-pill ${data.package.status === "approved" ? "pass" : "warn"}`}>
              <LockKeyhole size={14} aria-hidden="true" />
              {data.package.status}
            </span>
          </div>
          <dl className="evidence-list">
            <div>
              <dt>Candidate package</dt>
              <dd><code>{data.package.id}</code></dd>
            </div>
            <div>
              <dt>Feature contract</dt>
              <dd><code>{data.package.featureContractVersion}</code></dd>
            </div>
            <div>
              <dt>Dataset version</dt>
              <dd><code>{data.package.datasetVersion}</code></dd>
            </div>
            <div>
              <dt>Readiness report</dt>
              <dd><code>{data.release.reportPath}</code></dd>
            </div>
          </dl>
        </section>
        <section className="panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Blocking Checks</span>
              <h2>{data.release.blockers.length} open blockers</h2>
            </div>
            <ShieldAlert size={20} aria-hidden="true" />
          </div>
          <div className="check-list">
            {data.release.blockers.length === 0 ? (
              <p className="quiet">No blocking checks in the latest readiness report.</p>
            ) : (
              data.release.blockers.map((check) => (
                <div key={check.name}>
                  <strong>{check.name}</strong>
                  <span>{check.detail}</span>
                </div>
              ))
            )}
          </div>
        </section>
      </div>
      <ReleaseGovernancePanel data={data} />
    </div>
  );
}

export function MonitoringView({ data }: { data: PlatformData }) {
  const driftAlerts = data.release.checks.find((check) => check.name === "production_ops_evidence")?.status === "pass";
  return (
    <div className="view-stack">
      <SectionHeader
        eyebrow="Monitoring"
        title="Data freshness, drift evidence, and observed model quality"
        detail="Monitoring reads from generated artifacts and release reports, so the UI reflects the same evidence the gates inspect."
      />
      <div className="metric-grid">
        <MetricPanel icon={Database} label="Latest sale date" value={data.etl.latestSaleDate} detail={data.etl.reportPath} />
        <MetricPanel
          icon={AlertTriangle}
          label="Feature drift alerts"
          value={String(data.package.featureDriftAlerts)}
          detail={`${data.package.featureDriftWarnings} warnings from feature_drift diagnostics`}
          tone={data.package.featureDriftAlerts > 0 ? "warn" : "pass"}
        />
        <MetricPanel icon={CheckCircle2} label="Ops evidence" value={driftAlerts ? "present" : "missing"} detail="drift, performance, retrain policy" tone={driftAlerts ? "pass" : "warn"} />
        <MetricPanel icon={PackageCheck} label="Artifact hashes" value={String(data.package.artifactCount)} detail="files covered by SHA-256 manifest" />
      </div>
      <div className="two-column-grid">
        <SlicePerformance title="Segment Error Surface" rows={data.package.segmentMetrics} />
        <SlicePerformance title="Price Band Error Surface" rows={data.package.tierMetrics} />
      </div>
    </div>
  );
}

export function ArtifactsView({ data }: { data: PlatformData }) {
  return (
    <div className="view-stack">
      <SectionHeader
        eyebrow="Artifact Explorer"
        title="Reproducible model package evidence"
        detail="Every package is a directory of immutable artifacts: manifests, scorecards, hashes, release decision, and model card."
      />
      <div className="artifact-layout">
        <section className="panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Package</span>
              <h2>{data.package.modelVersion} candidate</h2>
            </div>
            <FileText size={20} aria-hidden="true" />
          </div>
          <dl className="evidence-list">
            <div><dt>Path</dt><dd><code>{data.package.path}</code></dd></div>
            <div><dt>Train/Test</dt><dd>{data.package.trainRows.toLocaleString()} / {data.package.testRows.toLocaleString()}</dd></div>
            <div><dt>Data window</dt><dd>{data.package.minSaleDate} to {data.package.maxSaleDate}</dd></div>
            <div><dt>Snapshot hash</dt><dd><code>{shortHash(data.package.dataSnapshotSha256)}</code></dd></div>
            <div><dt>Features</dt><dd>{data.package.featureCount} model features</dd></div>
            <div><dt>Release decision</dt><dd>{data.package.decision}: {data.package.release.reason}</dd></div>
          </dl>
        </section>
        <section className="panel model-card-preview">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Model Card</span>
              <h2>Reviewer preview</h2>
            </div>
            <GitBranch size={20} aria-hidden="true" />
          </div>
          <p>{data.package.modelCardPreview}</p>
        </section>
      </div>
      <div className="two-column-grid">
        <section className="panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Training Manifest</span>
              <h2>{data.package.training.modelClass}</h2>
            </div>
          </div>
          <dl className="evidence-list">
            <div><dt>Command</dt><dd><code>{data.package.training.command}</code></dd></div>
            <div><dt>Git SHA</dt><dd><code>{data.package.training.gitSha}</code></dd></div>
            <div><dt>Python</dt><dd>{data.package.training.pythonVersion}</dd></div>
            <div><dt>Objective</dt><dd>{data.package.training.objective}</dd></div>
            <div><dt>Random seed</dt><dd>{data.package.training.randomSeed}</dd></div>
          </dl>
        </section>
        <section className="panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Data Manifest</span>
              <h2>{data.package.datasetVersion}</h2>
            </div>
          </div>
          <dl className="evidence-list">
            {data.package.sources.map((source) => (
              <div key={`${source.name}-${source.uri}`}>
                <dt>{source.name}</dt>
                <dd><code>{source.uri}</code> · {source.rowCount.toLocaleString()} rows</dd>
              </div>
            ))}
            <div><dt>Known limitations</dt><dd>{data.package.limitations.join(" ")}</dd></div>
          </dl>
        </section>
      </div>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Feature Contract</span>
            <h2>Inference-safe inputs</h2>
          </div>
        </div>
        <div className="feature-chip-row">
          {data.package.featureExamples.map((feature) => (
            <span key={feature}>{feature}</span>
          ))}
        </div>
      </section>
      <ComparableEvidence data={data} />
      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Artifact Hashes</span>
            <h2>{data.package.artifactHashes.length} SHA-256 entries</h2>
          </div>
        </div>
        <div className="hash-grid">
          {data.package.artifactHashes.map((artifact) => (
            <div key={artifact.name}>
              <span>{artifact.name}</span>
              <code>{shortHash(artifact.hash)}</code>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function ComparableEvidence({ data }: { data: PlatformData }) {
  const comps = data.package.comps;
  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Comparable Sales Evidence</span>
          <h2>{comps.selectedRows.toLocaleString()} selected comps</h2>
        </div>
        <span className={`status-pill ${comps.selectedRows > 0 ? "pass" : "warn"}`}>
          {comps.selectedRows > 0 ? "generated" : "missing"}
        </span>
      </div>
      <div className="comp-evidence-summary">
        <MetricTile label="No-comp rate" value={formatPercent(comps.testNoCompRate)} />
        <MetricTile label="Sparse rate" value={formatPercent(comps.testSparseCompRate)} />
        <MetricTile label="High-error rows" value={comps.highErrorRows.toLocaleString()} />
      </div>
      <dl className="evidence-list compact">
        <div><dt>Manifest</dt><dd><code>{comps.manifestPath}</code></dd></div>
        <div><dt>Selected comps</dt><dd><code>{comps.selectedCompsPath}</code></dd></div>
        <div><dt>Review sample</dt><dd><code>{comps.highErrorReviewPath}</code></dd></div>
      </dl>
      <div className="feature-chip-row">
        {comps.featureNames.map((feature) => (
          <span key={feature}>{feature}</span>
        ))}
      </div>
      <div className="comp-table" aria-label="Selected comparable sales sample">
        <div className="comp-row header">
          <span>Rank</span>
          <span>Comp property</span>
          <span>Sale date</span>
          <span>Price</span>
          <span>PPSF</span>
          <span>Distance</span>
          <span>Recency</span>
          <span>Scope</span>
        </div>
        {comps.sample.length === 0 ? (
          <div className="comp-row empty">
            <span>No selected comps artifact is available for this package.</span>
          </div>
        ) : (
          comps.sample.map((comp) => (
            <div className="comp-row" key={`${comp.valuationRowId}-${comp.rank}-${comp.compPropertyId}`}>
              <span>{comp.rank}</span>
              <span>{comp.compPropertyId || "unknown"}</span>
              <span>{comp.compSaleDate || "unknown"}</span>
              <span>{formatCurrency(comp.compSalePrice)}</span>
              <span>{formatCurrency(comp.compPpsf)}</span>
              <span>{formatDistance(comp.compDistanceKm)}</span>
              <span>{Math.round(comp.compRecencyDays).toLocaleString()}d</span>
              <span>{comp.eligibilityScope || "unknown"}</span>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

function HeroBand({ data }: { data: PlatformData }) {
  return (
    <section className="hero-band">
      <div>
        <span className="eyebrow">S.P.E.C. NYC</span>
        <h1>Governed AVM workbench for auditable valuation models</h1>
        <p>
          A transparent model factory for NYC public-data AVM experiments: data contracts, feature contracts,
          candidate packages, release gates, model cards, and scorecards in one inspection surface.
        </p>
      </div>
      <div className="hero-scoreboard" aria-label="Current package metrics">
        <MetricTile label="PPE10" value={formatPercent(data.package.ppe10)} />
        <MetricTile label="MdAPE" value={formatPercent(data.package.mdape)} />
        <MetricTile label="Rows" value={data.etl.transformedRows.toLocaleString()} />
      </div>
    </section>
  );
}

function EvidenceRail({ data }: { data: PlatformData }) {
  const isApproved = data.package.status === "approved";
  const heading = isApproved
    ? "Approved champion package"
    : data.package.status === "missing"
      ? "No model package available"
      : "Candidate package under governance";
  const pillTone = isApproved ? "pass" : data.package.status === "missing" ? "fail" : "warn";
  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Evidence State</span>
          <h2>{heading}</h2>
        </div>
        <span className={`status-pill ${pillTone}`}>{data.package.status}</span>
      </div>
      <dl className="evidence-list">
        <div><dt>Package ID</dt><dd><code>{data.package.id}</code></dd></div>
        <div><dt>Resolver</dt><dd>{data.package.selection?.source ?? "unknown"}</dd></div>
        <div><dt>Release decision</dt><dd>{data.package.decision}</dd></div>
        <div><dt>Raw source rows</dt><dd>{data.etl.rawRows.toLocaleString()}</dd></div>
        <div><dt>Unique properties</dt><dd>{data.etl.uniqueProperties.toLocaleString()}</dd></div>
        <div><dt>Blocked gates</dt><dd>{data.release.gates.filter((gate) => gate.status !== "done").length}</dd></div>
        {data.package.selection?.fallbackReason ? (
          <div><dt>Fallback reason</dt><dd>{data.package.selection.fallbackReason}</dd></div>
        ) : null}
      </dl>
    </section>
  );
}

function GateCard({ gate }: { gate: GateSummary }) {
  const clear = gate.status === "done";
  return (
    <section className="panel gate-card">
      <span className={`status-pill ${clear ? "pass" : "fail"}`}>{gate.status}</span>
      <h2>{gate.name}</h2>
      <p>{clear ? "All required evidence passed." : [...gate.failedChecks, ...gate.missingChecks].join(", ") || "Blocked by upstream gate."}</p>
    </section>
  );
}

function SlicePerformance({ title, rows }: { title: string; rows: SliceMetric[] }) {
  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Scorecard</span>
          <h2>{title}</h2>
        </div>
      </div>
      <div className="score-table">
        <div className="score-row header">
          <span>Slice</span>
          <span>N</span>
          <span>PPE10</span>
          <span>MdAPE</span>
        </div>
        {rows.map((row) => (
          <div className="score-row" key={row.name}>
            <span>{row.name}</span>
            <span>{row.n.toLocaleString()}</span>
            <span>{formatPercent(row.ppe10)}</span>
            <span>{formatPercent(row.mdape)}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function SectionHeader({ eyebrow, title, detail }: { eyebrow: string; title: string; detail: string }) {
  return (
    <header className="section-header">
      <span className="eyebrow">{eyebrow}</span>
      <h1>{title}</h1>
      <p>{detail}</p>
    </header>
  );
}

function MetricPanel({
  icon: Icon,
  label,
  value,
  detail,
  tone
}: {
  icon: LucideIcon;
  label: string;
  value: string;
  detail: string;
  tone?: "pass" | "warn";
}) {
  return (
    <section className={`panel metric-panel ${tone ?? ""}`}>
      <Icon size={20} aria-hidden="true" />
      <span>{label}</span>
      <strong>{value}</strong>
      <p>{detail}</p>
    </section>
  );
}

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function formatPercent(value: number) {
  return `${Math.round(value * 1000) / 10}%`;
}

function formatCurrency(value: number) {
  if (!Number.isFinite(value) || value <= 0) {
    return "n/a";
  }
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0
  }).format(value);
}

function formatDistance(value: number) {
  if (!Number.isFinite(value) || value <= 0) {
    return "n/a";
  }
  return `${Math.round(value * 10) / 10} km`;
}

function shortHash(value: string) {
  return value.length > 16 ? `${value.slice(0, 12)}...${value.slice(-6)}` : value;
}
