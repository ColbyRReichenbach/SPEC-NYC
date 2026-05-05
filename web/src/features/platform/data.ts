import { promises as fs } from "node:fs";
import path from "node:path";

import { parse } from "csv-parse/sync";

import { resolveRepoRoot } from "@/src/bff/clients/artifactStore";
import {
  resolveDashboardPackageSelection,
  type ResolvedModelPackageSelection
} from "@/src/features/platform/packageResolver";
import {
  DEFAULT_PLATFORM_OPTIONS,
  type PlatformOptions
} from "@/src/features/platform/platformOptions";

type JsonObject = Record<string, any>;

export type GateSummary = {
  name: string;
  status: string;
  failedChecks: string[];
  missingChecks: string[];
};

export type CheckSummary = {
  name: string;
  status: string;
  detail: string;
};

export type SliceMetric = {
  name: string;
  n: number;
  ppe10: number;
  mdape: number;
  r2?: number;
};

export type CompEvidenceRow = {
  valuationRowId: string;
  rank: number;
  compPropertyId: string;
  compSaleDate: string;
  compSalePrice: number;
  compPpsf: number;
  compDistanceKm: number;
  compRecencyDays: number;
  eligibilityScope: string;
};

export type EdaRunSummary = {
  runTag: string;
  generatedAt: string;
  status: string;
  manifestPath: string;
  reportPath: string;
  inputCsv: string;
  predictionsCsv: string;
  artifactCount: number;
};

export type EdaSegmentRegionRow = {
  borough: string;
  propertySegment: string;
  n: number;
  rowShare: number;
  medianSalePrice: number;
  medianPpsf: number;
  ppsfIqr: number;
  sqftImputedRate: number;
};

export type EdaQuarterlyTrendRow = {
  period: string;
  borough: string;
  propertySegment: string;
  n: number;
  medianPpsf: number;
  ppsfQoqChange: number;
};

export type EdaFeatureSignalRow = {
  scope: string;
  borough: string;
  propertySegment: string;
  feature: string;
  n: number;
  correlation: number;
  direction: string;
  absCorr: number;
};

export type EdaErrorSliceRow = {
  sliceType: string;
  sliceName: string;
  n: number;
  mdape: number;
  ppe10: number;
  medianSignedPctError: number;
  overvaluationRate: number;
};

export type EdaHypothesis = {
  id: string;
  category: string;
  statement: string;
  expectedEffect: string;
  segment: string;
  primaryMetric: string;
  modelFamily: string;
  validationPlan: string;
  trialBudget: number;
  riskReview: boolean;
  sourceRunTag: string;
  sourceArtifact: string;
};

export type EdaReportSection = {
  title: string;
  body: string;
};

export type EdaSummaryCard = {
  id: string;
  label: string;
  value: string;
  detail: string;
  icon: "database" | "warning" | "chart" | "shield";
};

export type EdaHeroMetric = {
  id: string;
  label: string;
  value: string;
};

export type EdaArtifactLink = {
  id: string;
  label: string;
  path: string;
  kind: string;
  href: string;
  exists: boolean;
  sizeBytes: number;
  updatedAt: string;
};

export type EdaDisplayTable = {
  id: string;
  eyebrow: string;
  title: string;
  icon: "search" | "compare" | "database" | "line";
  emptyLabel: string;
  columns: string[];
  rows: string[][];
};

export type PlatformData = {
  generatedAt: string;
  package: {
    id: string;
    path: string;
    decision: string;
    status: "missing" | "candidate" | "approved";
    modelVersion: string;
    modelStrategy: string;
    datasetVersion: string;
    featureContractVersion: string;
    trainRows: number;
    testRows: number;
    ppe10: number;
    mdape: number;
    r2: number | null;
    featureCount: number;
    featureDriftAlerts: number;
    featureDriftWarnings: number;
    routerColumns: string[];
    featureExamples: string[];
    artifactCount: number;
    minSaleDate: string;
    maxSaleDate: string;
    dataSnapshotSha256: string;
    artifactHashes: Array<{ name: string; hash: string }>;
    modelCardPreview: string;
    limitations: string[];
    comps: {
      manifestPath: string;
      selectedCompsPath: string;
      highErrorReviewPath: string;
      featureNames: string[];
      selectedRows: number;
      highErrorRows: number;
      testNoCompRate: number;
      testSparseCompRate: number;
      sample: CompEvidenceRow[];
    };
    training: {
      command: string;
      gitSha: string;
      pythonVersion: string;
      modelClass: string;
      randomSeed: string;
      objective: string;
      preprocessingSteps: string[];
    };
    release: {
      approver: string;
      reason: string;
      rollbackPackageId: string;
    };
    sources: Array<{ name: string; uri: string; rowCount: number }>;
    segmentMetrics: SliceMetric[];
    tierMetrics: SliceMetric[];
    selection?: ResolvedModelPackageSelection;
  };
  etl: {
    reportPath: string;
    rawRows: number;
    transformedRows: number;
    uniqueProperties: number;
    latestSaleDate: string;
  };
  release: {
    reportPath: string;
    mode: string;
    allGreen: boolean;
    gates: GateSummary[];
    checks: CheckSummary[];
    blockers: CheckSummary[];
  };
  eda: {
    pageTitle: string;
    pageDescription: string;
    heroMetrics: EdaHeroMetric[];
    status: "missing" | "complete";
    latestRunTag: string;
    generatedAt: string;
    manifestPath: string;
    reportPath: string;
    inputCsv: string;
    predictionsCsv: string;
    command: string;
    externalReferences: string[];
    artifacts: Record<string, string>;
    profile: {
      rowCount: number;
      uniqueProperties: number;
      saleDateMin: string;
      saleDateMax: string;
      medianSalePrice: number;
      medianPpsf: number;
      sqftImputedRate: number;
      yearBuiltImputedRate: number;
      residentialUnitsMissingRate: number;
      neighborhoodCount: number;
      h3IndexCount: number;
      propertySegmentCount: number;
    };
    runs: EdaRunSummary[];
    segmentRegion: EdaSegmentRegionRow[];
    quarterlyTrends: EdaQuarterlyTrendRow[];
    featureSignals: EdaFeatureSignalRow[];
    errorSlices: EdaErrorSliceRow[];
    hypotheses: EdaHypothesis[];
    reportSections: EdaReportSection[];
    summaryCards: EdaSummaryCard[];
    artifactLinks: EdaArtifactLink[];
    tables: EdaDisplayTable[];
  };
  options?: PlatformOptions;
};

const EMPTY_PACKAGE: PlatformData["package"] = {
  id: "no_candidate_package",
  path: "models/packages",
  decision: "missing",
  status: "missing",
  modelVersion: "none",
  modelStrategy: "none",
  datasetVersion: "none",
  featureContractVersion: "none",
  trainRows: 0,
  testRows: 0,
  ppe10: 0,
  mdape: 0,
  r2: null,
  featureCount: 0,
  featureDriftAlerts: 0,
  featureDriftWarnings: 0,
  routerColumns: [],
  featureExamples: [],
  artifactCount: 0,
  minSaleDate: "unknown",
  maxSaleDate: "unknown",
  dataSnapshotSha256: "unknown",
  artifactHashes: [],
  modelCardPreview: "No candidate model card is available.",
  limitations: ["No candidate package has been generated."],
  comps: {
    manifestPath: "missing",
    selectedCompsPath: "missing",
    highErrorReviewPath: "missing",
    featureNames: [],
    selectedRows: 0,
    highErrorRows: 0,
    testNoCompRate: 1,
    testSparseCompRate: 1,
    sample: []
  },
  training: {
    command: "unknown",
    gitSha: "unknown",
    pythonVersion: "unknown",
    modelClass: "unknown",
    randomSeed: "unknown",
    objective: "unknown",
    preprocessingSteps: []
  },
  release: {
    approver: "unknown",
    reason: "unknown",
    rollbackPackageId: "unknown"
  },
  sources: [],
  segmentMetrics: [],
  tierMetrics: []
};

const EMPTY_EDA: PlatformData["eda"] = {
  pageTitle: "No governed EDA run available",
  pageDescription: "Run the governed EDA generator or write an EDA manifest under reports/eda to populate this page.",
  heroMetrics: [],
  status: "missing",
  latestRunTag: "no_eda_run",
  generatedAt: "unknown",
  manifestPath: "reports/eda",
  reportPath: "missing",
  inputCsv: "missing",
  predictionsCsv: "missing",
  command: "python3 -m src.eda.real_estate_eda",
  externalReferences: [],
  artifacts: {},
  profile: {
    rowCount: 0,
    uniqueProperties: 0,
    saleDateMin: "unknown",
    saleDateMax: "unknown",
    medianSalePrice: 0,
    medianPpsf: 0,
    sqftImputedRate: 0,
    yearBuiltImputedRate: 0,
    residentialUnitsMissingRate: 0,
    neighborhoodCount: 0,
    h3IndexCount: 0,
    propertySegmentCount: 0
  },
  runs: [],
  segmentRegion: [],
  quarterlyTrends: [],
  featureSignals: [],
  errorSlices: [],
  hypotheses: [],
  reportSections: [],
  summaryCards: [],
  artifactLinks: [],
  tables: []
};

export async function loadPlatformData(): Promise<PlatformData> {
  const repoRoot = await resolveRepoRoot();
  const [modelPackage, etl, release, eda, options] = await Promise.all([
    loadDashboardPackage(repoRoot),
    loadLatestEtl(repoRoot),
    loadReleaseReadiness(repoRoot),
    loadLatestEdaAnalysis(repoRoot),
    loadPlatformOptions(repoRoot)
  ]);

  return {
    generatedAt: new Date().toISOString(),
    package: modelPackage,
    etl,
    release,
    eda,
    options
  };
}

async function loadDashboardPackage(repoRoot: string): Promise<PlatformData["package"]> {
  const selection = await resolveDashboardPackageSelection(repoRoot);
  if (!selection.packagePath) {
    return { ...EMPTY_PACKAGE, selection };
  }
  return loadPackage(repoRoot, selection.packagePath, selection);
}

async function loadPackage(
  repoRoot: string,
  packageRel: string,
  selection?: ResolvedModelPackageSelection
): Promise<PlatformData["package"]> {

  const [
    metrics,
    releaseDecision,
    dataManifest,
    featureContract,
    trainingManifest,
    artifactHashes,
    modelCard,
    compsManifest,
    selectedCompsHead,
    highErrorReview
  ] = await Promise.all([
    readJson<JsonObject>(repoRoot, `${packageRel}/metrics.json`),
    readJson<JsonObject>(repoRoot, `${packageRel}/release_decision.json`),
    readJson<JsonObject>(repoRoot, `${packageRel}/data_manifest.json`),
    readJson<JsonObject>(repoRoot, `${packageRel}/feature_contract.json`),
    readJson<JsonObject>(repoRoot, `${packageRel}/training_manifest.json`),
    readJson<JsonObject>(repoRoot, `${packageRel}/artifact_hashes.json`),
    readText(repoRoot, `${packageRel}/model_card.md`),
    readJson<JsonObject>(repoRoot, `${packageRel}/comps_manifest.json`),
    readTextHead(repoRoot, `${packageRel}/selected_comps.csv`, 300_000),
    readText(repoRoot, `${packageRel}/high_error_review_sample.csv`)
  ]);

  const metadata = metrics?.metadata ?? {};
  const overall = metrics?.overall ?? {};
  const decision = String(releaseDecision?.decision ?? "pending").toLowerCase();
  const status = decision === "approved" ? "approved" : "candidate";
  const featureNames = Array.isArray(metadata.feature_columns) ? metadata.feature_columns.map(String) : [];
  const routerColumns = Array.isArray(metadata.router_columns) ? metadata.router_columns.map(String) : [];
  const artifactFiles = artifactHashes?.files && typeof artifactHashes.files === "object" ? Object.keys(artifactHashes.files) : [];
  const artifactHashRows = artifactHashes?.files && typeof artifactHashes.files === "object"
    ? Object.entries(artifactHashes.files).map(([name, hash]) => ({ name, hash: String(hash) }))
    : [];
  const sources = Array.isArray(dataManifest?.sources)
    ? dataManifest.sources.map((source: JsonObject) => ({
        name: String(source.name ?? "unknown"),
        uri: String(source.uri ?? "unknown"),
        rowCount: numberValue(source.row_count)
      }))
    : [];

  return {
    id: String(metadata.model_package_id ?? path.basename(packageRel)),
    path: packageRel,
    decision,
    status,
    modelVersion: String(metadata.model_version ?? "unknown"),
    modelStrategy: String(metadata.model_strategy ?? "unknown"),
    datasetVersion: String(metadata.dataset_version ?? dataManifest?.dataset_version ?? "unknown"),
    featureContractVersion: String(metadata.feature_contract_version ?? featureContract?.feature_contract_version ?? "unknown"),
    trainRows: numberValue(metadata.train_rows),
    testRows: numberValue(metadata.test_rows),
    ppe10: numberValue(overall.ppe10),
    mdape: numberValue(overall.mdape),
    r2: typeof overall.r2 === "number" ? overall.r2 : null,
    featureCount: featureNames.length,
    featureDriftAlerts: numberValue(metadata.feature_drift_alerts),
    featureDriftWarnings: numberValue(metadata.feature_drift_warnings),
    routerColumns,
    featureExamples: featureNames.slice(0, 7),
    artifactCount: artifactFiles.length,
    minSaleDate: String(dataManifest?.min_sale_date ?? "unknown"),
    maxSaleDate: String(dataManifest?.max_sale_date ?? "unknown"),
    dataSnapshotSha256: String(dataManifest?.data_snapshot_sha256 ?? "unknown"),
    artifactHashes: artifactHashRows,
    modelCardPreview: previewText(modelCard),
    limitations: Array.isArray(dataManifest?.known_limitations) ? dataManifest.known_limitations.map(String) : [],
    comps: buildCompsEvidence(packageRel, compsManifest, selectedCompsHead, highErrorReview),
    training: {
      command: sanitizeRepoPath(String(trainingManifest?.command ?? "unknown"), repoRoot),
      gitSha: String(trainingManifest?.git_sha ?? "unknown"),
      pythonVersion: String(trainingManifest?.python_version ?? "unknown"),
      modelClass: String(trainingManifest?.model_class ?? "unknown"),
      randomSeed: String(trainingManifest?.random_seed ?? "unknown"),
      objective: String(trainingManifest?.optimization_objective ?? "unknown"),
      preprocessingSteps: Array.isArray(trainingManifest?.preprocessing_steps)
        ? trainingManifest.preprocessing_steps.map(String)
        : []
    },
    release: {
      approver: String(releaseDecision?.approver ?? "unknown"),
      reason: String(releaseDecision?.reason ?? "unknown"),
      rollbackPackageId: String(releaseDecision?.rollback_package_id ?? "unknown")
    },
    sources,
    segmentMetrics: metricsToRows(metrics?.per_segment),
    tierMetrics: metricsToRows(metrics?.per_price_tier),
    selection
  };
}

async function loadLatestEtl(repoRoot: string): Promise<PlatformData["etl"]> {
  const reportRel = await latestFile(repoRoot, "reports/data", (name) => /^etl_run_.*\.csv$/.test(name));
  if (!reportRel) {
    return {
      reportPath: "reports/data",
      rawRows: 0,
      transformedRows: 0,
      uniqueProperties: 0,
      latestSaleDate: "unknown"
    };
  }

  const csv = await readText(repoRoot, reportRel);
  const rows = parseSimpleCsv(csv ?? "");
  const raw = rows.find((row) => row.stage === "extract_raw");
  const final = [...rows].reverse().find((row) => row.stage?.includes("feature") || row.stage?.includes("load")) ?? rows.at(-1);

  return {
    reportPath: reportRel,
    rawRows: numberValue(raw?.rows),
    transformedRows: numberValue(final?.rows),
    uniqueProperties: numberValue(final?.unique_properties),
    latestSaleDate: String(final?.latest_sale_date ?? raw?.latest_sale_date ?? "unknown")
  };
}

async function loadReleaseReadiness(repoRoot: string): Promise<PlatformData["release"]> {
  const reportPath = "reports/validation/v1_readiness_report.json";
  const payload = await readJson<JsonObject>(repoRoot, reportPath);
  if (!payload) {
    return {
      reportPath,
      mode: "missing",
      allGreen: false,
      gates: [],
      checks: [],
      blockers: [{ name: "release_report", status: "fail", detail: "Release readiness report is missing." }]
    };
  }

  const checks = Array.isArray(payload.checks)
    ? payload.checks.map((check: JsonObject) => ({
        name: String(check.name ?? "unknown_check"),
        status: String(check.status ?? "unknown"),
        detail: String(check.detail ?? "")
      }))
    : [];

  const gates = payload.gates && typeof payload.gates === "object"
    ? Object.entries(payload.gates).map(([name, gate]) => {
        const typedGate = gate as JsonObject;
        return {
          name,
          status: String(typedGate.status ?? "unknown"),
          failedChecks: Array.isArray(typedGate.failed_checks) ? typedGate.failed_checks.map(String) : [],
          missingChecks: Array.isArray(typedGate.missing_checks) ? typedGate.missing_checks.map(String) : []
        };
      })
    : [];

  return {
    reportPath,
    mode: String(payload.validation_mode ?? "unknown"),
    allGreen: Boolean(payload.gate_e_all_green),
    gates,
    checks,
    blockers: checks.filter((check) => check.status !== "pass")
  };
}

async function loadLatestEdaAnalysis(repoRoot: string): Promise<PlatformData["eda"]> {
  const manifestRel = await latestFile(repoRoot, "reports/eda", (name) => /^eda_manifest_.*\.json$/.test(name));
  if (!manifestRel) {
    return EMPTY_EDA;
  }

  const manifest = await readJson<JsonObject>(repoRoot, manifestRel);
  if (!manifest) {
    return { ...EMPTY_EDA, manifestPath: manifestRel };
  }

  const artifacts = objectStringValues(manifest.artifacts);
  const [
    runs,
    profile,
    reportText,
    hypothesisBacklog,
    segmentRegionCsv,
    quarterlyTrendsCsv,
    featureSignalsCsv,
    errorSlicesCsv,
    artifactLinks
  ] = await Promise.all([
    loadEdaRunSummaries(repoRoot),
    readJson<JsonObject>(repoRoot, artifacts.data_profile ?? ""),
    readText(repoRoot, artifacts.report ?? ""),
    readText(repoRoot, artifacts.hypothesis_backlog ?? ""),
    readText(repoRoot, artifacts.segment_region_summary ?? ""),
    readText(repoRoot, artifacts.quarterly_market_trends ?? ""),
    readText(repoRoot, artifacts.feature_interaction_signals ?? ""),
    readText(repoRoot, artifacts.model_error_slices ?? ""),
    buildEdaArtifactLinks(repoRoot, manifestRel, artifacts)
  ]);

  const runTag = String(manifest.run_tag ?? "unknown_run");
  const edaProfile = buildEdaProfile(profile);
  const segmentRegion = parseSimpleCsv(segmentRegionCsv ?? "")
    .map(toEdaSegmentRegionRow)
    .sort((a, b) => b.n - a.n)
    .slice(0, 12);
  const quarterlyTrends = parseSimpleCsv(quarterlyTrendsCsv ?? "")
    .map(toEdaQuarterlyTrendRow)
    .sort((a, b) => String(b.period).localeCompare(String(a.period)) || b.n - a.n)
    .slice(0, 12);
  const featureSignals = parseSimpleCsv(featureSignalsCsv ?? "")
    .map(toEdaFeatureSignalRow)
    .sort((a, b) => b.absCorr - a.absCorr)
    .slice(0, 12);
  const errorSlices = parseSimpleCsv(errorSlicesCsv ?? "")
    .map(toEdaErrorSliceRow)
    .sort((a, b) => b.mdape - a.mdape)
    .slice(0, 12);
  const hypotheses = parseEdaHypotheses({
    markdown: hypothesisBacklog ?? "",
    runTag,
    sourceArtifact: artifacts.hypothesis_backlog ?? "missing"
  });
  return {
    pageTitle: String(manifest.page_title ?? "Senior DS review surface for model-risk findings"),
    pageDescription: String(
      manifest.page_description ??
        "Current EDA evidence, segment behavior, underperformance slices, and governed hypothesis handoff are connected to the same artifact registry as model experiments."
    ),
    status: String(manifest.status ?? "missing") === "complete" ? "complete" : "missing",
    latestRunTag: runTag,
    generatedAt: String(manifest.generated_at_utc ?? "unknown"),
    manifestPath: manifestRel,
    reportPath: artifacts.report ?? "missing",
    inputCsv: String(manifest.input_csv ?? "unknown"),
    predictionsCsv: String(manifest.predictions_csv ?? "none"),
    command: sanitizeRepoPath(String(manifest.command ?? "python3 -m src.eda.real_estate_eda"), repoRoot),
    externalReferences: Array.isArray(manifest.external_references) ? manifest.external_references.map(String) : [],
    artifacts,
    profile: edaProfile,
    runs,
    segmentRegion,
    quarterlyTrends,
    featureSignals,
    errorSlices,
    hypotheses,
    reportSections: parseEdaReportSections(reportText ?? ""),
    heroMetrics: buildEdaHeroMetrics(edaProfile, hypotheses),
    summaryCards: buildEdaSummaryCards(edaProfile),
    artifactLinks,
    tables: buildEdaDisplayTables({ errorSlices, featureSignals, segmentRegion, quarterlyTrends })
  };
}

async function loadPlatformOptions(repoRoot: string): Promise<PlatformOptions> {
  const configured = await readJson<Partial<PlatformOptions>>(repoRoot, "config/platform_options.json");
  return {
    valuation: {
      model_aliases: arrayOrDefault(configured?.valuation?.model_aliases, DEFAULT_PLATFORM_OPTIONS.valuation.model_aliases),
      boroughs: arrayOrDefault(configured?.valuation?.boroughs, DEFAULT_PLATFORM_OPTIONS.valuation.boroughs)
    },
    experiments: {
      primary_metrics: arrayOrDefault(configured?.experiments?.primary_metrics, DEFAULT_PLATFORM_OPTIONS.experiments.primary_metrics),
      model_families: arrayOrDefault(configured?.experiments?.model_families, DEFAULT_PLATFORM_OPTIONS.experiments.model_families),
      validation_plans: arrayOrDefault(configured?.experiments?.validation_plans, DEFAULT_PLATFORM_OPTIONS.experiments.validation_plans)
    },
    identity: {
      dashboard_user: String(configured?.identity?.dashboard_user ?? DEFAULT_PLATFORM_OPTIONS.identity.dashboard_user),
      model_risk_reviewer: String(configured?.identity?.model_risk_reviewer ?? DEFAULT_PLATFORM_OPTIONS.identity.model_risk_reviewer),
      release_owner: String(configured?.identity?.release_owner ?? DEFAULT_PLATFORM_OPTIONS.identity.release_owner)
    }
  };
}

function arrayOrDefault(value: unknown, fallback: string[]): string[] {
  return Array.isArray(value) && value.length > 0 ? value.map(String) : fallback;
}

async function loadEdaRunSummaries(repoRoot: string): Promise<EdaRunSummary[]> {
  const dir = path.join(repoRoot, "reports/eda");
  let entries;
  try {
    entries = await fs.readdir(dir, { withFileTypes: true });
  } catch {
    return [];
  }

  const manifests = entries.filter((entry) => entry.isFile() && /^eda_manifest_.*\.json$/.test(entry.name));
  const runs = await Promise.all(
    manifests.map(async (entry) => {
      const rel = path.join("reports/eda", entry.name);
      const manifest = await readJson<JsonObject>(repoRoot, rel);
      const artifacts = objectStringValues(manifest?.artifacts);
      return {
        runTag: String(manifest?.run_tag ?? entry.name.replace(/^eda_manifest_|\.json$/g, "")),
        generatedAt: String(manifest?.generated_at_utc ?? "unknown"),
        status: String(manifest?.status ?? "unknown"),
        manifestPath: rel,
        reportPath: artifacts.report ?? "missing",
        inputCsv: String(manifest?.input_csv ?? "unknown"),
        predictionsCsv: String(manifest?.predictions_csv ?? "none"),
        artifactCount: Object.keys(artifacts).length
      };
    })
  );

  return runs
    .sort((a, b) => String(b.generatedAt).localeCompare(String(a.generatedAt)))
    .slice(0, 8);
}

async function buildEdaArtifactLinks(
  repoRoot: string,
  manifestRel: string,
  artifacts: Record<string, string>
): Promise<EdaArtifactLink[]> {
  const explicitArtifacts = Object.entries({ manifest: manifestRel, ...artifacts })
    .filter(([, relPath]) => isSafeArtifactPath(relPath))
    .map(([label, relPath]) => ({ label: label.replaceAll("_", " "), relPath }));
  const discoveredNotebooks = await discoverEdaNotebookArtifacts(repoRoot);
  const byPath = new Map<string, { label: string; relPath: string }>();

  for (const artifact of [...explicitArtifacts, ...discoveredNotebooks]) {
    byPath.set(artifact.relPath, artifact);
  }

  const links = await Promise.all(
    [...byPath.values()].map(async ({ label, relPath }) => {
      const stat = await statArtifact(repoRoot, relPath);
      return {
        id: slugify(`${label}-${relPath}`),
        label: label || path.basename(relPath),
        path: relPath,
        kind: artifactKind(relPath),
        href: `/artifact-viewer?path=${encodeURIComponent(relPath)}`,
        exists: Boolean(stat),
        sizeBytes: stat?.size ?? 0,
        updatedAt: stat ? new Date(stat.mtimeMs).toISOString() : "missing"
      };
    })
  );

  return links.sort((a, b) => {
    const notebookDelta = Number(b.kind === "notebook") - Number(a.kind === "notebook");
    if (notebookDelta !== 0) return notebookDelta;
    return a.label.localeCompare(b.label);
  });
}

async function discoverEdaNotebookArtifacts(repoRoot: string): Promise<Array<{ label: string; relPath: string }>> {
  const roots = ["reports/eda", "notebooks"];
  const results: Array<{ label: string; relPath: string }> = [];

  for (const relRoot of roots) {
    const absRoot = path.join(repoRoot, relRoot);
    if (!(await existsAt(absRoot))) {
      continue;
    }

    const files = await walkFiles(absRoot, 3);
    for (const file of files) {
      const relPath = path.relative(repoRoot, file);
      if (/\.(ipynb|html?)$/i.test(relPath) && isSafeArtifactPath(relPath)) {
        results.push({ label: notebookLabel(relPath), relPath });
      }
    }
  }

  return results;
}

async function walkFiles(root: string, maxDepth: number): Promise<string[]> {
  async function visit(dir: string, depth: number): Promise<string[]> {
    if (depth > maxDepth) {
      return [];
    }

    let entries;
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch {
      return [];
    }

    const nested = await Promise.all(
      entries.map(async (entry) => {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          return visit(fullPath, depth + 1);
        }
        return entry.isFile() ? [fullPath] : [];
      })
    );

    return nested.flat();
  }

  return visit(root, 0);
}

async function statArtifact(repoRoot: string, relPath: string) {
  if (!isSafeArtifactPath(relPath)) {
    return null;
  }

  try {
    return await fs.stat(path.join(repoRoot, relPath));
  } catch {
    return null;
  }
}

async function existsAt(target: string): Promise<boolean> {
  try {
    await fs.access(target);
    return true;
  } catch {
    return false;
  }
}

function isSafeArtifactPath(relPath: string) {
  if (!relPath || path.isAbsolute(relPath) || relPath.includes("..")) {
    return false;
  }
  return /^(reports|models|docs|config|notebooks)\//.test(relPath);
}

function artifactKind(relPath: string) {
  const ext = path.extname(relPath).toLowerCase();
  if (ext === ".ipynb") return "notebook";
  if (ext === ".html" || ext === ".htm") return "html";
  if (ext === ".md") return "markdown";
  if (ext === ".csv") return "csv";
  if (ext === ".json") return "json";
  return ext.replace(".", "") || "artifact";
}

function notebookLabel(relPath: string) {
  return path.basename(relPath).replace(/\.(ipynb|html?)$/i, "").replaceAll("_", " ");
}

async function latestDirectory(repoRoot: string, relDir: string, predicate: (name: string) => boolean): Promise<string | null> {
  return latestEntry(repoRoot, relDir, predicate, "directory");
}

async function latestFile(repoRoot: string, relDir: string, predicate: (name: string) => boolean): Promise<string | null> {
  return latestEntry(repoRoot, relDir, predicate, "file");
}

async function latestEntry(
  repoRoot: string,
  relDir: string,
  predicate: (name: string) => boolean,
  type: "file" | "directory"
): Promise<string | null> {
  const dir = path.join(repoRoot, relDir);
  let entries;
  try {
    entries = await fs.readdir(dir, { withFileTypes: true });
  } catch {
    return null;
  }

  const candidates = entries.filter((entry) =>
    predicate(entry.name) && (type === "file" ? entry.isFile() : entry.isDirectory())
  );
  if (candidates.length === 0) {
    return null;
  }

  const withStats = await Promise.all(
    candidates.map(async (entry) => {
      const rel = path.join(relDir, entry.name);
      const stat = await fs.stat(path.join(repoRoot, rel));
      return { rel, mtimeMs: stat.mtimeMs };
    })
  );
  withStats.sort((a, b) => b.mtimeMs - a.mtimeMs);
  return withStats[0]?.rel ?? null;
}

async function readJson<T>(repoRoot: string, relPath: string): Promise<T | null> {
  try {
    return JSON.parse(await fs.readFile(path.join(repoRoot, relPath), "utf-8")) as T;
  } catch {
    return null;
  }
}

async function readText(repoRoot: string, relPath: string): Promise<string | null> {
  try {
    return await fs.readFile(path.join(repoRoot, relPath), "utf-8");
  } catch {
    return null;
  }
}

async function readTextHead(repoRoot: string, relPath: string, maxBytes: number): Promise<string | null> {
  let handle: Awaited<ReturnType<typeof fs.open>> | undefined;
  try {
    handle = await fs.open(path.join(repoRoot, relPath), "r");
    const buffer = Buffer.alloc(maxBytes);
    const { bytesRead } = await handle.read(buffer, 0, maxBytes, 0);
    return buffer.subarray(0, bytesRead).toString("utf-8");
  } catch {
    return null;
  } finally {
    await handle?.close().catch(() => null);
  }
}

function buildCompsEvidence(
  packageRel: string,
  manifest: JsonObject | null,
  selectedCompsHead: string | null,
  highErrorReview: string | null
): PlatformData["package"]["comps"] {
  const testCompCount = (manifest?.test as JsonObject | undefined)?.comp_count as JsonObject | undefined;
  return {
    manifestPath: manifest ? `${packageRel}/comps_manifest.json` : "missing",
    selectedCompsPath: selectedCompsHead ? `${packageRel}/selected_comps.csv` : "missing",
    highErrorReviewPath: highErrorReview ? `${packageRel}/high_error_review_sample.csv` : "missing",
    featureNames: Array.isArray(manifest?.feature_names) ? manifest.feature_names.map(String) : [],
    selectedRows: numberValue(manifest?.selected_comps_rows),
    highErrorRows: parseSimpleCsv(highErrorReview ?? "").length,
    testNoCompRate: numberValue(testCompCount?.no_comp_rate),
    testSparseCompRate: numberValue(testCompCount?.sparse_comp_rate),
    sample: parseSimpleCsvHead(selectedCompsHead ?? "")
      .slice(0, 6)
      .map((row) => ({
        valuationRowId: String(row.valuation_row_id ?? ""),
        rank: numberValue(row.comp_rank),
        compPropertyId: String(row.comp_property_id ?? ""),
        compSaleDate: String(row.comp_sale_date ?? ""),
        compSalePrice: numberValue(row.comp_sale_price),
        compPpsf: numberValue(row.comp_ppsf),
        compDistanceKm: numberValue(row.comp_distance_km),
        compRecencyDays: numberValue(row.comp_recency_days),
        eligibilityScope: String(row.eligibility_scope ?? "")
      }))
  };
}

function metricsToRows(payload: unknown): SliceMetric[] {
  if (!payload || typeof payload !== "object") {
    return [];
  }
  return Object.entries(payload as JsonObject)
    .map(([name, value]) => {
      const row = value as JsonObject;
      return {
        name,
        n: numberValue(row.n),
        ppe10: numberValue(row.ppe10),
        mdape: numberValue(row.mdape),
        r2: typeof row.r2 === "number" ? row.r2 : undefined
      };
    })
    .sort((a, b) => b.n - a.n);
}

function parseSimpleCsv(csv: string): Array<Record<string, string>> {
  if (!csv.trim()) {
    return [];
  }

  try {
    return parse(csv, {
      columns: true,
      skip_empty_lines: true,
      trim: true
    }) as Array<Record<string, string>>;
  } catch {
    const lines = csv.trim().split(/\r?\n/).filter(Boolean);
    const [headerLine, ...body] = lines;
    if (!headerLine) {
      return [];
    }
    const headers = headerLine.split(",");
    return body.map((line) => {
      const values = line.split(",");
      return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
    });
  }
}

function parseSimpleCsvHead(csv: string): Array<Record<string, string>> {
  const text = csv.endsWith("\n") ? csv : csv.split(/\r?\n/).slice(0, -1).join("\n");
  return parseSimpleCsv(text);
}

function numberValue(value: unknown): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function objectStringValues(value: unknown): Record<string, string> {
  if (!value || typeof value !== "object") {
    return {};
  }
  return Object.fromEntries(Object.entries(value as JsonObject).map(([key, entry]) => [key, String(entry)]));
}

function buildEdaProfile(profile: JsonObject | null): PlatformData["eda"]["profile"] {
  const missingness = profile?.important_missingness as JsonObject | undefined;
  const cardinality = profile?.cardinality as JsonObject | undefined;
  const salePrice = profile?.sale_price as JsonObject | undefined;
  const ppsf = profile?.price_per_sqft as JsonObject | undefined;
  return {
    rowCount: numberValue(profile?.row_count),
    uniqueProperties: numberValue(profile?.cardinality && (profile.cardinality as JsonObject).property_id),
    saleDateMin: String(profile?.sale_date_min ?? "unknown"),
    saleDateMax: String(profile?.sale_date_max ?? "unknown"),
    medianSalePrice: numberValue(salePrice?.median),
    medianPpsf: numberValue(ppsf?.median),
    sqftImputedRate: numberValue(profile?.sqft_imputed_rate),
    yearBuiltImputedRate: numberValue(profile?.year_built_imputed_rate),
    residentialUnitsMissingRate: numberValue(missingness?.residential_units),
    neighborhoodCount: numberValue(cardinality?.neighborhood),
    h3IndexCount: numberValue(cardinality?.h3_index),
    propertySegmentCount: numberValue(cardinality?.property_segment)
  };
}

function buildEdaHeroMetrics(
  profile: PlatformData["eda"]["profile"],
  hypotheses: EdaHypothesis[]
): EdaHeroMetric[] {
  return [
    {
      id: "rows",
      label: "Rows",
      value: profile.rowCount.toLocaleString()
    },
    {
      id: "sale_window",
      label: "Sale window",
      value: `${profile.saleDateMin} to ${profile.saleDateMax}`
    },
    {
      id: "hypotheses",
      label: "Hypotheses",
      value: hypotheses.length.toLocaleString()
    }
  ];
}

function buildEdaSummaryCards(profile: PlatformData["eda"]["profile"]): EdaSummaryCard[] {
  return [
    {
      id: "dataset",
      label: "Dataset",
      value: `${profile.uniqueProperties.toLocaleString()} properties`,
      detail: `Median sale ${formatCurrency(profile.medianSalePrice)} · median PPSF ${formatCurrency(profile.medianPpsf)}`,
      icon: "database"
    },
    {
      id: "public_record_risk",
      label: "Public-record risk",
      value: `${formatPercent(profile.sqftImputedRate)} sqft imputed`,
      detail: `${formatPercent(profile.residentialUnitsMissingRate)} unit-count missingness`,
      icon: "warning"
    },
    {
      id: "market_granularity",
      label: "Market granularity",
      value: `${profile.neighborhoodCount.toLocaleString()} neighborhoods`,
      detail: `${profile.h3IndexCount.toLocaleString()} H3 cells · ${profile.propertySegmentCount} property segments`,
      icon: "chart"
    },
    {
      id: "date_window",
      label: "Sale window",
      value: `${profile.saleDateMin} to ${profile.saleDateMax}`,
      detail: `${profile.rowCount.toLocaleString()} source rows analyzed`,
      icon: "shield"
    }
  ];
}

function buildEdaDisplayTables({
  errorSlices,
  featureSignals,
  segmentRegion,
  quarterlyTrends
}: {
  errorSlices: EdaErrorSliceRow[];
  featureSignals: EdaFeatureSignalRow[];
  segmentRegion: EdaSegmentRegionRow[];
  quarterlyTrends: EdaQuarterlyTrendRow[];
}): EdaDisplayTable[] {
  return [
    {
      id: "model_error",
      eyebrow: "Model Error",
      title: "Underperforming Slices",
      icon: "search",
      emptyLabel: "No model-error slice artifact is available.",
      columns: ["Slice", "N", "MdAPE", "PPE10", "Overvalued"],
      rows: errorSlices.map((row) => [
        `${row.sliceType}: ${row.sliceName}`,
        row.n.toLocaleString(),
        formatPercent(row.mdape),
        formatPercent(row.ppe10),
        formatPercent(row.overvaluationRate)
      ])
    },
    {
      id: "feature_effects",
      eyebrow: "Non-Stationarity",
      title: "Feature Effects By Slice",
      icon: "compare",
      emptyLabel: "No feature interaction artifact is available.",
      columns: ["Scope", "Feature", "N", "Direction", "Corr"],
      rows: featureSignals.map((row) => [
        `${row.scope} ${row.borough}/${row.propertySegment}`,
        row.feature,
        row.n.toLocaleString(),
        row.direction,
        formatNumber(row.correlation)
      ])
    },
    {
      id: "segment_region",
      eyebrow: "Market Slices",
      title: "Segment And Region Structure",
      icon: "database",
      emptyLabel: "No segment-region summary is available.",
      columns: ["Borough/Segment", "N", "Share", "Median PPSF", "IQR"],
      rows: segmentRegion.map((row) => [
        `${row.borough} / ${row.propertySegment}`,
        row.n.toLocaleString(),
        formatPercent(row.rowShare),
        formatCurrency(row.medianPpsf),
        formatCurrency(row.ppsfIqr)
      ])
    },
    {
      id: "quarterly_trends",
      eyebrow: "Temporal",
      title: "Latest Quarterly Market Trend Sample",
      icon: "line",
      emptyLabel: "No quarterly trend artifact is available.",
      columns: ["Period", "Slice", "N", "Median PPSF", "QoQ"],
      rows: quarterlyTrends.map((row) => [
        row.period,
        `${row.borough} / ${row.propertySegment}`,
        row.n.toLocaleString(),
        formatCurrency(row.medianPpsf),
        formatPercent(row.ppsfQoqChange)
      ])
    }
  ];
}

function toEdaSegmentRegionRow(row: Record<string, string>): EdaSegmentRegionRow {
  return {
    borough: String(row.borough ?? "unknown"),
    propertySegment: String(row.property_segment ?? "unknown"),
    n: numberValue(row.n),
    rowShare: numberValue(row.row_share),
    medianSalePrice: numberValue(row.median_sale_price),
    medianPpsf: numberValue(row.median_ppsf),
    ppsfIqr: numberValue(row.ppsf_iqr),
    sqftImputedRate: numberValue(row.sqft_imputed_rate)
  };
}

function toEdaQuarterlyTrendRow(row: Record<string, string>): EdaQuarterlyTrendRow {
  return {
    period: String(row.period ?? "unknown"),
    borough: String(row.borough ?? "unknown"),
    propertySegment: String(row.property_segment ?? "unknown"),
    n: numberValue(row.n),
    medianPpsf: numberValue(row.median_ppsf),
    ppsfQoqChange: numberValue(row.ppsf_qoq_change)
  };
}

function toEdaFeatureSignalRow(row: Record<string, string>): EdaFeatureSignalRow {
  return {
    scope: String(row.scope ?? "unknown"),
    borough: String(row.borough ?? "ALL"),
    propertySegment: String(row.property_segment ?? "ALL"),
    feature: String(row.feature ?? "unknown"),
    n: numberValue(row.n),
    correlation: numberValue(row.spearman_corr_log_ppsf),
    direction: String(row.direction ?? "unknown"),
    absCorr: numberValue(row.abs_corr)
  };
}

function toEdaErrorSliceRow(row: Record<string, string>): EdaErrorSliceRow {
  return {
    sliceType: String(row.slice_type ?? "unknown"),
    sliceName: String(row.slice_name ?? "unknown"),
    n: numberValue(row.n),
    mdape: numberValue(row.mdape),
    ppe10: numberValue(row.ppe10),
    medianSignedPctError: numberValue(row.median_signed_pct_error),
    overvaluationRate: numberValue(row.overvaluation_rate)
  };
}

function parseEdaHypotheses({
  markdown,
  runTag,
  sourceArtifact
}: {
  markdown: string;
  runTag: string;
  sourceArtifact: string;
}): EdaHypothesis[] {
  const hypotheses: EdaHypothesis[] = [];
  let category = "Backlog";

  for (const line of markdown.split(/\r?\n/)) {
    const heading = /^##\s+(.+)$/.exec(line.trim());
    if (heading?.[1]) {
      category = heading[1].trim();
      continue;
    }

    if (!line.trim().startsWith("- ")) {
      continue;
    }

    const statement = line.trim().replace(/^-\s+/, "");
    const index = hypotheses.length + 1;
    hypotheses.push({
      id: `${runTag}_${slugify(category)}_${index}`,
      category,
      statement,
      expectedEffect: expectedEffectForHypothesis(category, statement),
      segment: inferSegment(statement),
      primaryMetric: statement.includes("PPE10") && !statement.includes("MdAPE") ? "PPE10" : "MdAPE",
      modelFamily: modelFamilyForHypothesis(category, statement),
      validationPlan: "Time split + borough/segment slices",
      trialBudget: category.includes("Architecture") ? 12 : 8,
      riskReview: true,
      sourceRunTag: runTag,
      sourceArtifact
    });
  }

  return hypotheses.slice(0, 24);
}

function parseEdaReportSections(markdown: string): EdaReportSection[] {
  const sections: EdaReportSection[] = [];
  let title = "";
  let body: string[] = [];

  for (const line of markdown.split(/\r?\n/)) {
    const heading = /^##\s+(.+)$/.exec(line.trim());
    if (heading?.[1]) {
      if (title) {
        sections.push({ title, body: body.join("\n").trim() });
      }
      title = heading[1].trim();
      body = [];
      continue;
    }
    body.push(line);
  }

  if (title) {
    sections.push({ title, body: body.join("\n").trim() });
  }

  return sections
    .filter((section) => section.body)
    .map((section) => ({
      title: section.title,
      body: section.body.length > 1200 ? `${section.body.slice(0, 1200).trim()}...` : section.body
    }));
}

function inferSegment(statement: string): string {
  for (const segment of ["SINGLE_FAMILY", "SMALL_MULTI", "WALKUP", "ELEVATOR"]) {
    if (statement.includes(segment)) {
      return segment;
    }
  }
  return "ALL";
}

function modelFamilyForHypothesis(category: string, statement: string): string {
  if (category.includes("Architecture") || statement.toLowerCase().includes("global")) {
    return "Global XGBoost baseline";
  }
  return "XGBoost segment specialist";
}

function expectedEffectForHypothesis(category: string, statement: string): string {
  const lower = statement.toLowerCase();
  if (category.includes("Underperforming")) {
    return "Reduce slice MdAPE without degrading global PPE10 or high-error review gates.";
  }
  if (category.includes("Non-Stationary")) {
    return "Reduce residual bias by allowing controlled segment or geography interaction effects.";
  }
  if (category.includes("Sparse") || lower.includes("abstention") || lower.includes("fallback")) {
    return "Improve hit/no-hit behavior and make low-evidence valuations auditable.";
  }
  return "Compare challenger architecture on the same locked rows and split signature.";
}

function slugify(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
}

function formatPercent(value: number) {
  if (!Number.isFinite(value)) {
    return "n/a";
  }
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

function formatNumber(value: number) {
  if (!Number.isFinite(value)) {
    return "n/a";
  }
  return String(Math.round(value * 1000) / 1000);
}

function previewText(value: string | null): string {
  if (!value) {
    return "No model card preview is available.";
  }
  return value.replace(/\s+/g, " ").trim().slice(0, 520);
}

function sanitizeRepoPath(value: string, repoRoot: string): string {
  return value.replaceAll(repoRoot, ".");
}
