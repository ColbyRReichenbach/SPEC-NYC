import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";

import type { CheckSummary, PlatformData } from "@/src/features/platform/data";

export type ExperimentStatus =
  | "spec_locked_preflight"
  | "review_requested"
  | "review_approved"
  | "review_rejected"
  | "queued"
  | "running"
  | "completed"
  | "failed";

export type ExperimentRequest = {
  hypothesis?: string;
  expectedEffect?: string;
  segment?: string;
  primaryMetric?: string;
  modelFamily?: string;
  validationPlan?: string;
  trialBudget?: number;
  riskReview?: boolean;
};

export type ExperimentGate = {
  name: string;
  status: "pass" | "warn" | "fail";
  detail: string;
};

export type ReviewRequestPayload = {
  requestedBy?: string;
  reason?: string;
};

export type ReviewDecisionPayload = {
  decision?: "approved" | "rejected";
  reviewer?: string;
  reason?: string;
};

export type QueueTrainingPayload = {
  queuedBy?: string;
  reason?: string;
};

export type TrainingJobManifest = {
  id: string;
  experiment_id: string;
  status: "queued" | "running" | "completed" | "failed";
  queued_at: string;
  queued_by: string;
  reason: string;
  worker_module: "src.experiments.worker";
  execution_mode: "repo_worker_subprocess";
  command: string[];
  command_display: string;
  cwd: ".";
  logs: {
    stdout: string;
    stderr: string;
  };
  training_plan: {
    model_family: string;
    trainer_strategy: "global" | "segmented_router";
    router_mode: "segment_only";
    optuna_trials: number;
    shap_sample_size: number;
    dataset_version: string;
    artifact_tag: string;
  };
  input_dataset: {
    source_path: string;
    data_snapshot_sha256: string;
    split_signature_sha256: string;
  };
  output: {
    model_package_id: string | null;
    model_package_path: string | null;
  };
  started_at?: string;
  completed_at?: string;
  failed_at?: string;
  exit_code?: number | null;
  error?: string;
};

export type ExperimentRunBundle = {
  id: string;
  created_at: string;
  updated_at?: string;
  status: ExperimentStatus;
  run_dir: string;
  manifest_path: string;
  spec_locked: true;
  spec_hash: string;
  artifact_paths: {
    experiment_spec: string;
    dataset_snapshot: string;
    run_manifest: string;
    comparison_report: string;
    audit_log: string;
    review_request?: string;
    review_decision?: string;
    job_manifest?: string;
    training_stdout?: string;
    training_stderr?: string;
  };
  hypothesis: {
    statement: string;
    expected_effect: string;
    segment: string;
    primary_metric: string;
    risk_review_required: boolean;
  };
  run_plan: {
    model_family: string;
    validation_plan: string;
    trial_budget: number;
    baseline_package_id: string;
    challenger_package_id: string | null;
    dataset_version: string;
    feature_contract_version: string;
    split_policy: string;
    row_selection_policy: string;
    promotion_policy: string;
  };
  dataset_snapshot: {
    dataset_version: string;
    source_path: string;
    raw_rows: number;
    transformed_rows: number;
    train_rows: number;
    test_rows: number;
    min_sale_date: string;
    max_sale_date: string;
    data_snapshot_sha256: string;
    split_signature_sha256: string;
    row_identity_note: string;
  };
  baseline_metrics: {
    ppe10: number;
    mdape: number;
    r2: number | null;
  };
  gates: ExperimentGate[];
  production_blockers: CheckSummary[];
  review?: {
    requested_at?: string;
    requested_by?: string;
    request_reason?: string;
    decision?: "approved" | "rejected";
    decided_at?: string;
    reviewer?: string;
    decision_reason?: string;
    required_risk_review: boolean;
  };
  training_job?: TrainingJobManifest | null;
  comparison: {
    status: "not_started" | "blocked" | "ready" | "passed" | "failed";
    champion_package_id: string;
    challenger_package_id: string | null;
    same_dataset_required: true;
    dataset_snapshot_sha256: string;
    split_signature_sha256: string;
    blocking_reason: string;
  };
};

type ExperimentSpec = {
  id: string;
  created_at: string;
  locked_at: string;
  locked: true;
  spec_hash: string;
  hypothesis: ExperimentRunBundle["hypothesis"];
  controls: {
    baseline_package_id: string;
    challenger_package_id: string | null;
    dataset_version: string;
    feature_contract_version: string;
    dataset_snapshot_sha256: string;
    split_signature_sha256: string;
    split_policy: string;
    row_selection_policy: string;
    promotion_policy: string;
  };
  model_spec: {
    model_family: string;
    validation_plan: string;
    trial_budget: number;
    primary_metric: string;
  };
  audit: {
    created_by: string;
    source: string;
  };
};

type LifecycleMutation = {
  bundle: ExperimentRunBundle;
  files?: Record<string, unknown>;
  auditEvent: Record<string, unknown>;
};

const SUPPORTED_TRAINER_FAMILIES: Record<string, TrainingJobManifest["training_plan"]["trainer_strategy"]> = {
  "Global XGBoost baseline": "global",
  "XGBoost segment specialist": "segmented_router"
};

export function validateExperimentRequest(payload: ExperimentRequest) {
  if (!payload.hypothesis || payload.hypothesis.trim().length < 12) {
    return "Hypothesis must be at least 12 characters.";
  }

  if (!payload.expectedEffect || payload.expectedEffect.trim().length < 8) {
    return "Expected effect must be at least 8 characters.";
  }

  if (!payload.segment || !payload.primaryMetric || !payload.modelFamily || !payload.validationPlan) {
    return "Segment, primary metric, model family, and validation plan are required.";
  }

  if (!Number.isFinite(Number(payload.trialBudget)) || Number(payload.trialBudget) < 1) {
    return "Trial budget must be a positive number.";
  }

  return null;
}

export function buildExperimentRunBundle({
  payload,
  data,
  createdAt = new Date().toISOString()
}: {
  payload: ExperimentRequest;
  data: PlatformData;
  createdAt?: string;
}): { bundle: ExperimentRunBundle; files: Record<string, unknown>; auditEvents: Array<Record<string, unknown>> } {
  const id = buildExperimentId(createdAt, payload.hypothesis ?? "");
  const runDir = `reports/experiments/runs/${id}`;
  const artifactPaths: ExperimentRunBundle["artifact_paths"] = {
    experiment_spec: `${runDir}/experiment_spec.json`,
    dataset_snapshot: `${runDir}/dataset_snapshot.json`,
    run_manifest: `${runDir}/run_manifest.json`,
    comparison_report: `${runDir}/comparison_report.json`,
    audit_log: `${runDir}/audit_log.jsonl`,
    review_request: `${runDir}/review_request.json`,
    review_decision: `${runDir}/review_decision.json`,
    job_manifest: `${runDir}/job_manifest.json`,
    training_stdout: `${runDir}/training_stdout.log`,
    training_stderr: `${runDir}/training_stderr.log`
  };
  const splitSignature = buildSplitSignature({
    datasetVersion: data.package.datasetVersion,
    featureContractVersion: data.package.featureContractVersion,
    dataSnapshotSha256: data.package.dataSnapshotSha256,
    trainRows: data.package.trainRows,
    testRows: data.package.testRows,
    minSaleDate: data.package.minSaleDate,
    maxSaleDate: data.package.maxSaleDate
  });
  const datasetSnapshot: ExperimentRunBundle["dataset_snapshot"] = {
    dataset_version: data.package.datasetVersion,
    source_path: data.package.sources[0]?.uri ?? "unknown",
    raw_rows: data.etl.rawRows,
    transformed_rows: data.etl.transformedRows,
    train_rows: data.package.trainRows,
    test_rows: data.package.testRows,
    min_sale_date: data.package.minSaleDate,
    max_sale_date: data.package.maxSaleDate,
    data_snapshot_sha256: data.package.dataSnapshotSha256,
    split_signature_sha256: splitSignature,
    row_identity_note:
      "This run is locked to the package dataset snapshot hash and split signature. Row-level ID materialization is still a future enhancement."
  };
  const hypothesis = {
    statement: String(payload.hypothesis),
    expected_effect: String(payload.expectedEffect),
    segment: String(payload.segment),
    primary_metric: String(payload.primaryMetric),
    risk_review_required: Boolean(payload.riskReview)
  };
  const specWithoutHash = {
    id,
    created_at: createdAt,
    locked_at: createdAt,
    locked: true as const,
    hypothesis,
    controls: {
      baseline_package_id: data.package.id,
      challenger_package_id: null,
      dataset_version: data.package.datasetVersion,
      feature_contract_version: data.package.featureContractVersion,
      dataset_snapshot_sha256: data.package.dataSnapshotSha256,
      split_signature_sha256: splitSignature,
      split_policy: "time_ordered_split",
      row_selection_policy: "champion_and_challenger_must_score_identical_dataset_snapshot_and_split_signature",
      promotion_policy: "requires_comparison_report_and_governance_approval"
    },
    model_spec: {
      model_family: String(payload.modelFamily),
      validation_plan: String(payload.validationPlan),
      trial_budget: Number(payload.trialBudget),
      primary_metric: String(payload.primaryMetric)
    },
    audit: {
      created_by: "local_dashboard_user",
      source: "dashboard_experiment_preflight"
    }
  };
  const specHash = sha256(canonicalJson(specWithoutHash));
  const experimentSpec: ExperimentSpec = { ...specWithoutHash, spec_hash: specHash };
  const runPlan: ExperimentRunBundle["run_plan"] = {
    model_family: experimentSpec.model_spec.model_family,
    validation_plan: experimentSpec.model_spec.validation_plan,
    trial_budget: experimentSpec.model_spec.trial_budget,
    baseline_package_id: experimentSpec.controls.baseline_package_id,
    challenger_package_id: null,
    dataset_version: experimentSpec.controls.dataset_version,
    feature_contract_version: experimentSpec.controls.feature_contract_version,
    split_policy: experimentSpec.controls.split_policy,
    row_selection_policy: experimentSpec.controls.row_selection_policy,
    promotion_policy: experimentSpec.controls.promotion_policy
  };
  const gates: ExperimentGate[] = [
    {
      name: "spec_locked",
      status: "pass",
      detail: `experiment_spec.json locked with hash ${specHash.slice(0, 12)}`
    },
    {
      name: "dataset_snapshot_available",
      status: data.etl.transformedRows > 0 ? "pass" : "fail",
      detail: `${data.etl.transformedRows.toLocaleString()} transformed rows available`
    },
    {
      name: "same_rows_comparison_contract",
      status: "pass",
      detail: `Champion and challenger must share split signature ${splitSignature.slice(0, 12)}`
    },
    {
      name: "feature_contract_available",
      status: data.package.featureContractVersion === "none" ? "fail" : "pass",
      detail: data.package.featureContractVersion
    },
    {
      name: "trainer_supported",
      status: trainingStrategyForModelFamily(runPlan.model_family) ? "pass" : "warn",
      detail: trainingStrategyForModelFamily(runPlan.model_family)
        ? "This model family is wired to the repository trainer."
        : "This model family is logged for research governance, but cannot be queued until a trainer adapter exists."
    },
    {
      name: "production_release_block",
      status: data.release.allGreen ? "warn" : "pass",
      detail: data.release.allGreen
        ? "release gates are clear; promotion still requires explicit approval"
        : "experiment is isolated from production because release gates are blocked"
    }
  ];
  const comparison: ExperimentRunBundle["comparison"] = {
    status: "not_started",
    champion_package_id: data.package.id,
    challenger_package_id: null,
    same_dataset_required: true,
    dataset_snapshot_sha256: data.package.dataSnapshotSha256,
    split_signature_sha256: splitSignature,
    blocking_reason: "No challenger training job has completed for this locked spec."
  };
  const bundle: ExperimentRunBundle = {
    id,
    created_at: createdAt,
    updated_at: createdAt,
    status: "spec_locked_preflight",
    run_dir: runDir,
    manifest_path: artifactPaths.run_manifest,
    spec_locked: true,
    spec_hash: specHash,
    artifact_paths: artifactPaths,
    hypothesis,
    run_plan: runPlan,
    dataset_snapshot: datasetSnapshot,
    baseline_metrics: {
      ppe10: data.package.ppe10,
      mdape: data.package.mdape,
      r2: data.package.r2
    },
    gates,
    production_blockers: data.release.blockers,
    review: {
      required_risk_review: hypothesis.risk_review_required
    },
    training_job: null,
    comparison
  };
  const runManifest = {
    id,
    created_at: createdAt,
    updated_at: createdAt,
    status: bundle.status,
    run_type: "governed_experiment_lifecycle",
    spec_locked: true,
    spec_hash: specHash,
    lifecycle: ["spec_locked", "dataset_snapshot_bound", "comparison_contract_created", "awaiting_review_request"],
    training_job: null,
    artifact_paths: artifactPaths
  };
  const comparisonReport = {
    ...comparison,
    required_before_promotion: true,
    champion_metrics: bundle.baseline_metrics,
    challenger_metrics: null
  };
  const auditEvents = [
    {
      event: "experiment_spec_locked",
      created_at: createdAt,
      experiment_id: id,
      spec_hash: specHash
    },
    {
      event: "dataset_snapshot_bound",
      created_at: createdAt,
      experiment_id: id,
      data_snapshot_sha256: data.package.dataSnapshotSha256,
      split_signature_sha256: splitSignature
    },
    {
      event: "comparison_contract_created",
      created_at: createdAt,
      experiment_id: id,
      same_dataset_required: true
    }
  ];

  return {
    bundle,
    files: {
      experiment_spec: experimentSpec,
      dataset_snapshot: datasetSnapshot,
      run_manifest: runManifest,
      comparison_report: comparisonReport
    },
    auditEvents
  };
}

export async function writeExperimentRunBundle(repoRoot: string, result: ReturnType<typeof buildExperimentRunBundle>) {
  const runDir = path.join(repoRoot, result.bundle.run_dir);
  await fs.mkdir(runDir, { recursive: true });
  await writeJson(repoRoot, result.bundle.artifact_paths.experiment_spec, result.files.experiment_spec);
  await writeJson(repoRoot, result.bundle.artifact_paths.dataset_snapshot, result.files.dataset_snapshot);
  await writeJson(repoRoot, result.bundle.artifact_paths.run_manifest, result.files.run_manifest);
  await writeJson(repoRoot, result.bundle.artifact_paths.comparison_report, result.files.comparison_report);
  await fs.writeFile(
    path.join(repoRoot, result.bundle.artifact_paths.audit_log),
    result.auditEvents.map((event) => JSON.stringify(event)).join("\n") + "\n"
  );

  const experimentDir = path.join(repoRoot, "reports/experiments");
  await fs.mkdir(experimentDir, { recursive: true });
  await writeExperimentSummary(repoRoot, result.bundle);
  await fs.appendFile(path.join(experimentDir, "hypothesis_log.jsonl"), `${JSON.stringify(result.bundle)}\n`);
}

export async function listExperimentRuns(repoRoot: string): Promise<ExperimentRunBundle[]> {
  const experimentDir = path.join(repoRoot, "reports/experiments");
  const runsDir = path.join(experimentDir, "runs");
  const runs: ExperimentRunBundle[] = [];

  try {
    const entries = await fs.readdir(runsDir, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) {
        continue;
      }
      const runManifest = await readJson<Record<string, unknown>>(path.join(runsDir, entry.name, "run_manifest.json"));
      const summary = await readJson<ExperimentRunBundle>(path.join(experimentDir, `${entry.name}.json`));
      if (summary) {
        runs.push(normalizeExperimentRun(summary));
      } else if (runManifest) {
        const migrated = await readJson<ExperimentRunBundle>(path.join(runsDir, entry.name, "summary.json"));
        if (migrated) {
          runs.push(normalizeExperimentRun(migrated));
        }
      }
    }
  } catch {
    // Empty registry is a valid state.
  }

  try {
    const entries = await fs.readdir(experimentDir);
    for (const entry of entries) {
      if (!entry.endsWith(".json") || entry === "hypothesis_log.json") {
        continue;
      }
      const maybeRun = await readJson<ExperimentRunBundle>(path.join(experimentDir, entry));
      if (maybeRun && !runs.some((run) => run.id === maybeRun.id)) {
        runs.push(normalizeExperimentRun(maybeRun));
      }
    }
  } catch {
    // Empty registry is a valid state.
  }

  return runs
    .filter(isExperimentRunBundle)
    .sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
}

export async function readExperimentRun(repoRoot: string, experimentId: string): Promise<ExperimentRunBundle | null> {
  assertExperimentId(experimentId);
  const summary = await readJson<ExperimentRunBundle>(path.join(repoRoot, "reports/experiments", `${experimentId}.json`));
  if (!summary) {
    return null;
  }
  const normalized = normalizeExperimentRun(summary);
  return isExperimentRunBundle(normalized) ? normalized : null;
}

export async function requestExperimentReview(
  repoRoot: string,
  experimentId: string,
  payload: ReviewRequestPayload = {}
) {
  const now = new Date().toISOString();
  const bundle = await requireExperimentRun(repoRoot, experimentId);
  ensureStatus(bundle, ["spec_locked_preflight"], "Only locked preflight specs can be queued for review.");

  const reviewRequest = {
    experiment_id: experimentId,
    requested_at: now,
    requested_by: payload.requestedBy?.trim() || "local_dashboard_user",
    reason: payload.reason?.trim() || "Ready for peer/model-risk review.",
    spec_hash: bundle.spec_hash,
    required_risk_review: bundle.hypothesis.risk_review_required,
    checklist: [
      "Hypothesis is logged before training.",
      "Dataset snapshot and split signature are locked.",
      "Champion/challenger comparisons require identical row contracts.",
      "Promotion remains blocked until comparison and governance approval."
    ]
  };

  bundle.status = "review_requested";
  bundle.updated_at = now;
  bundle.review = {
    ...bundle.review,
    requested_at: now,
    requested_by: reviewRequest.requested_by,
    request_reason: reviewRequest.reason,
    required_risk_review: bundle.hypothesis.risk_review_required
  };
  bundle.comparison = {
    ...bundle.comparison,
    status: "blocked",
    blocking_reason: "Experiment is awaiting human review before a training job can be queued."
  };

  await persistLifecycleMutation(repoRoot, {
    bundle,
    files: { review_request: reviewRequest },
    auditEvent: {
      event: "review_requested",
      created_at: now,
      experiment_id: experimentId,
      requested_by: reviewRequest.requested_by,
      required_risk_review: bundle.hypothesis.risk_review_required
    }
  });

  return bundle;
}

export async function recordExperimentReviewDecision(
  repoRoot: string,
  experimentId: string,
  payload: ReviewDecisionPayload
) {
  const now = new Date().toISOString();
  const decision = payload.decision;
  if (decision !== "approved" && decision !== "rejected") {
    throw new LifecycleError("Review decision must be approved or rejected.", 400);
  }

  const bundle = await requireExperimentRun(repoRoot, experimentId);
  ensureStatus(bundle, ["review_requested"], "Only experiments in review can receive a review decision.");

  const reason = payload.reason?.trim() || "";
  if (decision === "approved" && bundle.hypothesis.risk_review_required && reason.length < 12) {
    throw new LifecycleError("A risk-reviewed approval requires a decision reason.", 400);
  }

  const policyChecks = buildReviewPolicyChecks(bundle);
  const blockingPolicy = policyChecks.find((check) => check.status === "fail");
  if (decision === "approved" && blockingPolicy) {
    throw new LifecycleError(`Review approval blocked by policy check: ${blockingPolicy.name}.`, 409);
  }

  const reviewDecision = {
    experiment_id: experimentId,
    decided_at: now,
    decision,
    reviewer: payload.reviewer?.trim() || "local_reviewer",
    reason: reason || (decision === "approved" ? "Approved for controlled challenger training." : "Rejected by reviewer."),
    previous_status: bundle.status,
    next_status: decision === "approved" ? "review_approved" : "review_rejected",
    policy_checks: policyChecks
  };

  bundle.status = decision === "approved" ? "review_approved" : "review_rejected";
  bundle.updated_at = now;
  bundle.review = {
    ...bundle.review,
    decision,
    decided_at: now,
    reviewer: reviewDecision.reviewer,
    decision_reason: reviewDecision.reason,
    required_risk_review: bundle.hypothesis.risk_review_required
  };
  bundle.comparison = {
    ...bundle.comparison,
    status: "blocked",
    blocking_reason:
      decision === "approved"
        ? "Review approved. Training job has not been queued yet."
        : "Review rejected. Training and promotion are blocked for this spec."
  };

  await persistLifecycleMutation(repoRoot, {
    bundle,
    files: { review_decision: reviewDecision },
    auditEvent: {
      event: decision === "approved" ? "review_approved" : "review_rejected",
      created_at: now,
      experiment_id: experimentId,
      reviewer: reviewDecision.reviewer,
      reason: reviewDecision.reason
    }
  });

  return bundle;
}

export async function queueExperimentTraining(
  repoRoot: string,
  experimentId: string,
  payload: QueueTrainingPayload = {}
) {
  const now = new Date().toISOString();
  const bundle = await requireExperimentRun(repoRoot, experimentId);
  ensureStatus(bundle, ["review_approved"], "Only review-approved experiments can be queued for training.");

  const trainerStrategy = trainingStrategyForModelFamily(bundle.run_plan.model_family);
  if (!trainerStrategy) {
    throw new LifecycleError(
      `No production trainer adapter is configured for "${bundle.run_plan.model_family}".`,
      409
    );
  }

  if (bundle.dataset_snapshot.source_path === "unknown") {
    throw new LifecycleError("Training source path is unknown; cannot queue a reproducible job.", 409);
  }

  const sourcePath = path.join(repoRoot, bundle.dataset_snapshot.source_path);
  try {
    await fs.access(sourcePath);
  } catch {
    throw new LifecycleError(`Training source is missing: ${bundle.dataset_snapshot.source_path}.`, 409);
  }

  const job = buildTrainingJobManifest({
    bundle,
    queuedAt: now,
    queuedBy: payload.queuedBy?.trim() || "local_dashboard_user",
    reason: payload.reason?.trim() || "Approved challenger training job.",
    trainerStrategy
  });

  bundle.status = "queued";
  bundle.updated_at = now;
  bundle.training_job = job;
  bundle.comparison = {
    ...bundle.comparison,
    status: "blocked",
    blocking_reason: "Training job is queued. No challenger metrics are available yet."
  };

  await persistLifecycleMutation(repoRoot, {
    bundle,
    files: { job_manifest: job },
    auditEvent: {
      event: "training_job_queued",
      created_at: now,
      experiment_id: experimentId,
      job_id: job.id,
      queued_by: job.queued_by,
      command_display: job.command_display
    }
  });

  return bundle;
}

export function buildTrainingJobManifest({
  bundle,
  queuedAt,
  queuedBy,
  reason,
  trainerStrategy
}: {
  bundle: ExperimentRunBundle;
  queuedAt: string;
  queuedBy: string;
  reason: string;
  trainerStrategy: TrainingJobManifest["training_plan"]["trainer_strategy"];
}): TrainingJobManifest {
  const optunaTrials = Math.max(0, Math.floor(Number(bundle.run_plan.trial_budget) || 0));
  const shapSampleSize = 250;
  const command = [
    "python3",
    "-m",
    "src.model",
    "--input-csv",
    bundle.dataset_snapshot.source_path,
    "--model-version",
    "v2",
    "--artifact-tag",
    bundle.id,
    "--dataset-version",
    bundle.run_plan.dataset_version,
    "--optuna-trials",
    String(optunaTrials),
    "--shap-sample-size",
    String(shapSampleSize),
    "--no-mlflow",
    "--strategy",
    trainerStrategy
  ];

  if (trainerStrategy === "segmented_router") {
    command.push("--router-mode", "segment_only", "--min-segment-rows", "2000");
  }

  return {
    id: `job_${bundle.id}`,
    experiment_id: bundle.id,
    status: "queued",
    queued_at: queuedAt,
    queued_by: queuedBy,
    reason,
    worker_module: "src.experiments.worker",
    execution_mode: "repo_worker_subprocess",
    command,
    command_display: command.map(shellQuote).join(" "),
    cwd: ".",
    logs: {
      stdout: bundle.artifact_paths.training_stdout ?? `${bundle.run_dir}/training_stdout.log`,
      stderr: bundle.artifact_paths.training_stderr ?? `${bundle.run_dir}/training_stderr.log`
    },
    training_plan: {
      model_family: bundle.run_plan.model_family,
      trainer_strategy: trainerStrategy,
      router_mode: "segment_only",
      optuna_trials: optunaTrials,
      shap_sample_size: shapSampleSize,
      dataset_version: bundle.run_plan.dataset_version,
      artifact_tag: bundle.id
    },
    input_dataset: {
      source_path: bundle.dataset_snapshot.source_path,
      data_snapshot_sha256: bundle.dataset_snapshot.data_snapshot_sha256,
      split_signature_sha256: bundle.dataset_snapshot.split_signature_sha256
    },
    output: {
      model_package_id: null,
      model_package_path: null
    }
  };
}

export function buildExperimentId(createdAt: string, hypothesis: string) {
  const datePart = createdAt.replace(/[-:.]/g, "").slice(0, 15);
  const digest = sha256(`${createdAt}:${hypothesis}`).slice(0, 8);
  return `exp_${datePart}_${digest}`;
}

export function trainingStrategyForModelFamily(modelFamily: string) {
  return SUPPORTED_TRAINER_FAMILIES[modelFamily] ?? null;
}

export class LifecycleError extends Error {
  status: number;

  constructor(message: string, status = 400) {
    super(message);
    this.name = "LifecycleError";
    this.status = status;
  }
}

async function persistLifecycleMutation(repoRoot: string, mutation: LifecycleMutation) {
  const manifest = await readJson<Record<string, unknown>>(path.join(repoRoot, mutation.bundle.artifact_paths.run_manifest));
  const nextManifest = {
    ...(manifest ?? {}),
    id: mutation.bundle.id,
    updated_at: mutation.bundle.updated_at,
    status: mutation.bundle.status,
    lifecycle: appendLifecycleEvent(
      Array.isArray(manifest?.lifecycle) ? manifest.lifecycle.map(String) : [],
      mutation.auditEvent.event as string
    ),
    training_job: mutation.bundle.training_job,
    artifact_paths: mutation.bundle.artifact_paths
  };

  if (mutation.files?.review_request && mutation.bundle.artifact_paths.review_request) {
    await writeJson(repoRoot, mutation.bundle.artifact_paths.review_request, mutation.files.review_request);
  }
  if (mutation.files?.review_decision && mutation.bundle.artifact_paths.review_decision) {
    await writeJson(repoRoot, mutation.bundle.artifact_paths.review_decision, mutation.files.review_decision);
  }
  if (mutation.files?.job_manifest && mutation.bundle.artifact_paths.job_manifest) {
    await writeJson(repoRoot, mutation.bundle.artifact_paths.job_manifest, mutation.files.job_manifest);
  }
  await writeJson(repoRoot, mutation.bundle.artifact_paths.run_manifest, nextManifest);
  await writeExperimentSummary(repoRoot, mutation.bundle);
  await appendAuditEvent(repoRoot, mutation.bundle, mutation.auditEvent);
}

async function writeExperimentSummary(repoRoot: string, bundle: ExperimentRunBundle) {
  const experimentDir = path.join(repoRoot, "reports/experiments");
  await fs.mkdir(experimentDir, { recursive: true });
  await fs.writeFile(path.join(experimentDir, `${bundle.id}.json`), `${JSON.stringify(bundle, null, 2)}\n`);
}

async function appendAuditEvent(repoRoot: string, bundle: ExperimentRunBundle, event: Record<string, unknown>) {
  await fs.appendFile(path.join(repoRoot, bundle.artifact_paths.audit_log), `${JSON.stringify(event)}\n`);
}

async function writeJson(repoRoot: string, relativePath: string, value: unknown) {
  const filePath = path.join(repoRoot, relativePath);
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

async function requireExperimentRun(repoRoot: string, experimentId: string) {
  const run = await readExperimentRun(repoRoot, experimentId);
  if (!run) {
    throw new LifecycleError(`Experiment not found: ${experimentId}.`, 404);
  }
  return run;
}

function ensureStatus(bundle: ExperimentRunBundle, allowed: ExperimentStatus[], message: string) {
  if (!allowed.includes(bundle.status)) {
    throw new LifecycleError(`${message} Current status is ${bundle.status}.`, 409);
  }
}

function buildReviewPolicyChecks(bundle: ExperimentRunBundle): ExperimentGate[] {
  const existingFailures = bundle.gates.filter((gate) => gate.status === "fail");
  const checks: ExperimentGate[] = [
    {
      name: "spec_hash_present",
      status: bundle.spec_hash ? "pass" : "fail",
      detail: bundle.spec_hash ? `Locked hash ${bundle.spec_hash.slice(0, 12)}` : "Missing locked spec hash."
    },
    {
      name: "dataset_bound",
      status: bundle.dataset_snapshot.data_snapshot_sha256 === "unknown" ? "fail" : "pass",
      detail: bundle.dataset_snapshot.data_snapshot_sha256
    },
    {
      name: "trainer_supported",
      status: trainingStrategyForModelFamily(bundle.run_plan.model_family) ? "pass" : "fail",
      detail: trainingStrategyForModelFamily(bundle.run_plan.model_family)
        ? "Trainer adapter exists."
        : "Trainer adapter must be implemented before this experiment can run."
    }
  ];

  return [...checks, ...existingFailures];
}

function buildSplitSignature({
  datasetVersion,
  dataSnapshotSha256,
  trainRows,
  testRows,
  minSaleDate,
  maxSaleDate
}: {
  datasetVersion: string;
  featureContractVersion: string;
  dataSnapshotSha256: string;
  trainRows: number;
  testRows: number;
  minSaleDate: string;
  maxSaleDate: string;
}) {
  return sha256(
    [
      datasetVersion,
      dataSnapshotSha256,
      trainRows,
      testRows,
      minSaleDate,
      maxSaleDate,
      "time_ordered_split"
    ].join("|")
  );
}

function appendLifecycleEvent(current: string[], event: string) {
  const next = [...current, event].filter(Boolean);
  return Array.from(new Set(next));
}

function normalizeExperimentRun(run: ExperimentRunBundle): ExperimentRunBundle {
  const runDir = run.run_dir || `reports/experiments/runs/${run.id}`;
  return {
    ...run,
    status: run.status ?? "spec_locked_preflight",
    run_dir: runDir,
    manifest_path: run.manifest_path || `${runDir}/run_manifest.json`,
    artifact_paths: {
      ...(run.artifact_paths ?? {}),
      review_request: run.artifact_paths?.review_request ?? `${runDir}/review_request.json`,
      review_decision: run.artifact_paths?.review_decision ?? `${runDir}/review_decision.json`,
      job_manifest: run.artifact_paths?.job_manifest ?? `${runDir}/job_manifest.json`,
      training_stdout: run.artifact_paths?.training_stdout ?? `${runDir}/training_stdout.log`,
      training_stderr: run.artifact_paths?.training_stderr ?? `${runDir}/training_stderr.log`
    },
    review: {
      required_risk_review: Boolean(run.hypothesis?.risk_review_required),
      ...run.review
    },
    training_job: run.training_job ?? null,
    comparison: {
      ...(run.comparison ?? {}),
      status: run.comparison?.status ?? "not_started"
    }
  };
}

function assertExperimentId(experimentId: string) {
  if (!/^exp_[a-zA-Z0-9_]+$/.test(experimentId)) {
    throw new LifecycleError("Invalid experiment id.", 400);
  }
}

function sha256(value: string) {
  return createHash("sha256").update(value).digest("hex");
}

function canonicalJson(value: unknown) {
  return JSON.stringify(sortKeys(value));
}

function sortKeys(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortKeys);
  }

  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([key, item]) => [key, sortKeys(item)])
    );
  }

  return value;
}

async function readJson<T>(filePath: string): Promise<T | null> {
  try {
    return JSON.parse(await fs.readFile(filePath, "utf-8")) as T;
  } catch {
    return null;
  }
}

function shellQuote(value: string) {
  if (/^[a-zA-Z0-9_./:=@-]+$/.test(value)) {
    return value;
  }
  return `'${value.replaceAll("'", "'\\''")}'`;
}

function isExperimentRunBundle(value: unknown): value is ExperimentRunBundle {
  const run = value as Partial<ExperimentRunBundle>;
  return Boolean(
    run.id &&
      run.created_at &&
      run.spec_locked === true &&
      run.spec_hash &&
      run.artifact_paths?.experiment_spec &&
      run.artifact_paths?.dataset_snapshot &&
      run.artifact_paths?.run_manifest &&
      run.artifact_paths?.comparison_report &&
      run.dataset_snapshot?.split_signature_sha256 &&
      run.comparison?.same_dataset_required === true
  );
}
