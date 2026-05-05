import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";

import { listExperimentRuns, readExperimentRun, type ExperimentRunBundle } from "@/src/features/platform/experimentRegistry";

export type ReleaseGate = {
  name: string;
  status: "pass" | "warn" | "fail";
  detail: string;
};

export type ChampionAlias = {
  registered_model_name: string;
  alias: "champion";
  champion_package_id: string | null;
  champion_package_path: string | null;
  previous_champion_package_id: string | null;
  rollback_package_id: string | null;
  active_proposal_id: string | null;
  approved_at_utc: string | null;
  approved_by: string | null;
  history: Array<{
    proposal_id: string;
    champion_package_id: string;
    previous_champion_package_id: string | null;
    approved_at_utc: string;
    approved_by: string;
    reason: string;
  }>;
};

export type ReleaseProposal = {
  proposal_id: string;
  experiment_id: string;
  created_at_utc: string;
  updated_at_utc: string;
  expires_at_utc: string;
  status: "pending" | "blocked" | "approved" | "rejected" | "expired";
  registered_model_name: string;
  proposal_dir: string;
  artifact_paths: {
    release_proposal: string;
    approval_decision: string;
    audit_log: string;
  };
  candidate: {
    package_id: string;
    package_path: string;
    metrics: {
      ppe10: number;
      mdape: number;
      r2: number | null;
    };
  };
  champion: {
    package_id: string;
    package_path: string;
    metrics: {
      ppe10: number;
      mdape: number;
      r2: number | null;
    };
  };
  rollback: {
    package_id: string;
    package_path: string;
  };
  comparison: {
    status: string;
    comparison_report_path: string;
    same_dataset_contract: boolean;
    metric_deltas: {
      ppe10: number;
      mdape: number;
      r2: number | null;
    };
  };
  gate_results: ReleaseGate[];
  source_artifacts: {
    experiment_spec: string;
    run_manifest: string;
    job_manifest: string | null;
    comparison_report: string;
    model_package: string;
    artifact_hashes: string;
  };
  decision: {
    decided_at_utc: string | null;
    decided_by: string | null;
    reason: string | null;
  };
};

export type ProposalDecisionPayload = {
  decidedBy?: string;
  reason?: string;
};

const REGISTERED_MODEL_NAME = "spec-nyc-avm";
const PROPOSAL_EXPIRY_HOURS = 24;

export async function listGovernanceState(repoRoot: string) {
  const [proposals, eligibleExperiments, champion] = await Promise.all([
    listReleaseProposals(repoRoot),
    listEligibleExperiments(repoRoot),
    readChampionAlias(repoRoot)
  ]);

  return { proposals, eligibleExperiments, champion };
}

export async function listEligibleExperiments(repoRoot: string): Promise<ExperimentRunBundle[]> {
  const experiments = await listExperimentRuns(repoRoot);
  return experiments.filter((experiment) => experiment.status === "completed" && experiment.comparison.status === "passed");
}

export async function listReleaseProposals(repoRoot: string): Promise<ReleaseProposal[]> {
  const root = path.join(repoRoot, "reports/governance/proposals");
  const proposals: ReleaseProposal[] = [];

  try {
    const entries = await fs.readdir(root, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) {
        continue;
      }
      const proposal = await readJson<ReleaseProposal>(path.join(root, entry.name, "release_proposal.json"));
      if (proposal && isReleaseProposal(proposal)) {
        proposals.push(expireIfNeeded(proposal));
      }
    }
  } catch {
    // Empty proposal registry is valid.
  }

  return proposals.sort((a, b) => b.created_at_utc.localeCompare(a.created_at_utc));
}

export async function createReleaseProposal(repoRoot: string, experimentId: string) {
  const experiment = await readExperimentRun(repoRoot, experimentId);
  if (!experiment) {
    throw new ReleaseLifecycleError(`Experiment not found: ${experimentId}.`, 404);
  }
  if (experiment.status !== "completed" || experiment.comparison.status !== "passed") {
    throw new ReleaseLifecycleError("Only completed experiments with passed comparisons can become release proposals.", 409);
  }
  if (!experiment.run_plan.challenger_package_id) {
    throw new ReleaseLifecycleError("Completed experiment is missing challenger package id.", 409);
  }

  const now = new Date().toISOString();
  const proposalId = buildProposalId(now, experiment.id, experiment.run_plan.challenger_package_id);
  const proposalDir = `reports/governance/proposals/${proposalId}`;
  const comparisonReportPath = experiment.artifact_paths.comparison_report;
  const comparisonReport = await readJson<Record<string, unknown>>(path.join(repoRoot, comparisonReportPath));
  const candidatePackageId = experiment.run_plan.challenger_package_id;
  const candidatePackagePath = `models/packages/${candidatePackageId}`;
  const championAlias = await readChampionAlias(repoRoot);
  const championPackageId = championAlias.champion_package_id ?? experiment.run_plan.baseline_package_id;
  const championPackagePath = championAlias.champion_package_path ?? `models/packages/${championPackageId}`;
  const [candidateMetrics, championMetrics] = await Promise.all([
    readPackageMetrics(repoRoot, candidatePackagePath),
    readPackageMetrics(repoRoot, championPackagePath, experiment.baseline_metrics)
  ]);
  const gateResults = await buildProposalGates(repoRoot, {
    experiment,
    comparisonReport,
    candidatePackagePath,
    championPackagePath,
    candidateMetrics
  });
  const hasFailure = gateResults.some((gate) => gate.status === "fail");
  const proposal: ReleaseProposal = {
    proposal_id: proposalId,
    experiment_id: experiment.id,
    created_at_utc: now,
    updated_at_utc: now,
    expires_at_utc: new Date(Date.now() + PROPOSAL_EXPIRY_HOURS * 60 * 60 * 1000).toISOString(),
    status: hasFailure ? "blocked" : "pending",
    registered_model_name: REGISTERED_MODEL_NAME,
    proposal_dir: proposalDir,
    artifact_paths: {
      release_proposal: `${proposalDir}/release_proposal.json`,
      approval_decision: `${proposalDir}/approval_decision.json`,
      audit_log: `${proposalDir}/audit_log.jsonl`
    },
    candidate: {
      package_id: candidatePackageId,
      package_path: candidatePackagePath,
      metrics: candidateMetrics
    },
    champion: {
      package_id: championPackageId,
      package_path: championPackagePath,
      metrics: championMetrics
    },
    rollback: {
      package_id: championPackageId,
      package_path: championPackagePath
    },
    comparison: {
      status: String(comparisonReport?.status ?? experiment.comparison.status),
      comparison_report_path: comparisonReportPath,
      same_dataset_contract: Boolean(comparisonReport?.same_dataset_contract),
      metric_deltas: {
        ppe10: numberValue((comparisonReport?.metric_deltas as Record<string, unknown> | undefined)?.ppe10),
        mdape: numberValue((comparisonReport?.metric_deltas as Record<string, unknown> | undefined)?.mdape),
        r2: nullableNumber((comparisonReport?.metric_deltas as Record<string, unknown> | undefined)?.r2)
      }
    },
    gate_results: gateResults,
    source_artifacts: {
      experiment_spec: experiment.artifact_paths.experiment_spec,
      run_manifest: experiment.artifact_paths.run_manifest,
      job_manifest: experiment.artifact_paths.job_manifest ?? null,
      comparison_report: comparisonReportPath,
      model_package: candidatePackagePath,
      artifact_hashes: `${candidatePackagePath}/artifact_hashes.json`
    },
    decision: {
      decided_at_utc: null,
      decided_by: null,
      reason: null
    }
  };

  await writeProposal(repoRoot, proposal);
  await appendProposalAudit(repoRoot, proposal, {
    event: "release_proposal_created",
    created_at: now,
    proposal_id: proposal.proposal_id,
    experiment_id: proposal.experiment_id,
    candidate_package_id: proposal.candidate.package_id,
    status: proposal.status
  });
  await appendGovernanceAudit(repoRoot, {
    event: "release_proposal_created",
    created_at: now,
    proposal_id: proposal.proposal_id,
    experiment_id: proposal.experiment_id,
    candidate_package_id: proposal.candidate.package_id,
    status: proposal.status
  });

  return proposal;
}

export async function approveReleaseProposal(
  repoRoot: string,
  proposalId: string,
  payload: ProposalDecisionPayload = {}
) {
  const proposal = await requireProposal(repoRoot, proposalId);
  const current = expireIfNeeded(proposal);
  if (current.status !== "pending") {
    if (current.status !== proposal.status) {
      await writeProposal(repoRoot, current);
    }
    throw new ReleaseLifecycleError(`Only pending proposals can be approved. Current status is ${current.status}.`, 409);
  }
  const reason = payload.reason?.trim() ?? "";
  if (reason.length < 12) {
    throw new ReleaseLifecycleError("Approval reason must be at least 12 characters.", 400);
  }
  const failingGate = current.gate_results.find((gate) => gate.status === "fail");
  if (failingGate) {
    throw new ReleaseLifecycleError(`Approval blocked by failing gate: ${failingGate.name}.`, 409);
  }

  const now = new Date().toISOString();
  const approver = payload.decidedBy?.trim() || "local_release_owner";
  const approved: ReleaseProposal = {
    ...current,
    status: "approved",
    updated_at_utc: now,
    decision: {
      decided_at_utc: now,
      decided_by: approver,
      reason
    }
  };
  const decisionRecord = {
    proposal_id: approved.proposal_id,
    decision: "approved",
    decided_at_utc: now,
    decided_by: approver,
    reason,
    candidate_package_id: approved.candidate.package_id,
    previous_champion_package_id: approved.champion.package_id,
    rollback_package_id: approved.rollback.package_id,
    gate_results: approved.gate_results
  };

  await writeProposal(repoRoot, approved);
  await writeJson(repoRoot, approved.artifact_paths.approval_decision, decisionRecord);
  await writeChampionAlias(repoRoot, approved, approver, reason, now);
  await approvePackageRelease(repoRoot, approved, approver, reason, now);
  await appendProposalAudit(repoRoot, approved, {
    event: "release_proposal_approved",
    created_at: now,
    proposal_id: approved.proposal_id,
    candidate_package_id: approved.candidate.package_id,
    approved_by: approver
  });
  await appendGovernanceAudit(repoRoot, {
    event: "champion_changed",
    created_at: now,
    proposal_id: approved.proposal_id,
    champion_package_id: approved.candidate.package_id,
    previous_champion_package_id: approved.champion.package_id,
    approved_by: approver
  });

  return approved;
}

export async function rejectReleaseProposal(
  repoRoot: string,
  proposalId: string,
  payload: ProposalDecisionPayload = {}
) {
  const proposal = await requireProposal(repoRoot, proposalId);
  if (proposal.status !== "pending" && proposal.status !== "blocked") {
    throw new ReleaseLifecycleError(`Only pending or blocked proposals can be rejected. Current status is ${proposal.status}.`, 409);
  }
  const reason = payload.reason?.trim() ?? "";
  if (reason.length < 8) {
    throw new ReleaseLifecycleError("Rejection reason must be at least 8 characters.", 400);
  }

  const now = new Date().toISOString();
  const reviewer = payload.decidedBy?.trim() || "local_release_owner";
  const rejected: ReleaseProposal = {
    ...proposal,
    status: "rejected",
    updated_at_utc: now,
    decision: {
      decided_at_utc: now,
      decided_by: reviewer,
      reason
    }
  };
  const decisionRecord = {
    proposal_id: rejected.proposal_id,
    decision: "rejected",
    decided_at_utc: now,
    decided_by: reviewer,
    reason,
    candidate_package_id: rejected.candidate.package_id,
    gate_results: rejected.gate_results
  };

  await writeProposal(repoRoot, rejected);
  await writeJson(repoRoot, rejected.artifact_paths.approval_decision, decisionRecord);
  await appendProposalAudit(repoRoot, rejected, {
    event: "release_proposal_rejected",
    created_at: now,
    proposal_id: rejected.proposal_id,
    candidate_package_id: rejected.candidate.package_id,
    rejected_by: reviewer,
    reason
  });
  await appendGovernanceAudit(repoRoot, {
    event: "release_proposal_rejected",
    created_at: now,
    proposal_id: rejected.proposal_id,
    candidate_package_id: rejected.candidate.package_id,
    rejected_by: reviewer,
    reason
  });

  return rejected;
}

export async function readChampionAlias(repoRoot: string): Promise<ChampionAlias> {
  const alias = await readJson<ChampionAlias>(path.join(repoRoot, championAliasPath()));
  if (alias) {
    return {
      ...emptyChampionAlias(),
      ...alias,
      history: Array.isArray(alias.history) ? alias.history : []
    };
  }
  return emptyChampionAlias();
}

export class ReleaseLifecycleError extends Error {
  status: number;

  constructor(message: string, status = 400) {
    super(message);
    this.name = "ReleaseLifecycleError";
    this.status = status;
  }
}

async function buildProposalGates(
  repoRoot: string,
  {
    experiment,
    comparisonReport,
    candidatePackagePath,
    championPackagePath,
    candidateMetrics
  }: {
    experiment: ExperimentRunBundle;
    comparisonReport: Record<string, unknown> | null;
    candidatePackagePath: string;
    championPackagePath: string;
    candidateMetrics: ReleaseProposal["candidate"]["metrics"];
  }
): Promise<ReleaseGate[]> {
  const candidateExists = await existsAt(path.join(repoRoot, candidatePackagePath));
  const championExists = await existsAt(path.join(repoRoot, championPackagePath));
  const artifactHashes = await readJson<Record<string, unknown>>(path.join(repoRoot, candidatePackagePath, "artifact_hashes.json"));
  const comparisonStatus = String(comparisonReport?.status ?? experiment.comparison.status);
  const sameDatasetContract = Boolean(comparisonReport?.same_dataset_contract);
  const policy = await readArenaPolicy(repoRoot);
  const quality = (policy.champion_quality as Record<string, unknown> | undefined) ?? {};
  const minOverallPpe10 = numberValue(quality.min_overall_ppe10 ?? 0);
  const maxOverallMdape = numberValue(quality.max_overall_mdape ?? 1);

  return [
    {
      name: "experiment_completed",
      status: experiment.status === "completed" ? "pass" : "fail",
      detail: `Experiment status is ${experiment.status}.`
    },
    {
      name: "comparison_passed",
      status: comparisonStatus === "passed" ? "pass" : "fail",
      detail: `Comparison status is ${comparisonStatus}.`
    },
    {
      name: "same_dataset_contract",
      status: sameDatasetContract ? "pass" : "fail",
      detail: sameDatasetContract
        ? "Champion and challenger comparison used the locked dataset contract."
        : "Comparison report does not prove same-dataset parity."
    },
    {
      name: "candidate_package_exists",
      status: candidateExists ? "pass" : "fail",
      detail: candidatePackagePath
    },
    {
      name: "rollback_pointer_exists",
      status: championExists ? "pass" : "warn",
      detail: championExists
        ? `Rollback target available at ${championPackagePath}.`
        : `Rollback target ${championPackagePath} is not present locally.`
    },
    {
      name: "artifact_hash_manifest",
      status: artifactHashes?.algorithm === "sha256" ? "pass" : "fail",
      detail: `${candidatePackagePath}/artifact_hashes.json`
    },
    {
      name: "min_overall_ppe10",
      status: candidateMetrics.ppe10 >= minOverallPpe10 ? "pass" : "fail",
      detail: `Candidate PPE10 ${formatPercent(candidateMetrics.ppe10)} vs threshold ${formatPercent(minOverallPpe10)}.`
    },
    {
      name: "max_overall_mdape",
      status: candidateMetrics.mdape <= maxOverallMdape ? "pass" : "fail",
      detail: `Candidate MdAPE ${formatPercent(candidateMetrics.mdape)} vs threshold ${formatPercent(maxOverallMdape)}.`
    }
  ];
}

async function readArenaPolicy(repoRoot: string) {
  const policy = await readJson<Record<string, unknown>>(path.join(repoRoot, "config/arena_policy.yaml"));
  return policy ?? { champion_quality: { min_overall_ppe10: 0, max_overall_mdape: 1 } };
}

async function readPackageMetrics(
  repoRoot: string,
  packagePath: string,
  fallback?: ReleaseProposal["candidate"]["metrics"]
) {
  const metrics = await readJson<Record<string, unknown>>(path.join(repoRoot, packagePath, "metrics.json"));
  const overall = metrics?.overall as Record<string, unknown> | undefined;
  if (!overall && fallback) {
    return fallback;
  }
  return {
    ppe10: numberValue(overall?.ppe10),
    mdape: numberValue(overall?.mdape),
    r2: nullableNumber(overall?.r2)
  };
}

async function requireProposal(repoRoot: string, proposalId: string) {
  assertProposalId(proposalId);
  const proposal = await readJson<ReleaseProposal>(
    path.join(repoRoot, "reports/governance/proposals", proposalId, "release_proposal.json")
  );
  if (!proposal || !isReleaseProposal(proposal)) {
    throw new ReleaseLifecycleError(`Release proposal not found: ${proposalId}.`, 404);
  }
  return proposal;
}

async function writeProposal(repoRoot: string, proposal: ReleaseProposal) {
  await writeJson(repoRoot, proposal.artifact_paths.release_proposal, proposal);
}

async function writeChampionAlias(
  repoRoot: string,
  proposal: ReleaseProposal,
  approvedBy: string,
  reason: string,
  approvedAt: string
) {
  const existing = await readChampionAlias(repoRoot);
  const next: ChampionAlias = {
    registered_model_name: REGISTERED_MODEL_NAME,
    alias: "champion",
    champion_package_id: proposal.candidate.package_id,
    champion_package_path: proposal.candidate.package_path,
    previous_champion_package_id: existing.champion_package_id ?? proposal.champion.package_id,
    rollback_package_id: proposal.rollback.package_id,
    active_proposal_id: proposal.proposal_id,
    approved_at_utc: approvedAt,
    approved_by: approvedBy,
    history: [
      {
        proposal_id: proposal.proposal_id,
        champion_package_id: proposal.candidate.package_id,
        previous_champion_package_id: existing.champion_package_id ?? proposal.champion.package_id,
        approved_at_utc: approvedAt,
        approved_by: approvedBy,
        reason
      },
      ...existing.history
    ].slice(0, 25)
  };
  await writeJson(repoRoot, championAliasPath(), next);
}

async function approvePackageRelease(
  repoRoot: string,
  proposal: ReleaseProposal,
  approver: string,
  reason: string,
  decidedAt: string
) {
  const packageDir = path.join(repoRoot, proposal.candidate.package_path);
  const releaseDecisionPath = path.join(packageDir, "release_decision.json");
  const validationReportPath = path.join(packageDir, "validation_report.json");
  const artifactHashesPath = path.join(packageDir, "artifact_hashes.json");
  const releaseDecision = (await readJson<Record<string, unknown>>(releaseDecisionPath)) ?? {};
  const validationReport = (await readJson<Record<string, unknown>>(validationReportPath)) ?? {};

  const nextReleaseDecision = {
    ...releaseDecision,
    proposal_id: proposal.proposal_id,
    decision: "approved",
    candidate_package_id: proposal.candidate.package_id,
    previous_champion_package_id: proposal.champion.package_id,
    rollback_package_id: proposal.rollback.package_id,
    approver,
    reason,
    gate_results: proposal.gate_results,
    decided_at_utc: decidedAt,
    artifact_hashes_sha256: "updated_after_release_decision_write"
  };
  await fs.writeFile(releaseDecisionPath, `${JSON.stringify(nextReleaseDecision, null, 2)}\n`);

  const nextValidationReport = {
    ...validationReport,
    validation_status: "approved_for_local_champion",
    gate_results: mergeValidationGateResults(validationReport.gate_results, proposal.gate_results)
  };
  await fs.writeFile(validationReportPath, `${JSON.stringify(nextValidationReport, null, 2)}\n`);

  const hashes = await buildArtifactHashes(packageDir);
  await fs.writeFile(artifactHashesPath, `${JSON.stringify(hashes, null, 2)}\n`);
}

function mergeValidationGateResults(existing: unknown, proposalGates: ReleaseGate[]) {
  const rows = Array.isArray(existing) ? existing.filter((row) => typeof row === "object" && row !== null) : [];
  const withoutGovernance = rows.filter((row) => (row as Record<string, unknown>).gate !== "governance_approval");
  return [
    ...withoutGovernance,
    {
      gate: "governance_approval",
      status: "pass",
      reason: "release proposal approved through local governance workflow"
    },
    ...proposalGates.map((gate) => ({
      gate: `release_${gate.name}`,
      status: gate.status,
      reason: gate.detail
    }))
  ];
}

async function buildArtifactHashes(packageDir: string) {
  const requiredFiles = [
    "model.joblib",
    "metrics.json",
    "model_card.md",
    "training_manifest.json",
    "data_manifest.json",
    "feature_contract.json",
    "validation_report.json",
    "slice_scorecard.csv",
    "temporal_scorecard.csv",
    "drift_report.json",
    "explainability_manifest.json",
    "release_decision.json"
  ];
  const files: Record<string, string> = {};
  for (const relPath of requiredFiles) {
    const filePath = path.join(packageDir, relPath);
    if (await existsAt(filePath)) {
      files[relPath] = await sha256File(filePath);
    }
  }
  return { algorithm: "sha256", files };
}

async function appendProposalAudit(repoRoot: string, proposal: ReleaseProposal, event: Record<string, unknown>) {
  await appendJsonLine(path.join(repoRoot, proposal.artifact_paths.audit_log), event);
}

async function appendGovernanceAudit(repoRoot: string, event: Record<string, unknown>) {
  await appendJsonLine(path.join(repoRoot, "reports/governance/audit_log.jsonl"), event);
}

async function appendJsonLine(filePath: string, event: Record<string, unknown>) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.appendFile(filePath, `${JSON.stringify(event)}\n`);
}

async function writeJson(repoRoot: string, relativePath: string, value: unknown) {
  const filePath = path.join(repoRoot, relativePath);
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

async function readJson<T>(filePath: string): Promise<T | null> {
  try {
    return JSON.parse(await fs.readFile(filePath, "utf-8")) as T;
  } catch {
    return null;
  }
}

async function existsAt(filePath: string) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function sha256File(filePath: string) {
  const digest = createHash("sha256");
  const content = await fs.readFile(filePath);
  digest.update(content);
  return digest.digest("hex");
}

function emptyChampionAlias(): ChampionAlias {
  return {
    registered_model_name: REGISTERED_MODEL_NAME,
    alias: "champion",
    champion_package_id: null,
    champion_package_path: null,
    previous_champion_package_id: null,
    rollback_package_id: null,
    active_proposal_id: null,
    approved_at_utc: null,
    approved_by: null,
    history: []
  };
}

function championAliasPath() {
  return `models/packages/aliases/${REGISTERED_MODEL_NAME}.json`;
}

function expireIfNeeded(proposal: ReleaseProposal): ReleaseProposal {
  if (proposal.status !== "pending") {
    return proposal;
  }
  if (Date.now() < Date.parse(proposal.expires_at_utc)) {
    return proposal;
  }
  return {
    ...proposal,
    status: "expired",
    updated_at_utc: new Date().toISOString()
  };
}

function buildProposalId(createdAt: string, experimentId: string, packageId: string) {
  const datePart = createdAt.replace(/[-:.]/g, "").slice(0, 15);
  const digest = createHash("sha256").update(`${createdAt}:${experimentId}:${packageId}`).digest("hex").slice(0, 8);
  return `proposal_${datePart}_${digest}`;
}

function assertProposalId(proposalId: string) {
  if (!/^proposal_[a-zA-Z0-9_]+$/.test(proposalId)) {
    throw new ReleaseLifecycleError("Invalid release proposal id.", 400);
  }
}

function numberValue(value: unknown) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
}

function nullableNumber(value: unknown) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function formatPercent(value: number) {
  return `${Math.round(value * 10000) / 100}%`;
}

function isReleaseProposal(value: unknown): value is ReleaseProposal {
  const proposal = value as Partial<ReleaseProposal>;
  return Boolean(
    proposal.proposal_id &&
      proposal.experiment_id &&
      proposal.status &&
      proposal.candidate?.package_id &&
      proposal.champion?.package_id &&
      proposal.artifact_paths?.release_proposal
  );
}
