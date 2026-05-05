import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";

import type { PlatformData } from "@/src/features/platform/data";
import { buildExperimentRunBundle, writeExperimentRunBundle } from "@/src/features/platform/experimentRegistry";
import {
  approveReleaseProposal,
  createReleaseProposal,
  listGovernanceState,
  rejectReleaseProposal
} from "@/src/features/platform/releaseRegistry";

const platformData: PlatformData = {
  generatedAt: "2026-05-05T00:00:00.000Z",
  package: {
    id: "champion_package",
    path: "models/packages/champion_package",
    decision: "pending",
    status: "candidate",
    modelVersion: "v2",
    modelStrategy: "global",
    datasetVersion: "fixture_ds",
    featureContractVersion: "fc_fixture",
    trainRows: 80,
    testRows: 20,
    ppe10: 0.2,
    mdape: 0.3,
    r2: 0.4,
    featureCount: 2,
    featureDriftAlerts: 0,
    featureDriftWarnings: 0,
    routerColumns: [],
    featureExamples: ["gross_square_feet"],
    artifactCount: 3,
    minSaleDate: "2024-01-01",
    maxSaleDate: "2024-12-31",
    dataSnapshotSha256: "snapshot_fixture",
    artifactHashes: [],
    modelCardPreview: "card",
    limitations: [],
    comps: {
      manifestPath: "models/packages/champion_package/comps_manifest.json",
      selectedCompsPath: "models/packages/champion_package/selected_comps.csv",
      highErrorReviewPath: "models/packages/champion_package/high_error_review_sample.csv",
      featureNames: ["comp_count"],
      selectedRows: 4,
      highErrorRows: 1,
      testNoCompRate: 0,
      testSparseCompRate: 0,
      sample: []
    },
    training: {
      command: "fixture",
      gitSha: "abc123",
      pythonVersion: "3.10",
      modelClass: "XGBRegressor",
      randomSeed: "42",
      objective: "mdape",
      preprocessingSteps: []
    },
    release: {
      approver: "not_approved",
      reason: "pending",
      rollbackPackageId: "not_set"
    },
    sources: [{ name: "training_source", uri: "data/processed/training.csv", rowCount: 100 }],
    segmentMetrics: [],
    tierMetrics: []
  },
  etl: {
    reportPath: "reports/data/etl.csv",
    rawRows: 100,
    transformedRows: 100,
    uniqueProperties: 100,
    latestSaleDate: "2024-12-31"
  },
  release: {
    reportPath: "reports/validation/report.json",
    mode: "production",
    allGreen: false,
    gates: [],
    checks: [],
    blockers: []
  },
  eda: {
    pageTitle: "Fixture EDA",
    pageDescription: "Fixture EDA description",
    heroMetrics: [],
    status: "complete",
    latestRunTag: "fixture",
    generatedAt: "2026-05-05T00:00:00.000Z",
    manifestPath: "reports/eda/eda_manifest_fixture.json",
    reportPath: "reports/eda/avm_eda_report_fixture.md",
    inputCsv: "data/processed/training.csv",
    predictionsCsv: "reports/model/predictions.csv",
    command: "python3 -m src.eda.real_estate_eda",
    externalReferences: [],
    artifacts: {},
    profile: {
      rowCount: 100,
      uniqueProperties: 90,
      saleDateMin: "2024-01-01",
      saleDateMax: "2024-12-31",
      medianSalePrice: 500000,
      medianPpsf: 500,
      sqftImputedRate: 0,
      yearBuiltImputedRate: 0,
      residentialUnitsMissingRate: 0,
      neighborhoodCount: 1,
      h3IndexCount: 1,
      propertySegmentCount: 1
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
  }
};

const request = {
  hypothesis: "Fixture completed experiment should produce a release proposal.",
  expectedEffect: "Lower MdAPE without reducing PPE10.",
  segment: "ALL",
  primaryMetric: "MdAPE",
  modelFamily: "Global XGBoost baseline",
  validationPlan: "Time split + borough/segment slices",
  trialBudget: 1,
  riskReview: true
};

describe("release registry contract", () => {
  it("creates a release proposal from a completed experiment", async () => {
    const { repoRoot, experimentId } = await makeCompletedExperimentFixture();

    const proposal = await createReleaseProposal(repoRoot, experimentId);

    expect(proposal.status).toBe("pending");
    expect(proposal.candidate.package_id).toBe("challenger_package");
    expect(proposal.champion.package_id).toBe("champion_package");
    expect(proposal.rollback.package_id).toBe("champion_package");
    expect(proposal.gate_results.every((gate) => gate.status !== "fail")).toBe(true);

    const state = await listGovernanceState(repoRoot);
    expect(state.proposals.map((item) => item.proposal_id)).toContain(proposal.proposal_id);
    expect(state.eligibleExperiments.map((item) => item.id)).toContain(experimentId);
  });

  it("approval writes champion alias, approval decision, audit event, and refreshed package hash", async () => {
    const { repoRoot, experimentId } = await makeCompletedExperimentFixture();
    const proposal = await createReleaseProposal(repoRoot, experimentId);

    const approved = await approveReleaseProposal(repoRoot, proposal.proposal_id, {
      decidedBy: "fixture_owner",
      reason: "Approved after fixture governance review."
    });

    expect(approved.status).toBe("approved");

    const championAlias = JSON.parse(await readFile(path.join(repoRoot, "models/packages/aliases/spec-nyc-avm.json"), "utf-8"));
    expect(championAlias.champion_package_id).toBe("challenger_package");
    expect(championAlias.rollback_package_id).toBe("champion_package");

    const releaseDecisionPath = path.join(repoRoot, "models/packages/challenger_package/release_decision.json");
    const releaseDecision = JSON.parse(await readFile(releaseDecisionPath, "utf-8"));
    expect(releaseDecision.decision).toBe("approved");
    expect(releaseDecision.approver).toBe("fixture_owner");

    const artifactHashes = JSON.parse(await readFile(path.join(repoRoot, "models/packages/challenger_package/artifact_hashes.json"), "utf-8"));
    expect(artifactHashes.files["release_decision.json"]).toBe(await sha256File(releaseDecisionPath));

    const auditLog = await readFile(path.join(repoRoot, "reports/governance/audit_log.jsonl"), "utf-8");
    expect(auditLog).toContain("champion_changed");
  });

  it("rejects a release proposal without changing champion alias", async () => {
    const { repoRoot, experimentId } = await makeCompletedExperimentFixture();
    const proposal = await createReleaseProposal(repoRoot, experimentId);

    const rejected = await rejectReleaseProposal(repoRoot, proposal.proposal_id, {
      decidedBy: "fixture_owner",
      reason: "Insufficient review confidence."
    });

    expect(rejected.status).toBe("rejected");
    const state = await listGovernanceState(repoRoot);
    expect(state.champion.champion_package_id).toBeNull();
  });
});

async function makeCompletedExperimentFixture() {
  const repoRoot = await mkdtemp(path.join(os.tmpdir(), "spec-release-registry-"));
  await mkdir(path.join(repoRoot, "reports/experiments/runs"), { recursive: true });
  await mkdir(path.join(repoRoot, "models/packages/champion_package"), { recursive: true });
  await mkdir(path.join(repoRoot, "models/packages/challenger_package"), { recursive: true });
  await mkdir(path.join(repoRoot, "config"), { recursive: true });
  await writeFile(
    path.join(repoRoot, "config/arena_policy.yaml"),
    JSON.stringify({ champion_quality: { min_overall_ppe10: 0.24, max_overall_mdape: 0.3 } }, null, 2),
    "utf-8"
  );
  await writePackage(repoRoot, "champion_package", { ppe10: 0.25, mdape: 0.3, r2: 0.4 });
  await writePackage(repoRoot, "challenger_package", { ppe10: 0.27, mdape: 0.29, r2: 0.42 });

  const result = buildExperimentRunBundle({
    payload: request,
    data: platformData,
    createdAt: "2026-05-05T00:00:00.000Z"
  });
  result.bundle.status = "completed";
  result.bundle.updated_at = "2026-05-05T00:05:00.000Z";
  result.bundle.run_plan.challenger_package_id = "challenger_package";
  result.bundle.comparison = {
    ...result.bundle.comparison,
    status: "passed",
    challenger_package_id: "challenger_package",
    blocking_reason: "Comparison passed metric gates."
  };
  await writeExperimentRunBundle(repoRoot, result);

  const comparisonReport = {
    status: "passed",
    champion_package_id: "champion_package",
    challenger_package_id: "challenger_package",
    same_dataset_required: true,
    same_dataset_contract: true,
    dataset_snapshot_sha256: result.bundle.dataset_snapshot.data_snapshot_sha256,
    split_signature_sha256: result.bundle.dataset_snapshot.split_signature_sha256,
    candidate_split_signature_sha256: result.bundle.dataset_snapshot.split_signature_sha256,
    champion_metrics: { ppe10: 0.25, mdape: 0.3, r2: 0.4 },
    challenger_metrics: { ppe10: 0.27, mdape: 0.29, r2: 0.42 },
    metric_deltas: { ppe10: 0.02, mdape: -0.01, r2: 0.02 },
    blocking_reason: "Comparison passed metric gates."
  };
  await writeFile(
    path.join(repoRoot, result.bundle.artifact_paths.comparison_report),
    `${JSON.stringify(comparisonReport, null, 2)}\n`,
    "utf-8"
  );
  await writeFile(
    path.join(repoRoot, "reports/experiments", `${result.bundle.id}.json`),
    `${JSON.stringify(result.bundle, null, 2)}\n`,
    "utf-8"
  );

  return { repoRoot, experimentId: result.bundle.id };
}

async function writePackage(repoRoot: string, packageId: string, metrics: { ppe10: number; mdape: number; r2: number }) {
  const packageDir = path.join(repoRoot, "models/packages", packageId);
  const files: Record<string, string> = {
    "metrics.json": JSON.stringify({ overall: metrics, metadata: { model_package_id: packageId } }, null, 2),
    "release_decision.json": JSON.stringify({
      proposal_id: `proposal_${packageId}`,
      decision: "pending",
      candidate_package_id: packageId,
      previous_champion_package_id: "not_set",
      rollback_package_id: "not_set",
      approver: "not_approved",
      reason: "pending",
      decided_at_utc: "2026-05-05T00:00:00Z",
      artifact_hashes_sha256: "pending"
    }, null, 2),
    "validation_report.json": JSON.stringify({ gate_results: [], validation_status: "pending" }, null, 2),
    "model.joblib": "model",
    "model_card.md": "card",
    "training_manifest.json": "{}",
    "data_manifest.json": "{}",
    "feature_contract.json": "{}",
    "slice_scorecard.csv": "segment,n\nALL,1\n",
    "temporal_scorecard.csv": "period,n\n2024Q4,1\n",
    "drift_report.json": "{}",
    "explainability_manifest.json": "{}"
  };
  for (const [name, content] of Object.entries(files)) {
    await writeFile(path.join(packageDir, name), `${content}\n`, "utf-8");
  }
  const hashes: Record<string, string> = {};
  for (const name of Object.keys(files)) {
    hashes[name] = await sha256File(path.join(packageDir, name));
  }
  await writeFile(path.join(packageDir, "artifact_hashes.json"), `${JSON.stringify({ algorithm: "sha256", files: hashes }, null, 2)}\n`, "utf-8");
}

async function sha256File(filePath: string) {
  const digest = createHash("sha256");
  digest.update(await readFile(filePath));
  return digest.digest("hex");
}
