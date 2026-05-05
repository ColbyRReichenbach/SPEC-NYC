import { describe, expect, it } from "vitest";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import type { PlatformData } from "@/src/features/platform/data";
import {
  buildExperimentRunBundle,
  queueExperimentTraining,
  recordExperimentReviewDecision,
  requestExperimentReview,
  validateExperimentRequest,
  writeExperimentRunBundle
} from "@/src/features/platform/experimentRegistry";

const platformData: PlatformData = {
  generatedAt: "2026-05-05T00:00:00.000Z",
  package: {
    id: "spec_nyc_avm_v2_test",
    path: "models/packages/spec_nyc_avm_v2_test",
    decision: "pending",
    status: "candidate",
    modelVersion: "v2",
    modelStrategy: "global",
    datasetVersion: "nyc_open_data_2019_2024_test",
    featureContractVersion: "fc_test",
    trainRows: 236365,
    testRows: 59092,
    ppe10: 0.17,
    mdape: 0.31,
    r2: 0.29,
    featureCount: 15,
    featureDriftAlerts: 0,
    featureDriftWarnings: 0,
    routerColumns: [],
    featureExamples: ["gross_square_feet"],
    artifactCount: 12,
    minSaleDate: "2019-01-01",
    maxSaleDate: "2024-12-31",
    dataSnapshotSha256: "abc123snapshot",
    artifactHashes: [],
    modelCardPreview: "model card",
    limitations: [],
    comps: {
      manifestPath: "models/packages/spec_nyc_avm_v2_test/comps_manifest.json",
      selectedCompsPath: "models/packages/spec_nyc_avm_v2_test/selected_comps.csv",
      highErrorReviewPath: "models/packages/spec_nyc_avm_v2_test/high_error_review_sample.csv",
      featureNames: ["comp_count"],
      selectedRows: 8,
      highErrorRows: 1,
      testNoCompRate: 0,
      testSparseCompRate: 0,
      sample: []
    },
    training: {
      command: "./src/model.py",
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
    sources: [{ name: "training_source", uri: "data/processed/training.csv", rowCount: 295457 }],
    segmentMetrics: [],
    tierMetrics: []
  },
  etl: {
    reportPath: "reports/data/etl.csv",
    rawRows: 498666,
    transformedRows: 295457,
    uniqueProperties: 256430,
    latestSaleDate: "2024-12-31"
  },
  release: {
    reportPath: "reports/validation/v1_readiness_report.json",
    mode: "production",
    allGreen: false,
    gates: [],
    checks: [],
    blockers: [{ name: "production_model_evidence", status: "fail", detail: "pending approval" }]
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
  hypothesis: "A segment specialist should improve single-family MdAPE.",
  expectedEffect: "Lower MdAPE without reducing PPE10.",
  segment: "SINGLE_FAMILY",
  primaryMetric: "MdAPE",
  modelFamily: "XGBoost segment specialist",
  validationPlan: "Time split + borough/segment slices",
  trialBudget: 8,
  riskReview: true
};

describe("experiment registry contract", () => {
  it("builds locked run artifacts with shared dataset and comparison signatures", () => {
    const result = buildExperimentRunBundle({
      payload: request,
      data: platformData,
      createdAt: "2026-05-05T00:00:00.000Z"
    });

    expect(result.bundle.spec_locked).toBe(true);
    expect(result.bundle.artifact_paths.experiment_spec).toContain("experiment_spec.json");
    expect(result.bundle.artifact_paths.dataset_snapshot).toContain("dataset_snapshot.json");
    expect(result.bundle.artifact_paths.comparison_report).toContain("comparison_report.json");
    expect(result.bundle.dataset_snapshot.split_signature_sha256).toHaveLength(64);
    expect(result.bundle.comparison.same_dataset_required).toBe(true);
    expect(result.bundle.comparison.split_signature_sha256).toBe(result.bundle.dataset_snapshot.split_signature_sha256);
    expect(result.bundle.run_plan.row_selection_policy).toContain("identical_dataset_snapshot");
    expect(result.auditEvents.map((event) => event.event)).toEqual([
      "experiment_spec_locked",
      "dataset_snapshot_bound",
      "comparison_contract_created"
    ]);
  });

  it("keeps spec hashes deterministic for the same locked request", () => {
    const first = buildExperimentRunBundle({ payload: request, data: platformData, createdAt: "2026-05-05T00:00:00.000Z" });
    const second = buildExperimentRunBundle({ payload: request, data: platformData, createdAt: "2026-05-05T00:00:00.000Z" });

    expect(first.bundle.id).toBe(second.bundle.id);
    expect(first.bundle.spec_hash).toBe(second.bundle.spec_hash);
  });

  it("rejects underspecified experiment requests before artifact creation", () => {
    expect(validateExperimentRequest({ hypothesis: "too short" })).toBe("Hypothesis must be at least 12 characters.");
    expect(validateExperimentRequest({ ...request, trialBudget: 0 })).toBe("Trial budget must be a positive number.");
  });

  it("moves a locked spec through review approval and training queue artifacts", async () => {
    const repoRoot = await makeRepoFixture();
    const result = buildExperimentRunBundle({
      payload: request,
      data: platformData,
      createdAt: "2026-05-05T00:00:00.000Z"
    });

    await writeExperimentRunBundle(repoRoot, result);
    const requested = await requestExperimentReview(repoRoot, result.bundle.id, { reason: "Ready for review." });
    const approved = await recordExperimentReviewDecision(repoRoot, result.bundle.id, {
      decision: "approved",
      reviewer: "fixture_reviewer",
      reason: "Approved for controlled challenger training."
    });
    const queued = await queueExperimentTraining(repoRoot, result.bundle.id, { reason: "Queue test job." });

    expect(requested.status).toBe("review_requested");
    expect(approved.status).toBe("review_approved");
    expect(queued.status).toBe("queued");
    expect(queued.training_job?.command_display).toContain("python3 -m src.model");
    expect(queued.training_job?.training_plan.trainer_strategy).toBe("segmented_router");

    const jobManifest = await readFile(path.join(repoRoot, queued.artifact_paths.job_manifest ?? ""), "utf-8");
    const auditLog = await readFile(path.join(repoRoot, queued.artifact_paths.audit_log), "utf-8");

    expect(jobManifest).toContain(queued.training_job?.id as string);
    expect(auditLog).toContain("review_requested");
    expect(auditLog).toContain("review_approved");
    expect(auditLog).toContain("training_job_queued");
  });

  it("blocks approval for research ideas that do not have a trainer adapter", async () => {
    const repoRoot = await makeRepoFixture();
    const result = buildExperimentRunBundle({
      payload: { ...request, modelFamily: "Neural tabular prototype" },
      data: platformData,
      createdAt: "2026-05-05T00:00:00.000Z"
    });

    await writeExperimentRunBundle(repoRoot, result);
    await requestExperimentReview(repoRoot, result.bundle.id, { reason: "Ready for review." });

    await expect(
      recordExperimentReviewDecision(repoRoot, result.bundle.id, {
        decision: "approved",
        reviewer: "fixture_reviewer",
        reason: "Approved for controlled challenger training."
      })
    ).rejects.toThrow("Review approval blocked by policy check: trainer_supported.");
  });
});

async function makeRepoFixture() {
  const repoRoot = await mkdtemp(path.join(os.tmpdir(), "spec-experiment-registry-"));
  await mkdir(path.join(repoRoot, "reports/experiments/runs"), { recursive: true });
  await mkdir(path.join(repoRoot, "data/processed"), { recursive: true });
  await mkdir(path.join(repoRoot, "models/packages"), { recursive: true });
  await mkdir(path.join(repoRoot, "config"), { recursive: true });
  await writeFile(path.join(repoRoot, "data/processed/training.csv"), "sale_date,sale_price\n2024-01-01,100000\n", "utf-8");
  return repoRoot;
}
