import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";

import type { SourceContext } from "@/src/bff/types/baseContracts";
import { resolveRepoRoot } from "@/src/bff/clients/artifactStore";
import type { SingleValuationRequest } from "@/src/features/valuation/schemas/valuationSchemas";
import {
  resolveModelPackageSelection,
  type ModelAliasMode
} from "@/src/features/platform/packageResolver";

type Driver = {
  feature: string;
  impact: number;
  display: string;
};

type ScorerPayload = {
  predicted_price: number;
  prediction_interval: { low: number; high: number; method: string };
  confidence: {
    score: number;
    band: "high" | "medium" | "low";
    factors: {
      segment_calibration: number;
      support_coverage: number;
      input_completeness: number;
    };
    caveats: string[];
  };
  explanation: {
    status: "ready" | "degraded" | "unavailable";
    explainer_type: string;
    local_accuracy?: number | null;
    drivers_positive: Driver[];
    drivers_negative: Driver[];
  };
  model: {
    run_id: string;
    model_version: string;
    route: string;
    model_package_id: string;
  };
  evidence: {
    model_package_path: string;
    metrics_path: string;
    feature_contract_path: string;
    data_manifest_path: string;
    model_card_path: string;
    feature_importance_artifact: string;
    training_source_path: string | null;
    feature_vector_sha256: string;
    missing_features: string[];
    feature_generation: Record<string, unknown>;
  };
};

export async function buildCanonicalValuationResponse(input: SingleValuationRequest): Promise<{
  payload: {
    valuation_id: string;
    predicted_price: number;
    prediction_interval: { low: number; high: number; method: string };
    confidence: {
      score: number;
      band: "high" | "medium" | "low";
      factors: {
        segment_calibration: number;
        support_coverage: number;
        input_completeness: number;
      };
      caveats: string[];
    };
    explanation: {
      status: "ready" | "degraded" | "unavailable";
      explainer_type: string;
      local_accuracy?: number;
      drivers_positive: Driver[];
      drivers_negative: Driver[];
    };
    model: {
      alias: "champion" | "challenger" | "candidate";
      run_id: string;
      model_version: string;
      route: string;
    };
    evidence: {
      run_card_path: string;
      metrics_path: string;
      shap_summary_path: string;
    };
  };
  sourceContext: SourceContext;
}> {
  const repoRoot = await resolveRepoRoot();
  const requestedAlias = input.context.model_alias as ModelAliasMode;
  const selection = await resolveModelPackageSelection(repoRoot, requestedAlias);
  if (!selection.packagePath) {
    throw new Error(selection.fallbackReason ?? `No ${requestedAlias} model package is available for scoring.`);
  }

  const scorer = await runPythonScorer(repoRoot, selection.packagePath, input);
  const valuationId = buildValuationId(input, scorer.evidence.feature_vector_sha256);
  const explanation = {
    ...scorer.explanation,
    local_accuracy: scorer.explanation.local_accuracy ?? undefined
  };

  const payload = {
    valuation_id: valuationId,
    predicted_price: scorer.predicted_price,
    prediction_interval: scorer.prediction_interval,
    confidence: scorer.confidence,
    explanation,
    model: {
      alias: requestedAlias,
      run_id: scorer.model.run_id,
      model_version: scorer.model.model_version,
      route: scorer.model.route
    },
    evidence: {
      run_card_path: scorer.evidence.model_card_path,
      metrics_path: scorer.evidence.metrics_path,
      shap_summary_path: scorer.evidence.feature_importance_artifact
    }
  };

  await persistValuation(repoRoot, {
    valuation_id: valuationId,
    created_at_utc: new Date().toISOString(),
    request: input,
    package_selection: selection,
    payload,
    scorer_evidence: scorer.evidence
  });

  return {
    payload,
    sourceContext: {
      source_id: `${selection.packagePath}|${scorer.evidence.feature_vector_sha256}`,
      source_type: "other"
    }
  };
}

export async function readStoredValuationExplanation(valuationId: string): Promise<{
  explanation: {
    status: "ready" | "degraded" | "unavailable";
    explainer_type: string;
    drivers_positive: Driver[];
    drivers_negative: Driver[];
  };
  sourceContext: SourceContext;
} | null> {
  const repoRoot = await resolveRepoRoot();
  const safeId = assertValuationId(valuationId);
  const recordPath = path.join(repoRoot, "reports/valuations", `${safeId}.json`);
  try {
    const record = JSON.parse(await fs.readFile(recordPath, "utf-8")) as {
      package_selection?: { packagePath?: string | null };
      scorer_evidence?: { feature_vector_sha256?: string };
      payload?: {
        explanation?: {
          status: "ready" | "degraded" | "unavailable";
          explainer_type: string;
          drivers_positive: Driver[];
          drivers_negative: Driver[];
        };
      };
    };
    if (!record.payload?.explanation) {
      return null;
    }
    return {
      explanation: record.payload.explanation,
      sourceContext: {
        source_id: `${record.package_selection?.packagePath ?? "unknown_package"}|${record.scorer_evidence?.feature_vector_sha256 ?? safeId}`,
        source_type: "other"
      }
    };
  } catch {
    return null;
  }
}

async function runPythonScorer(
  repoRoot: string,
  packagePath: string,
  input: SingleValuationRequest
): Promise<ScorerPayload> {
  return new Promise((resolve, reject) => {
    const child = spawn(
      "python3",
      ["-m", "src.serving.score_single", "--repo-root", repoRoot, "--package-path", packagePath],
      { cwd: repoRoot, stdio: ["pipe", "pipe", "pipe"] }
    );
    let stdout = "";
    let stderr = "";
    const timeout = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error("Model scorer timed out."));
    }, 30_000);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timeout);
      if (code !== 0) {
        reject(new Error(parseScorerError(stderr) ?? `Model scorer failed with exit code ${code}.`));
        return;
      }
      try {
        resolve(JSON.parse(stdout) as ScorerPayload);
      } catch (error) {
        reject(error instanceof Error ? error : new Error("Unable to parse model scorer output."));
      }
    });
    child.stdin.end(JSON.stringify(input));
  });
}

async function persistValuation(repoRoot: string, record: Record<string, unknown>) {
  const id = assertValuationId(String(record.valuation_id));
  const dir = path.join(repoRoot, "reports/valuations");
  await fs.mkdir(dir, { recursive: true });
  await fs.writeFile(path.join(dir, `${id}.json`), `${JSON.stringify(record, null, 2)}\n`);
  await fs.appendFile(
    path.join(dir, "audit_log.jsonl"),
    `${JSON.stringify({
      event: "valuation_scored",
      created_at_utc: record.created_at_utc,
      valuation_id: id,
      package_path: (record.package_selection as { packagePath?: string })?.packagePath,
      feature_vector_sha256: (record.scorer_evidence as { feature_vector_sha256?: string })?.feature_vector_sha256
    })}\n`
  );
}

function buildValuationId(input: SingleValuationRequest, featureVectorSha: string) {
  const digest = createHash("sha256")
    .update(JSON.stringify({ input, featureVectorSha, createdAt: new Date().toISOString() }))
    .digest("hex")
    .slice(0, 16);
  return `val_${digest}`;
}

function assertValuationId(value: string) {
  if (!/^val_[a-f0-9]{8,32}$/i.test(value)) {
    throw new Error("Invalid valuation id.");
  }
  return value;
}

function parseScorerError(stderr: string) {
  const trimmed = stderr.trim();
  if (!trimmed) {
    return null;
  }
  try {
    const parsed = JSON.parse(trimmed) as { error?: string };
    return parsed.error ?? trimmed;
  } catch {
    return trimmed;
  }
}
