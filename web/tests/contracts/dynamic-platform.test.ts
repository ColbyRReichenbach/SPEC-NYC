import { spawn } from "node:child_process";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  resolveDashboardPackageSelection,
  resolveModelPackageSelection
} from "@/src/features/platform/packageResolver";

describe("dynamic platform package resolver", () => {
  it("uses champion alias when present", async () => {
    const repoRoot = await makePackageRepo();
    await writePackage(repoRoot, "champion_pkg", { approved: true, withModel: true });
    await mkdir(path.join(repoRoot, "models/packages/aliases"), { recursive: true });
    await writeFile(
      path.join(repoRoot, "models/packages/aliases/spec-nyc-avm.json"),
      JSON.stringify({ champion_package_id: "champion_pkg", champion_package_path: "models/packages/champion_pkg" }),
      "utf-8"
    );

    const selection = await resolveModelPackageSelection(repoRoot, "champion");

    expect(selection.source).toBe("champion_alias");
    expect(selection.packageId).toBe("champion_pkg");
    expect(selection.packagePath).toBe("models/packages/champion_pkg");
  });

  it("falls back to latest approved package before candidate for dashboard state", async () => {
    const repoRoot = await makePackageRepo();
    await writePackage(repoRoot, "spec_nyc_avm_v2_old", { approved: true });
    await writePackage(repoRoot, "spec_nyc_avm_v2_new", { approved: false });

    const champion = await resolveModelPackageSelection(repoRoot, "champion");
    const dashboard = await resolveDashboardPackageSelection(repoRoot);
    const candidate = await resolveModelPackageSelection(repoRoot, "candidate");

    expect(champion.source).toBe("latest_approved_package");
    expect(champion.packageId).toBe("spec_nyc_avm_v2_old");
    expect(dashboard.packageId).toBe("spec_nyc_avm_v2_old");
    expect(candidate.packageId).toBe("spec_nyc_avm_v2_new");
  });
});

describe("model-backed scoring boundary", () => {
  it("scores a property through the Python model package instead of a static TS heuristic", async () => {
    const repoRoot = path.resolve(process.cwd(), "..");
    const packagePath = "models/packages/spec_nyc_avm_v2_20260505T053609Z_b6538c8";
    const request = {
      property: {
        address: "123 Example St",
        borough: "BROOKLYN",
        gross_square_feet: 1850,
        year_built: 1931,
        residential_units: 2,
        total_units: 2,
        building_class: "B1",
        property_segment: "SMALL_MULTI",
        sale_date: "2025-05-01"
      },
      context: {
        dataset_version: "rows_96_date_2020-01-01_2025-03-15",
        model_alias: "candidate"
      }
    };

    const stdout = await runPythonScorer(repoRoot, packagePath, request);
    const scored = JSON.parse(stdout) as {
      predicted_price: number;
      evidence: { feature_vector_sha256: string; metrics_path: string };
      explanation: { drivers_positive: unknown[]; drivers_negative: unknown[]; explainer_type: string };
    };

    expect(scored.predicted_price).toBeGreaterThan(0);
    expect(scored.evidence.metrics_path).toContain("/metrics.json");
    expect(scored.evidence.feature_vector_sha256).toHaveLength(64);
    expect(scored.explanation.explainer_type).toContain("xgboost_pred_contribs");
    expect(scored.explanation.drivers_positive.length + scored.explanation.drivers_negative.length).toBeGreaterThan(0);
  }, 60_000);
});

async function makePackageRepo() {
  const repoRoot = await mkdtemp(path.join(os.tmpdir(), "spec-platform-resolver-"));
  await mkdir(path.join(repoRoot, "models/packages"), { recursive: true });
  await mkdir(path.join(repoRoot, "reports"), { recursive: true });
  await mkdir(path.join(repoRoot, "config"), { recursive: true });
  return repoRoot;
}

async function runPythonScorer(repoRoot: string, packagePath: string, request: unknown): Promise<string> {
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
      reject(new Error("Python scorer timed out."));
    }, 60_000);
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
        reject(new Error(stderr || `Python scorer exited with ${code}`));
        return;
      }
      resolve(stdout);
    });
    child.stdin.end(JSON.stringify(request));
  });
}

async function writePackage(repoRoot: string, packageId: string, options: { approved: boolean; withModel?: boolean }) {
  const packageDir = path.join(repoRoot, "models/packages", packageId);
  await mkdir(packageDir, { recursive: true });
  await writeFile(
    path.join(packageDir, "release_decision.json"),
    JSON.stringify({ decision: options.approved ? "approved" : "pending" }),
    "utf-8"
  );
  await writeFile(path.join(packageDir, "model.joblib"), options.withModel ? "model" : "candidate", "utf-8");
  await new Promise((resolve) => setTimeout(resolve, 5));
}
