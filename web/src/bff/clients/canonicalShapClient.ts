import { spawn } from "node:child_process";

import type { SourceContext } from "@/src/bff/types/baseContracts";
import { resolveRepoRoot } from "@/src/bff/clients/artifactStore";
import { resolveDashboardPackageSelection } from "@/src/features/platform/packageResolver";

type ImportancePayload = {
  segment: string;
  window: string;
  features: Array<{
    feature_name: string;
    mean_abs_shap: number;
    direction_hint: "positive" | "negative" | "mixed";
  }>;
  generated_from: string[];
  model_package_id: string;
};

export async function buildCanonicalGlobalShapSummary(input: {
  segment: string;
  window: string;
}): Promise<{
  payload: {
    segment: string;
    window: string;
    features: Array<{
      feature_name: string;
      mean_abs_shap: number;
      direction_hint: "positive" | "negative" | "mixed";
    }>;
    generated_from: string[];
  };
  sourceContext: SourceContext;
}> {
  const repoRoot = await resolveRepoRoot();
  const selection = await resolveDashboardPackageSelection(repoRoot);
  if (!selection.packagePath) {
    return {
      payload: {
        segment: input.segment.toUpperCase(),
        window: input.window,
        features: [],
        generated_from: []
      },
      sourceContext: {
        source_id: selection.fallbackReason ?? "no_model_package",
        source_type: "other"
      }
    };
  }

  const importance = await runImportanceExtractor(repoRoot, selection.packagePath, input.segment, input.window);
  return {
    payload: {
      segment: importance.segment,
      window: importance.window,
      features: importance.features,
      generated_from: importance.generated_from
    },
    sourceContext: {
      source_id: `${selection.packagePath}|${importance.generated_from.join("|")}`,
      source_type: "other"
    }
  };
}

async function runImportanceExtractor(
  repoRoot: string,
  packagePath: string,
  segment: string,
  window: string
): Promise<ImportancePayload> {
  return new Promise((resolve, reject) => {
    const child = spawn(
      "python3",
      [
        "-m",
        "src.serving.score_single",
        "--repo-root",
        repoRoot,
        "--package-path",
        packagePath,
        "--mode",
        "global-importance",
        "--segment",
        segment,
        "--window",
        window
      ],
      { cwd: repoRoot, stdio: ["ignore", "pipe", "pipe"] }
    );
    let stdout = "";
    let stderr = "";
    const timeout = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error("Model importance extraction timed out."));
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
        reject(new Error(parseScorerError(stderr) ?? `Model importance extraction failed with exit code ${code}.`));
        return;
      }
      try {
        resolve(JSON.parse(stdout) as ImportancePayload);
      } catch (error) {
        reject(error instanceof Error ? error : new Error("Unable to parse importance output."));
      }
    });
  });
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
