import { promises as fs } from "node:fs";
import path from "node:path";

export type ModelAliasMode = "champion" | "candidate" | "challenger";

export type PackageSelectionSource =
  | "champion_alias"
  | "latest_approved_package"
  | "latest_candidate_package"
  | "missing";

export type ResolvedModelPackageSelection = {
  requestedAlias: ModelAliasMode;
  effectiveAlias: ModelAliasMode;
  source: PackageSelectionSource;
  packageId: string | null;
  packagePath: string | null;
  championAliasPath: string;
  registeredModelName: string;
  fallbackReason: string | null;
};

type ChampionAliasFile = {
  champion_package_id?: string | null;
  champion_package_path?: string | null;
};

type ReleaseDecision = {
  decision?: string | null;
};

const REGISTERED_MODEL_NAME = "spec-nyc-avm";
const CHAMPION_ALIAS_PATH = `models/packages/aliases/${REGISTERED_MODEL_NAME}.json`;

export async function resolveModelPackageSelection(
  repoRoot: string,
  requestedAlias: ModelAliasMode = "champion"
): Promise<ResolvedModelPackageSelection> {
  const champion = await resolveChampionPackage(repoRoot);

  if (requestedAlias === "champion") {
    if (champion) {
      return selection({
        requestedAlias,
        effectiveAlias: "champion",
        source: "champion_alias",
        packagePath: champion.packagePath,
        fallbackReason: null
      });
    }

    const approved = await latestPackageDirectory(repoRoot, { requireApproved: true });
    if (approved) {
      return selection({
        requestedAlias,
        effectiveAlias: "champion",
        source: "latest_approved_package",
        packagePath: approved,
        fallbackReason: `No champion alias exists at ${CHAMPION_ALIAS_PATH}; using latest approved package.`
      });
    }

    return selection({
      requestedAlias,
      effectiveAlias: "champion",
      source: "missing",
      packagePath: null,
      fallbackReason: `No champion alias exists at ${CHAMPION_ALIAS_PATH} and no approved package was found.`
    });
  }

  const latest = await latestPackageDirectory(repoRoot, { requireApproved: false });
  return selection({
    requestedAlias,
    effectiveAlias: requestedAlias,
    source: latest ? "latest_candidate_package" : "missing",
    packagePath: latest,
    fallbackReason: latest ? null : "No candidate package was found under models/packages."
  });
}

export async function resolveDashboardPackageSelection(repoRoot: string): Promise<ResolvedModelPackageSelection> {
  const champion = await resolveModelPackageSelection(repoRoot, "champion");
  if (champion.packagePath) {
    return champion;
  }

  const candidate = await resolveModelPackageSelection(repoRoot, "candidate");
  if (!candidate.packagePath) {
    return candidate;
  }

  return {
    ...candidate,
    fallbackReason: champion.fallbackReason ?? "Dashboard is showing a candidate package because no champion exists."
  };
}

async function resolveChampionPackage(repoRoot: string): Promise<{ packagePath: string } | null> {
  const alias = await readJson<ChampionAliasFile>(path.join(repoRoot, CHAMPION_ALIAS_PATH));
  const aliasPath = alias?.champion_package_path || (
    alias?.champion_package_id ? `models/packages/${alias.champion_package_id}` : null
  );

  if (!aliasPath) {
    return null;
  }

  if (!(await existsAt(path.join(repoRoot, aliasPath, "model.joblib")))) {
    return null;
  }

  return { packagePath: aliasPath };
}

async function latestPackageDirectory(
  repoRoot: string,
  options: { requireApproved: boolean }
): Promise<string | null> {
  const root = path.join(repoRoot, "models/packages");
  let entries;
  try {
    entries = await fs.readdir(root, { withFileTypes: true });
  } catch {
    return null;
  }

  const candidates = await Promise.all(
    entries
      .filter((entry) => entry.isDirectory() && entry.name.startsWith("spec_nyc_avm_"))
      .map(async (entry) => {
        const rel = path.join("models/packages", entry.name);
        const abs = path.join(repoRoot, rel);
        const stat = await fs.stat(abs);
        const release = await readJson<ReleaseDecision>(path.join(abs, "release_decision.json"));
        const approved = String(release?.decision ?? "").toLowerCase() === "approved";
        return { rel, mtimeMs: stat.mtimeMs, approved };
      })
  );

  const filtered = options.requireApproved ? candidates.filter((item) => item.approved) : candidates;
  filtered.sort((a, b) => b.mtimeMs - a.mtimeMs);
  return filtered[0]?.rel ?? null;
}

function selection({
  requestedAlias,
  effectiveAlias,
  source,
  packagePath,
  fallbackReason
}: {
  requestedAlias: ModelAliasMode;
  effectiveAlias: ModelAliasMode;
  source: PackageSelectionSource;
  packagePath: string | null;
  fallbackReason: string | null;
}): ResolvedModelPackageSelection {
  return {
    requestedAlias,
    effectiveAlias,
    source,
    packageId: packagePath ? path.basename(packagePath) : null,
    packagePath,
    championAliasPath: CHAMPION_ALIAS_PATH,
    registeredModelName: REGISTERED_MODEL_NAME,
    fallbackReason
  };
}

async function readJson<T>(filePath: string): Promise<T | null> {
  try {
    return JSON.parse(await fs.readFile(filePath, "utf-8")) as T;
  } catch {
    return null;
  }
}

async function existsAt(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}
