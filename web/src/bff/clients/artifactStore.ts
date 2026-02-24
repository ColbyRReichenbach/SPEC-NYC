import { promises as fs } from "node:fs";
import path from "node:path";

const REQUIRED_ROOT_MARKERS = ["reports", "models", "config"];

async function existsAt(target: string): Promise<boolean> {
  try {
    await fs.access(target);
    return true;
  } catch {
    return false;
  }
}

export async function resolveRepoRoot(): Promise<string> {
  const cwd = process.cwd();
  const candidates = [cwd, path.resolve(cwd, "..")];

  for (const candidate of candidates) {
    const checks = await Promise.all(
      REQUIRED_ROOT_MARKERS.map((marker) => existsAt(path.join(candidate, marker)))
    );
    if (checks.every(Boolean)) {
      return candidate;
    }
  }

  throw new Error("Unable to resolve repository root for artifact access.");
}

export async function readJsonArtifact<T>(repoRelativePath: string): Promise<T | null> {
  try {
    const root = await resolveRepoRoot();
    const fullPath = path.join(root, repoRelativePath);
    const raw = await fs.readFile(fullPath, "utf-8");
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export async function readTextArtifact(repoRelativePath: string): Promise<string | null> {
  try {
    const root = await resolveRepoRoot();
    const fullPath = path.join(root, repoRelativePath);
    return await fs.readFile(fullPath, "utf-8");
  } catch {
    return null;
  }
}

export async function readTextArtifactHead(
  repoRelativePath: string,
  maxBytes = 2_500_000
): Promise<string | null> {
  const root = await resolveRepoRoot();
  const fullPath = path.join(root, repoRelativePath);

  let handle;
  try {
    handle = await fs.open(fullPath, "r");
    const buffer = Buffer.alloc(maxBytes);
    const { bytesRead } = await handle.read(buffer, 0, maxBytes, 0);
    return buffer.subarray(0, bytesRead).toString("utf-8");
  } catch {
    return null;
  } finally {
    await handle?.close().catch(() => null);
  }
}

export async function latestArtifactPath(
  repoRelativeDir: string,
  predicate: (name: string) => boolean
): Promise<string | null> {
  const root = await resolveRepoRoot();
  const dirPath = path.join(root, repoRelativeDir);

  let entries;
  try {
    entries = await fs.readdir(dirPath, { withFileTypes: true });
  } catch {
    return null;
  }

  const matched = entries
    .filter((entry) => entry.isFile() && predicate(entry.name))
    .map((entry) => entry.name);

  if (matched.length === 0) {
    return null;
  }

  const withStats = await Promise.all(
    matched.map(async (name) => {
      const rel = path.join(repoRelativeDir, name);
      const stat = await fs.stat(path.join(root, rel));
      return { rel, mtimeMs: stat.mtimeMs };
    })
  );

  withStats.sort((a, b) => b.mtimeMs - a.mtimeMs);
  return withStats[0]?.rel ?? null;
}
