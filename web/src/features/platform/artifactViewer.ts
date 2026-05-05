import { promises as fs } from "node:fs";
import path from "node:path";

import { parse } from "csv-parse/sync";

import { resolveRepoRoot } from "@/src/bff/clients/artifactStore";

export type ArtifactPreview =
  | {
      status: "missing" | "blocked";
      requestedPath: string;
      reason: string;
    }
  | {
      status: "available";
      requestedPath: string;
      path: string;
      title: string;
      kind: string;
      sizeBytes: number;
      updatedAt: string;
      content: string;
      notebookCells: NotebookCellPreview[];
      csvPreview: CsvPreview | null;
    };

export type NotebookCellPreview = {
  index: number;
  cellType: string;
  executionCount: string;
  source: string;
  outputs: string[];
};

export type CsvPreview = {
  columns: string[];
  rows: string[][];
  rowCount: number;
};

type NotebookCell = {
  cell_type?: string;
  execution_count?: number | string | null;
  source?: string | string[];
  outputs?: Array<{
    name?: string;
    text?: string | string[];
    data?: Record<string, string | string[]>;
  }>;
};

type NotebookDocument = {
  cells?: NotebookCell[];
};

const MAX_TEXT_PREVIEW_BYTES = 2_500_000;
const MAX_CSV_ROWS = 100;
const ALLOWED_ARTIFACT_ROOTS = ["reports", "models", "docs", "config", "notebooks"];

export async function loadArtifactPreview(rawPath: string): Promise<ArtifactPreview> {
  const requestedPath = rawPath.trim();
  const safePath = normalizeArtifactPath(requestedPath);
  if (!safePath) {
    return {
      status: "blocked",
      requestedPath,
      reason: "Artifact path is outside the governed repository artifact roots."
    };
  }

  const repoRoot = await resolveRepoRoot();
  const fullPath = path.resolve(repoRoot, safePath);
  const repoRootWithSep = repoRoot.endsWith(path.sep) ? repoRoot : `${repoRoot}${path.sep}`;
  if (!fullPath.startsWith(repoRootWithSep)) {
    return {
      status: "blocked",
      requestedPath,
      reason: "Artifact path resolves outside the repository."
    };
  }

  let stat;
  try {
    stat = await fs.stat(fullPath);
  } catch {
    return {
      status: "missing",
      requestedPath,
      reason: "Artifact was referenced but no file exists at that path."
    };
  }

  if (!stat.isFile()) {
    return {
      status: "blocked",
      requestedPath,
      reason: "Artifact preview only supports files."
    };
  }

  const content = await readPreview(fullPath);
  const kind = artifactKind(safePath);
  return {
    status: "available",
    requestedPath,
    path: safePath,
    title: artifactTitle(safePath),
    kind,
    sizeBytes: stat.size,
    updatedAt: stat.mtime.toISOString(),
    content,
    notebookCells: kind === "notebook" ? parseNotebook(content) : [],
    csvPreview: kind === "csv" ? parseCsvPreview(content) : null
  };
}

function normalizeArtifactPath(rawPath: string) {
  if (!rawPath || path.isAbsolute(rawPath)) {
    return null;
  }
  const normalized = path.posix.normalize(rawPath.replaceAll("\\", "/"));
  if (normalized === "." || normalized.startsWith("../") || normalized.includes("/../")) {
    return null;
  }
  return ALLOWED_ARTIFACT_ROOTS.some((root) => normalized === root || normalized.startsWith(`${root}/`))
    ? normalized
    : null;
}

async function readPreview(fullPath: string) {
  const handle = await fs.open(fullPath, "r");
  try {
    const buffer = Buffer.alloc(MAX_TEXT_PREVIEW_BYTES);
    const { bytesRead } = await handle.read(buffer, 0, MAX_TEXT_PREVIEW_BYTES, 0);
    return buffer.subarray(0, bytesRead).toString("utf-8");
  } finally {
    await handle.close();
  }
}

function artifactKind(relPath: string) {
  const ext = path.extname(relPath).toLowerCase();
  if (ext === ".ipynb") {
    return "notebook";
  }
  if (ext === ".html" || ext === ".htm") {
    return "html";
  }
  if (ext === ".md" || ext === ".markdown") {
    return "markdown";
  }
  if (ext === ".json") {
    return "json";
  }
  if (ext === ".csv") {
    return "csv";
  }
  return ext.replace(".", "") || "artifact";
}

function artifactTitle(relPath: string) {
  const basename = path.basename(relPath, path.extname(relPath));
  return basename
    .split(/[_\-\s]+/)
    .filter(Boolean)
    .map((word) => `${word.charAt(0).toUpperCase()}${word.slice(1)}`)
    .join(" ");
}

function parseNotebook(content: string): NotebookCellPreview[] {
  let parsed: NotebookDocument;
  try {
    parsed = JSON.parse(content) as NotebookDocument;
  } catch {
    return [];
  }

  if (!Array.isArray(parsed.cells)) {
    return [];
  }

  return parsed.cells.slice(0, 80).map((cell, index) => ({
    index,
    cellType: String(cell.cell_type ?? "cell"),
    executionCount: cell.execution_count == null ? "" : String(cell.execution_count),
    source: normalizeText(cell.source),
    outputs: Array.isArray(cell.outputs)
      ? cell.outputs
          .map((output) => normalizeText(output.text ?? output.data?.["text/plain"] ?? ""))
          .filter(Boolean)
          .slice(0, 4)
      : []
  }));
}

function parseCsvPreview(content: string): CsvPreview | null {
  try {
    const records = parse(content, {
      columns: true,
      skip_empty_lines: true,
      trim: true,
      relax_column_count: true
    }) as Record<string, string>[];
    const columns = Object.keys(records[0] ?? {}).slice(0, 12);
    return {
      columns,
      rows: records.slice(0, MAX_CSV_ROWS).map((record) => columns.map((column) => String(record[column] ?? ""))),
      rowCount: records.length
    };
  } catch {
    return null;
  }
}

function normalizeText(value: string | string[] | undefined) {
  if (Array.isArray(value)) {
    return value.join("");
  }
  return value ?? "";
}
