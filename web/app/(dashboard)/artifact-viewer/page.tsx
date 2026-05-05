import Link from "next/link";
import type { CSSProperties } from "react";

import { loadArtifactPreview } from "@/src/features/platform/artifactViewer";

type ArtifactViewerPageProps = {
  searchParams?: {
    path?: string | string[];
  };
};

export default async function ArtifactViewerPage({ searchParams }: ArtifactViewerPageProps) {
  const artifactPath = Array.isArray(searchParams?.path) ? searchParams?.path[0] ?? "" : searchParams?.path ?? "";
  const artifact = await loadArtifactPreview(artifactPath);

  if (artifact.status !== "available") {
    return (
      <div className="view-stack">
        <section className="panel artifact-viewer-shell">
          <span className="eyebrow">Artifact Viewer</span>
          <h1>Artifact unavailable</h1>
          <p>{artifact.reason}</p>
          <code>{artifact.requestedPath || "missing path"}</code>
          <Link className="command-button secondary" href="/eda">
            Back to EDA
          </Link>
        </section>
      </div>
    );
  }

  return (
    <div className="view-stack">
      <section className="artifact-viewer-header">
        <div>
          <span className="eyebrow">Artifact Viewer</span>
          <h1>{artifact.title}</h1>
          <p>
            {artifact.kind} artifact generated at <code>{artifact.path}</code>
          </p>
        </div>
        <div className="artifact-viewer-meta">
          <div>
            <span>Size</span>
            <strong>{formatBytes(artifact.sizeBytes)}</strong>
          </div>
          <div>
            <span>Updated</span>
            <strong>{formatTimestamp(artifact.updatedAt)}</strong>
          </div>
          <Link className="command-button secondary" href="/eda">
            Back to EDA
          </Link>
        </div>
      </section>

      {artifact.kind === "notebook" ? <NotebookPreview cells={artifact.notebookCells} /> : null}
      {artifact.kind === "csv" ? <CsvPreviewTable artifact={artifact} /> : null}
      {artifact.kind === "html" ? <HtmlPreview content={artifact.content} /> : null}
      {artifact.kind !== "notebook" && artifact.kind !== "csv" && artifact.kind !== "html" ? (
        <TextPreview title={`${artifact.kind} preview`} content={formatPlainText(artifact.kind, artifact.content)} />
      ) : null}
    </div>
  );
}

function NotebookPreview({ cells }: { cells: Array<{ index: number; cellType: string; executionCount: string; source: string; outputs: string[] }> }) {
  return (
    <section className="panel artifact-viewer-shell">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Notebook</span>
          <h2>{cells.length} rendered cells</h2>
        </div>
      </div>
      {cells.length === 0 ? (
        <div className="empty-state compact">
          <strong>No notebook cells could be rendered.</strong>
          <p>The artifact exists, but the notebook JSON could not be parsed.</p>
        </div>
      ) : (
        <div className="notebook-cell-list">
          {cells.map((cell) => (
            <article className="notebook-cell" key={cell.index}>
              <div className="notebook-cell-meta">
                <span>{cell.cellType}</span>
                {cell.executionCount ? <code>In [{cell.executionCount}]</code> : null}
              </div>
              <pre>{cell.source || "(empty cell)"}</pre>
              {cell.outputs.length > 0 ? (
                <div className="notebook-output-list">
                  {cell.outputs.map((output, index) => (
                    <pre key={`${cell.index}-${index}`}>{output}</pre>
                  ))}
                </div>
              ) : null}
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

function CsvPreviewTable({
  artifact
}: {
  artifact: Extract<Awaited<ReturnType<typeof loadArtifactPreview>>, { status: "available" }>;
}) {
  if (!artifact.csvPreview) {
    return <TextPreview title="CSV preview" content={artifact.content} />;
  }

  return (
    <section className="panel artifact-viewer-shell">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">CSV Preview</span>
          <h2>{artifact.csvPreview.rowCount.toLocaleString()} rows loaded from preview window</h2>
        </div>
      </div>
      <div className="artifact-csv-table">
        <div className="artifact-csv-row header" style={{ "--artifact-column-count": artifact.csvPreview.columns.length } as CSSProperties}>
          {artifact.csvPreview.columns.map((column) => (
            <span key={column}>{column}</span>
          ))}
        </div>
        {artifact.csvPreview.rows.map((row, rowIndex) => (
          <div className="artifact-csv-row" key={rowIndex} style={{ "--artifact-column-count": artifact.csvPreview?.columns.length ?? 1 } as CSSProperties}>
            {row.map((value, valueIndex) => (
              <span key={`${rowIndex}-${valueIndex}`}>{value}</span>
            ))}
          </div>
        ))}
      </div>
    </section>
  );
}

function HtmlPreview({ content }: { content: string }) {
  return (
    <section className="panel artifact-viewer-shell">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">HTML Preview</span>
          <h2>Rendered artifact</h2>
        </div>
      </div>
      <iframe className="artifact-html-frame" sandbox="" srcDoc={content} title="Rendered artifact" />
    </section>
  );
}

function TextPreview({ title, content }: { title: string; content: string }) {
  return (
    <section className="panel artifact-viewer-shell">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Text Preview</span>
          <h2>{title}</h2>
        </div>
      </div>
      <pre className="artifact-text-preview">{content}</pre>
    </section>
  );
}

function formatPlainText(kind: string, content: string) {
  if (kind !== "json") {
    return content;
  }
  try {
    return JSON.stringify(JSON.parse(content), null, 2);
  } catch {
    return content;
  }
}

function formatTimestamp(value: string) {
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) {
    return value;
  }
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit"
  }).format(parsed);
}

function formatBytes(value: number) {
  if (!Number.isFinite(value) || value <= 0) {
    return "missing";
  }
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${Math.round(value / 102.4) / 10} KB`;
  }
  return `${Math.round(value / 104857.6) / 10} MB`;
}
