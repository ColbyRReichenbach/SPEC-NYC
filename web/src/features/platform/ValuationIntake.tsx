"use client";

import { useState } from "react";
import { ClipboardCheck, ShieldCheck } from "lucide-react";

import type { PlatformData } from "@/src/features/platform/data";
import { DEFAULT_PLATFORM_OPTIONS } from "@/src/features/platform/platformOptions";
import type { SingleValuationResponse } from "@/src/features/valuation/schemas/valuationSchemas";

type IntakeState = {
  borough: string;
  segment: string;
  grossSquareFeet: string;
  yearBuilt: string;
  residentialUnits: string;
  totalUnits: string;
  modelAlias: "champion" | "candidate";
};

const initialState: IntakeState = {
  borough: "BROOKLYN",
  segment: "SMALL_MULTI",
  grossSquareFeet: "1850",
  yearBuilt: "1931",
  residentialUnits: "2",
  totalUnits: "2",
  modelAlias: "candidate"
};

const requiredFields: Array<keyof IntakeState> = [
  "borough",
  "segment",
  "grossSquareFeet",
  "yearBuilt",
  "residentialUnits",
  "totalUnits"
];

export function ValuationIntake({ data }: { data: PlatformData }) {
  const defaultAlias = data.package.status === "approved" ? "champion" : "candidate";
  const [state, setState] = useState<IntakeState>({ ...initialState, modelAlias: defaultAlias });
  const [result, setResult] = useState<SingleValuationResponse | null>(null);
  const [requestStatus, setRequestStatus] = useState<"idle" | "running" | "complete" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const options = data.options ?? DEFAULT_PLATFORM_OPTIONS;
  const boroughOptions = options.valuation.boroughs.length
    ? options.valuation.boroughs
    : uniqueSorted(data.eda.segmentRegion.map((row) => row.borough));
  const segmentOptions = uniqueSorted([
    ...data.package.segmentMetrics.map((row) => row.name),
    ...data.eda.segmentRegion.map((row) => row.propertySegment)
  ]).filter((segment) => segment !== "ALL" && segment !== "unknown");

  const filled = requiredFields.filter((field) => String(state[field]).trim().length > 0).length;
  const completeness = Math.round((filled / requiredFields.length) * 100);
  const featureHash = featureVectorHash(state);
  const canScore = data.package.status === "approved" && data.release.allGreen;
  const statusLabel = canScore ? "Approved scoring path" : "Candidate research scoring";

  async function evaluateRequest() {
    setRequestStatus("running");
    setError(null);
    setResult(null);

    const payload = {
      property: {
        address: `${state.borough} governed intake`,
        borough: state.borough,
        gross_square_feet: Number(state.grossSquareFeet),
        year_built: Number(state.yearBuilt),
        residential_units: Number(state.residentialUnits),
        total_units: Number(state.totalUnits),
        building_class: "B1",
        property_segment: state.segment,
        sale_date: data.etl.latestSaleDate === "unknown" ? new Date().toISOString().slice(0, 10) : data.etl.latestSaleDate
      },
      context: {
        dataset_version: data.package.datasetVersion,
        model_alias: state.modelAlias
      }
    };

    try {
      const response = await fetch("/api/v1/valuations/single", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const body = (await response.json()) as SingleValuationResponse & { error?: string };
      if (!response.ok) {
        throw new Error(body.error ?? "Valuation request failed.");
      }
      setResult(body);
      setRequestStatus("complete");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Valuation request failed.");
      setRequestStatus("error");
    }
  }

  return (
    <section className="panel valuation-panel" aria-labelledby="valuation-intake-title">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Valuation Intake</span>
          <h2 id="valuation-intake-title">Governed single-property request</h2>
        </div>
        <span className={`status-pill ${canScore ? "pass" : "warn"}`}>
          <ShieldCheck size={14} aria-hidden="true" />
          {statusLabel}
        </span>
      </div>

      <div className="intake-grid">
        <label>
          Borough
          <select value={state.borough} onChange={(event) => setState({ ...state, borough: event.target.value })}>
            {boroughOptions.map((borough) => (
              <option key={borough}>{borough}</option>
            ))}
          </select>
        </label>
        <label>
          Segment
          <select value={state.segment} onChange={(event) => setState({ ...state, segment: event.target.value })}>
            {(segmentOptions.length ? segmentOptions : ["SINGLE_FAMILY", "WALKUP", "ELEVATOR", "SMALL_MULTI"]).map((segment) => (
              <option key={segment}>{segment}</option>
            ))}
          </select>
        </label>
        <label>
          Model alias
          <select value={state.modelAlias} onChange={(event) => setState({ ...state, modelAlias: event.target.value as IntakeState["modelAlias"] })}>
            {options.valuation.model_aliases.map((alias) => (
              <option key={alias} value={alias}>
                {alias}
              </option>
            ))}
          </select>
        </label>
        <label>
          Gross square feet
          <input
            value={state.grossSquareFeet}
            onChange={(event) => setState({ ...state, grossSquareFeet: event.target.value })}
            inputMode="numeric"
          />
        </label>
        <label>
          Year built
          <input
            value={state.yearBuilt}
            onChange={(event) => setState({ ...state, yearBuilt: event.target.value })}
            inputMode="numeric"
          />
        </label>
        <label>
          Residential units
          <input
            value={state.residentialUnits}
            onChange={(event) => setState({ ...state, residentialUnits: event.target.value })}
            inputMode="numeric"
          />
        </label>
        <label>
          Total units
          <input
            value={state.totalUnits}
            onChange={(event) => setState({ ...state, totalUnits: event.target.value })}
            inputMode="numeric"
          />
        </label>
      </div>

      <div className="valuation-action-row">
        <button className="command-button" type="button" onClick={evaluateRequest} disabled={requestStatus === "running"}>
          <ClipboardCheck size={16} aria-hidden="true" />
          {requestStatus === "running" ? "Scoring" : "Evaluate Request"}
        </button>
        <div className="feature-hash">
          <span>Feature vector hash</span>
          <code>{featureHash}</code>
        </div>
      </div>

      <div className="decision-output" aria-live="polite">
        <div>
          <span className="eyebrow">Request Decision</span>
          <strong>
            {result
              ? formatCurrency(result.predicted_price)
              : requestStatus === "error"
                ? "Scoring failed"
                : "Awaiting request"}
          </strong>
          <p>
            {result
              ? `${result.model.alias} package ${data.package.id} scored route ${result.model.route}.`
              : error ?? `${state.modelAlias} scoring will use the resolved package contract and persist a valuation artifact.`}
          </p>
        </div>
        <div className="quality-meter" aria-label={`Input completeness ${completeness}%`}>
          <span>Completeness</span>
          <strong>{result ? `${Math.round(result.confidence.score * 100)}%` : `${completeness}%`}</strong>
          <div>
            <i style={{ width: `${result ? Math.round(result.confidence.score * 100) : completeness}%` }} />
          </div>
        </div>
      </div>
      {result ? (
        <div className="valuation-result-grid">
          <div>
            <span>Interval</span>
            <strong>{formatCurrency(result.prediction_interval.low)} - {formatCurrency(result.prediction_interval.high)}</strong>
          </div>
          <div>
            <span>Valuation artifact</span>
            <code>{result.valuation_id}</code>
          </div>
          <div>
            <span>Metrics</span>
            <code>{result.evidence.metrics_path}</code>
          </div>
          <div>
            <span>Top driver</span>
            <strong>{result.explanation.drivers_positive[0]?.display ?? result.explanation.drivers_negative[0]?.display ?? "No driver"}</strong>
          </div>
        </div>
      ) : null}
    </section>
  );
}

function featureVectorHash(state: IntakeState) {
  const raw = Object.values(state).join("|");
  let hash = 0;
  for (let index = 0; index < raw.length; index += 1) {
    hash = (hash << 5) - hash + raw.charCodeAt(index);
    hash |= 0;
  }
  return `fv_${Math.abs(hash).toString(16).padStart(8, "0")}`;
}

function formatCurrency(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0
  }).format(value);
}

function uniqueSorted(values: string[]) {
  return Array.from(new Set(values.filter(Boolean).map((value) => value.toUpperCase()))).sort();
}
