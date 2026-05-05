"use client";

import { useState } from "react";
import { ClipboardCheck, ShieldCheck } from "lucide-react";

import type { PlatformData } from "@/src/features/platform/data";

type IntakeState = {
  borough: string;
  segment: string;
  grossSquareFeet: string;
  yearBuilt: string;
  residentialUnits: string;
  totalUnits: string;
};

const initialState: IntakeState = {
  borough: "Brooklyn",
  segment: "SMALL_MULTI",
  grossSquareFeet: "1850",
  yearBuilt: "1931",
  residentialUnits: "2",
  totalUnits: "2"
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
  const [state, setState] = useState<IntakeState>(initialState);
  const [submitted, setSubmitted] = useState(false);

  const filled = requiredFields.filter((field) => String(state[field]).trim().length > 0).length;
  const completeness = Math.round((filled / requiredFields.length) * 100);
  const featureHash = featureVectorHash(state);
  const canScore = data.package.status === "approved" && data.release.allGreen;
  const statusLabel = canScore ? "Approved scoring path" : "Governed no-score";

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
            <option>Manhattan</option>
            <option>Bronx</option>
            <option>Brooklyn</option>
            <option>Queens</option>
            <option>Staten Island</option>
          </select>
        </label>
        <label>
          Segment
          <select value={state.segment} onChange={(event) => setState({ ...state, segment: event.target.value })}>
            <option>SINGLE_FAMILY</option>
            <option>WALKUP</option>
            <option>ELEVATOR</option>
            <option>SMALL_MULTI</option>
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
        <button className="command-button" type="button" onClick={() => setSubmitted(true)}>
          <ClipboardCheck size={16} aria-hidden="true" />
          Evaluate Request
        </button>
        <div className="feature-hash">
          <span>Feature vector hash</span>
          <code>{featureHash}</code>
        </div>
      </div>

      <div className="decision-output" aria-live="polite">
        <div>
          <span className="eyebrow">Request Decision</span>
          <strong>{submitted ? (canScore ? "Ready for model-backed scoring" : "Blocked before price generation") : "Awaiting request"}</strong>
          <p>
            {canScore
              ? `The request can use approved package ${data.package.id}.`
              : "No price is generated because the current package is pending approval or release gates are blocked."}
          </p>
        </div>
        <div className="quality-meter" aria-label={`Input completeness ${completeness}%`}>
          <span>Completeness</span>
          <strong>{completeness}%</strong>
          <div>
            <i style={{ width: `${completeness}%` }} />
          </div>
        </div>
      </div>
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
