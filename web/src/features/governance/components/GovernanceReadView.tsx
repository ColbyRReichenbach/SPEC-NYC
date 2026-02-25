"use client";

import { useEffect, useState } from "react";

import { fetchGovernanceStatus } from "@/src/features/governance/services/governanceApi";
import type { GovernanceStatusResponse } from "@/src/features/governance/schemas/governanceSchemas";

export default function GovernanceReadView() {
  const [data, setData] = useState<GovernanceStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchGovernanceStatus()
      .then((payload) => {
        if (!cancelled) setData(payload);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Unable to load governance data");
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return <div className="card"><p className="error-text">{error}</p></div>;
  }

  if (!data) {
    return <div className="card"><p>Loading governance state...</p></div>;
  }

  return (
    <section className="stack-lg fade-in-up">
      <div className="card">
        <h1>Model Governance</h1>
        <p className="muted">{data.status_reason}</p>
        <div className="alias-grid">
          {Object.entries(data.aliases).map(([key, value]) => (
            <div key={key} className="alias-card">
              <span className="context-label">{key}</span>
              <p><code>{value.model_version ?? "none"}</code></p>
              <p className="muted"><code>{value.run_id ?? "n/a"}</code></p>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h2>Latest Proposal</h2>
        <p>
          <strong>{data.latest_proposal.proposal_id}</strong> · status <strong>{data.latest_proposal.status}</strong>
        </p>
        <p className="muted">Created {new Date(data.latest_proposal.created_at_utc).toLocaleString()}</p>
      </div>

      <div className="card">
        <h2>Policy Gates</h2>
        <table className="table">
          <thead>
            <tr>
              <th>Gate</th>
              <th>Threshold</th>
              <th>Actual</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {data.gate_results.map((gate) => (
              <tr key={gate.gate_key}>
                <td>{gate.label}</td>
                <td><code>{gate.threshold}</code></td>
                <td>{gate.actual.toFixed(4)}</td>
                <td>
                  <span className={`pill ${gate.status}`}>{gate.status.toUpperCase()}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h2>Promotion Actions</h2>
        <p className="muted">Read-only mode: auth/RBAC is required before approve/reject is enabled.</p>
        <div className="button-row">
          <button className="secondary-btn" disabled>Approve Proposal</button>
          <button className="secondary-btn" disabled>Reject Proposal</button>
        </div>
      </div>
    </section>
  );
}
