"use client";

import { useEffect, useState } from "react";

import { fetchMonitoringOverview } from "@/src/features/monitoring/services/monitoringApi";

export default function MonitoringReadView() {
  const [window, setWindow] = useState("30d");
  const [data, setData] = useState<Awaited<ReturnType<typeof fetchMonitoringOverview>> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchMonitoringOverview(window)
      .then((payload) => {
        if (!cancelled) {
          setData(payload);
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Unable to load monitoring data");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [window]);

  if (error) {
    return <div className="card"><p className="error-text">{error}</p></div>;
  }

  if (!data) {
    return <div className="card"><p>Loading monitoring snapshot...</p></div>;
  }

  return (
    <section className="stack-lg fade-in-up">
      <div className="card monitoring-header-card">
        <h1>Monitoring and Drift</h1>
        <p className="muted">Temporal stability and drift snapshot from canonical monitoring artifacts.</p>
        <label className="window-filter">
          Window
          <select value={window} onChange={(event) => setWindow(event.target.value)}>
            <option value="7d">7d</option>
            <option value="30d">30d</option>
            <option value="90d">90d</option>
          </select>
        </label>
      </div>

      <div className="kpi-grid">
        <div className="card kpi-card">
          <h2>Drift Status</h2>
          <p className="kpi-value">{data.drift_summary.status.toUpperCase()}</p>
          <p className="muted">Alerts: {data.drift_summary.alerts} · Warnings: {data.drift_summary.warnings}</p>
        </div>
        <div className="card kpi-card">
          <h2>PPE10</h2>
          <p className="kpi-value">{(data.performance_summary.overall.ppe10 * 100).toFixed(1)}%</p>
          <p className="muted">Overall window performance</p>
        </div>
        <div className="card kpi-card">
          <h2>MdAPE</h2>
          <p className="kpi-value">{(data.performance_summary.overall.mdape * 100).toFixed(1)}%</p>
          <p className="muted">Median absolute percentage error</p>
        </div>
      </div>

      <div className="card">
        <h2>Segment Slice Metrics</h2>
        <table className="table">
          <thead>
            <tr>
              <th>Slice</th>
              <th>Rows</th>
              <th>PPE10</th>
              <th>MdAPE</th>
              <th>R2</th>
            </tr>
          </thead>
          <tbody>
            {data.slice_metrics.map((slice) => (
              <tr key={slice.slice_key}>
                <td>{slice.slice_key}</td>
                <td>{slice.n}</td>
                <td>{(slice.ppe10 * 100).toFixed(1)}%</td>
                <td>{(slice.mdape * 100).toFixed(1)}%</td>
                <td>{slice.r2.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h2>Retrain Decision</h2>
        <p>
          Decision: <strong>{data.retrain_decision.decision.toUpperCase()}</strong> · Should retrain:
          <strong> {String(data.retrain_decision.should_retrain)}</strong>
        </p>
        <ul>
          {data.retrain_decision.reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
        {data.degraded ? (
          <p className="warn-text">Degraded mode: {data.warnings.join(" ")}</p>
        ) : null}
      </div>
    </section>
  );
}
