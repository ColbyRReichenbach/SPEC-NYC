"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

import { fetchGlobalShapSummary } from "@/src/features/valuation/services/explainabilityApi";
import type { GlobalShapSummaryResponse } from "@/src/features/valuation/schemas/explainabilitySchemas";

type ShapGlobalSummaryProps = {
  segment: string;
  window?: string;
};

export default function ShapGlobalSummary({ segment, window = "180d" }: ShapGlobalSummaryProps) {
  const [data, setData] = useState<GlobalShapSummaryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchGlobalShapSummary({ segment, window })
      .then((response) => {
        if (!cancelled) {
          setData(response);
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Unable to load global SHAP summary");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [segment, window]);

  return (
    <div className="card shap-card">
      <h3>Global SHAP Summary</h3>
      <p className="muted">Segment-level feature influence over the last {window} window.</p>
      {error ? <p className="error-text">{error}</p> : null}
      {data ? (
        <div className="shap-chart-shell">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data.features} margin={{ top: 8, right: 16, left: 8, bottom: 28 }}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis
                dataKey="feature_name"
                angle={-20}
                textAnchor="end"
                interval={0}
                height={72}
                tick={{ fontSize: 11 }}
              />
              <YAxis />
              <Tooltip formatter={(value: number) => value.toFixed(4)} />
              <Bar dataKey="mean_abs_shap" radius={[8, 8, 0, 0]}>
                {data.features.map((feature) => (
                  <Cell
                    key={feature.feature_name}
                    fill={feature.mean_abs_shap >= 0 ? "var(--status-success)" : "var(--status-danger)"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <p className="muted">Loading global SHAP summary…</p>
      )}
    </div>
  );
}
