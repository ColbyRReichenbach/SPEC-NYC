"use client";

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

import type { SingleValuationResponse } from "@/src/features/valuation/schemas/valuationSchemas";

type ShapWaterfallProps = {
  valuation: SingleValuationResponse | null;
};

type WaterfallRow = {
  feature: string;
  impact: number;
  label: string;
};

function toRows(valuation: SingleValuationResponse): WaterfallRow[] {
  const positive = valuation.explanation.drivers_positive.map((driver) => ({
    feature: driver.feature,
    impact: driver.impact,
    label: driver.display
  }));
  const negative = valuation.explanation.drivers_negative.map((driver) => ({
    feature: driver.feature,
    impact: driver.impact,
    label: driver.display
  }));

  return [...positive, ...negative]
    .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))
    .slice(0, 8);
}

function formatDollar(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0
  }).format(value);
}

export default function ShapWaterfall({ valuation }: ShapWaterfallProps) {
  if (!valuation) {
    return (
      <div className="card">
        <h3>Local SHAP Waterfall</h3>
        <p className="muted">Run a valuation to visualize top positive and negative contribution drivers.</p>
      </div>
    );
  }

  const rows = toRows(valuation);

  return (
    <div className="card shap-card">
      <h3>Local SHAP Waterfall</h3>
      <p className="muted">Contribution deltas around model baseline for this property.</p>
      <div className="shap-chart-shell">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={rows} layout="vertical" margin={{ top: 8, right: 24, left: 24, bottom: 8 }}>
            <CartesianGrid strokeDasharray="4 4" />
            <XAxis type="number" tickFormatter={(value) => `$${Math.round(Math.abs(value) / 1000)}k`} />
            <YAxis type="category" dataKey="feature" width={120} />
            <Tooltip
              formatter={(value: number) => formatDollar(value)}
              labelFormatter={(value) => `Feature: ${value}`}
            />
            <Bar dataKey="impact" radius={[6, 6, 6, 6]}>
              {rows.map((row) => (
                <Cell key={row.feature} fill={row.impact >= 0 ? "var(--status-success)" : "var(--status-danger)"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
