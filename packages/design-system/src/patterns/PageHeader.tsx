import type { ReactNode } from "react";

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: string; actions?: ReactNode }) {
  return (
    <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
      <div>
        <h1 style={{ margin: 0 }}>{title}</h1>
        {subtitle ? <p style={{ margin: "6px 0 0", color: "#43524B" }}>{subtitle}</p> : null}
      </div>
      {actions}
    </header>
  );
}
