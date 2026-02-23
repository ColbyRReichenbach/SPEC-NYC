import Link from "next/link";

const navItems = [
  { href: "/valuation/single", label: "Valuation" },
  { href: "/valuation/batch", label: "Batch" },
  { href: "/governance", label: "Governance" },
  { href: "/monitoring", label: "Monitoring" },
  { href: "/copilot", label: "Copilot" },
  { href: "/artifacts", label: "Artifacts" }
];

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "220px 1fr", minHeight: "100vh" }}>
      <aside style={{ borderRight: "1px solid var(--border-subtle)", padding: "20px" }}>
        <h2 style={{ marginTop: 0 }}>SPEC NYC</h2>
        <nav>
          <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: "10px" }}>
            {navItems.map((item) => (
              <li key={item.href}>
                <Link href={item.href}>{item.label}</Link>
              </li>
            ))}
          </ul>
        </nav>
      </aside>
      <main>{children}</main>
    </div>
  );
}
