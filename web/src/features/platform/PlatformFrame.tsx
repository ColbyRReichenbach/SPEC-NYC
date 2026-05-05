"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  BarChart3,
  Boxes,
  ChevronLeft,
  ChevronRight,
  FileSearch,
  FlaskConical,
  Gauge,
  Home,
  ShieldCheck
} from "lucide-react";

import type { PlatformData } from "@/src/features/platform/data";

const SIDEBAR_COLLAPSED_KEY = "spec-sidebar-collapsed";

const navItems = [
  { href: "/valuation/single", label: "Workbench", icon: Home },
  { href: "/eda", label: "EDA Lab", icon: BarChart3 },
  { href: "/experiments", label: "Experiments", icon: FlaskConical },
  { href: "/governance", label: "Governance", icon: ShieldCheck },
  { href: "/monitoring", label: "Monitoring", icon: Gauge },
  { href: "/artifacts", label: "Artifacts", icon: Boxes }
];

type PlatformFrameProps = {
  appName: string;
  data: PlatformData;
  children: React.ReactNode;
};

export function PlatformFrame({ appName, data, children }: PlatformFrameProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  useEffect(() => {
    setIsCollapsed(window.localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "true");
  }, []);

  function toggleSidebar() {
    setIsCollapsed((current) => {
      const next = !current;
      window.localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(next));
      return next;
    });
  }

  return (
    <div className={["platform-shell", isCollapsed ? "is-collapsed" : ""].filter(Boolean).join(" ")}>
      <aside className="platform-sidebar" aria-label="Platform navigation">
        <div className="platform-sidebar-header">
          <div className="platform-lockup">
            <span className="platform-mark">SP</span>
            <div>
              <strong>{appName}</strong>
              <span>Governed AVM Lab</span>
            </div>
          </div>
          <button
            className="icon-button sidebar-toggle"
            type="button"
            onClick={toggleSidebar}
            aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-expanded={!isCollapsed}
          >
            {isCollapsed ? <ChevronRight size={17} aria-hidden="true" /> : <ChevronLeft size={17} aria-hidden="true" />}
          </button>
        </div>
        <nav>
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <Link key={item.href} href={item.href} className="nav-link" title={item.label}>
                <Icon size={17} aria-hidden="true" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
        <div className="sidebar-evidence">
          <FileSearch size={18} aria-hidden="true" />
          <span>Latest package</span>
          <strong>{data.package.modelVersion}</strong>
          <code>{data.package.id}</code>
        </div>
      </aside>

      <div className="platform-main">
        <header className="platform-topbar">
          <div>
            <span className="eyebrow">Production Readiness</span>
            <strong>{data.release.allGreen ? "Release gates clear" : "Release gates blocked"}</strong>
          </div>
          <div className="topbar-metrics" aria-label="Current model summary">
            <span>
              <BarChart3 size={15} aria-hidden="true" />
              PPE10 {formatPercent(data.package.ppe10)}
            </span>
            <span>MdAPE {formatPercent(data.package.mdape)}</span>
            <span>{data.package.trainRows.toLocaleString()} train rows</span>
          </div>
        </header>
        <main className="platform-content">{children}</main>
      </div>
    </div>
  );
}

function formatPercent(value: number) {
  return `${Math.round(value * 1000) / 10}%`;
}
