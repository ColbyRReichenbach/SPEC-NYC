"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { useBrandProfile } from "@/src/components/providers/BrandProvider";

const SIDEBAR_STORAGE_KEY = "spec_dashboard_sidebar_collapsed";

const navItems = [
  { href: "/valuation/single", label: "Map Valuation", shortLabel: "MV" },
  { href: "/governance", label: "Governance", shortLabel: "GV" },
  { href: "/monitoring", label: "Monitoring", shortLabel: "MN" },
  { href: "/artifacts", label: "Artifacts", shortLabel: "AR" }
];

function isActive(pathname: string, href: string) {
  return pathname === href || pathname.startsWith(`${href}/`);
}

export default function SidebarNav() {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const { profile } = useBrandProfile();

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(SIDEBAR_STORAGE_KEY);
      if (stored === "true") {
        setCollapsed(true);
      }
    } catch {
      // Local storage is optional and should never block rendering.
    }
  }, []);

  const toggleSidebar = () => {
    setCollapsed((current) => {
      const next = !current;
      try {
        window.localStorage.setItem(SIDEBAR_STORAGE_KEY, String(next));
      } catch {
        // Ignore persistence failures and keep runtime behavior.
      }
      return next;
    });
  };

  return (
    <aside className={`dashboard-nav${collapsed ? " collapsed" : ""}`} aria-label="Dashboard navigation">
      <div className="dashboard-nav-header">
        <div className="brand-lockup">
          {profile.logoPath ? (
            <Image
              src={profile.logoPath}
              alt={profile.logoAlt ?? profile.appName}
              width={180}
              height={56}
              priority
              className="brand-logo"
            />
          ) : (
            <span className="brand-text">{profile.appName}</span>
          )}
          <span className="brand-sub">{profile.navSubtitle}</span>
        </div>
        <button
          type="button"
          className="sidebar-toggle"
          onClick={toggleSidebar}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          aria-expanded={!collapsed}
        >
          {collapsed ? ">" : "<"}
        </button>
      </div>
      <nav>
        <ul>
          {navItems.map((item) => {
            const active = isActive(pathname, item.href);
            return (
              <li key={item.href}>
                <Link href={item.href} className={active ? "active" : undefined}>
                  <span className="nav-item-pill" aria-hidden="true">
                    {item.shortLabel}
                  </span>
                  <span className="nav-item-label">{item.label}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
    </aside>
  );
}
