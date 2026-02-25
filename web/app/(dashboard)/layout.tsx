import ContextStrip from "@/src/components/layout/ContextStrip";
import CopilotDrawer from "@/src/components/layout/CopilotDrawer";
import SidebarNav from "@/src/components/layout/SidebarNav";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="dashboard-shell">
      <SidebarNav />
      <div className="dashboard-main">
        <ContextStrip />
        <div className="dashboard-content">{children}</div>
      </div>
      <CopilotDrawer />
    </div>
  );
}
