import { loadPlatformData } from "@/src/features/platform/data";
import { PlatformShell } from "@/src/features/platform/PlatformShell";

export default async function DashboardLayout({ children }: { children: React.ReactNode }) {
  const data = await loadPlatformData();
  return <PlatformShell data={data}>{children}</PlatformShell>;
}
