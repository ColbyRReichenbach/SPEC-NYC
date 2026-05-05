import { loadPlatformData } from "@/src/features/platform/data";
import { MonitoringView } from "@/src/features/platform/views";

export default async function MonitoringPage() {
  const data = await loadPlatformData();
  return <MonitoringView data={data} />;
}
