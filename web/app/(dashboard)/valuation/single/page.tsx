import { loadPlatformData } from "@/src/features/platform/data";
import { WorkbenchView } from "@/src/features/platform/views";

export default async function SingleValuationPage() {
  const data = await loadPlatformData();
  return <WorkbenchView data={data} />;
}
