import { loadPlatformData } from "@/src/features/platform/data";
import { GovernanceView } from "@/src/features/platform/views";

export default async function GovernancePage() {
  const data = await loadPlatformData();
  return <GovernanceView data={data} />;
}
