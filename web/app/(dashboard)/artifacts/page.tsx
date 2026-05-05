import { loadPlatformData } from "@/src/features/platform/data";
import { ArtifactsView } from "@/src/features/platform/views";

export default async function ArtifactsPage() {
  const data = await loadPlatformData();
  return <ArtifactsView data={data} />;
}
