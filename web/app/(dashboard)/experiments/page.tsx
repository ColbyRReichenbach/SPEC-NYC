import { loadPlatformData } from "@/src/features/platform/data";
import { ExperimentLabView } from "@/src/features/platform/ExperimentLab";

export default async function ExperimentsPage() {
  const data = await loadPlatformData();
  return <ExperimentLabView data={data} />;
}
