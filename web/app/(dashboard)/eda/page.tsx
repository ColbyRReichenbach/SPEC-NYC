import { loadPlatformData } from "@/src/features/platform/data";
import { EdaHypothesisLabView } from "@/src/features/platform/EdaHypothesisLab";

export default async function EdaPage() {
  const data = await loadPlatformData();
  return <EdaHypothesisLabView data={data} />;
}
