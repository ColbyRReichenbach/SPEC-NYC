import {
  globalShapSummaryResponseSchema,
  type GlobalShapSummaryResponse
} from "@/src/features/valuation/schemas/explainabilitySchemas";

type GlobalShapInput = {
  segment: string;
  window?: string;
};

export async function fetchGlobalShapSummary({
  segment,
  window = "180d"
}: GlobalShapInput): Promise<GlobalShapSummaryResponse> {
  const params = new URLSearchParams();
  params.set("segment", segment);
  params.set("window", window);

  const response = await fetch(`/api/v1/explanations/shap/global?${params.toString()}`, {
    cache: "no-store"
  });
  if (!response.ok) {
    throw new Error(`Global SHAP request failed with status ${response.status}`);
  }

  return globalShapSummaryResponseSchema.parse(await response.json());
}
