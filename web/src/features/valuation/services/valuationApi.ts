import {
  propertyExplanationRequestSchema,
  propertyExplanationResponseSchema,
  singleValuationRequestSchema,
  singleValuationResponseSchema,
  type SingleValuationRequest,
  type SingleValuationResponse
} from "@/src/features/valuation/schemas/valuationSchemas";

export async function requestSingleValuation(
  payload: SingleValuationRequest
): Promise<SingleValuationResponse> {
  const parsedPayload = singleValuationRequestSchema.parse(payload);

  const response = await fetch("/api/v1/valuations/single", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(parsedPayload)
  });

  if (!response.ok) {
    throw new Error(`Valuation request failed with status ${response.status}`);
  }

  const data = await response.json();
  return singleValuationResponseSchema.parse(data);
}

export async function requestPropertyExplanation(valuationId: string, topN = 5) {
  const payload = propertyExplanationRequestSchema.parse({ valuation_id: valuationId, top_n: topN });

  const response = await fetch("/api/v1/explanations/property", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(`Explanation request failed with status ${response.status}`);
  }

  const data = await response.json();
  return propertyExplanationResponseSchema.parse(data);
}
