import { errorJson, okJson } from "@/src/lib/http";
import { proposalActionRequestSchema } from "@/src/features/governance/schemas/governanceSchemas";

export async function POST(req: Request, { params }: { params: { proposalId: string } }) {
  const payload = await req.json().catch(() => null);
  const parsed = proposalActionRequestSchema.safeParse(payload);
  if (!parsed.success) {
    return errorJson(`Invalid payload: ${parsed.error.issues[0]?.message ?? "unknown error"}`);
  }

  return okJson({
    proposal_id: params.proposalId,
    status: "approved",
    actor: parsed.data.actor,
    approved_at_utc: new Date().toISOString()
  });
}
