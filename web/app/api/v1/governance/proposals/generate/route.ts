import { okJson } from "@/src/lib/http";

export async function POST() {
  return okJson({
    proposal_id: crypto.randomUUID().replace(/-/g, "").slice(0, 12),
    status: "pending",
    created_at_utc: new Date().toISOString()
  }, 201);
}
