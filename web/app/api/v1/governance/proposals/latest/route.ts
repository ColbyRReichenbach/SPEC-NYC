import { okJson } from "@/src/lib/http";

export async function GET() {
  return okJson({
    proposal_id: "57e6c66f5205",
    status: "no_winner",
    champion: { run_id: "34e917e198af4e58adb2097b8d9ca229", model_version: "1" },
    winner: null,
    candidates_ranked: [
      {
        run_id: "879ab7838c214d3a907e34a687978264",
        gate_pass: false,
        weighted_segment_mdape_improvement: -0.7001,
        overall_ppe10_lift: -0.1068,
        max_major_segment_ppe10_drop: 0.1268,
        min_major_segment_ppe10: 0.1219
      }
    ]
  });
}
