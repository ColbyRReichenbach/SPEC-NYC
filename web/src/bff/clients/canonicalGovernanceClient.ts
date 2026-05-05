import type { SourceContext } from "@/src/bff/types/baseContracts";
import { resolveRepoRoot } from "@/src/bff/clients/artifactStore";
import { listGovernanceState, type ReleaseProposal } from "@/src/features/platform/releaseRegistry";

type ProposalCandidate = {
  run_id: string;
  model_version: string;
  gate_pass: boolean;
  weighted_segment_mdape_improvement: number;
  overall_ppe10_lift: number;
  max_major_segment_ppe10_drop: number;
  min_major_segment_ppe10: number;
  drift_alert_delta: number | null;
  fairness_alert_delta: number | null;
};

export async function buildCanonicalGovernanceStatus(): Promise<{
  payload: {
    registered_model_name: string;
    aliases: {
      champion: { model_version: string | null; run_id: string | null };
      challenger: { model_version: string | null; run_id: string | null };
      candidate: { model_version: string | null; run_id: string | null };
    };
    latest_proposal: {
      proposal_id: string;
      status: string;
      created_at_utc: string;
      expires_at_utc: string;
      champion: { model_version: string | null; run_id: string | null };
      winner: { run_id: string; model_version: string } | null;
      candidates_ranked: ProposalCandidate[];
    };
    gate_results: Array<{
      gate_key: string;
      label: string;
      status: "pass" | "fail";
      threshold: string;
      actual: number;
    }>;
    status_reason: string;
    actions_enabled: boolean;
  };
  sourceContext: SourceContext;
}> {
  const repoRoot = await resolveRepoRoot();
  const state = await listGovernanceState(repoRoot);
  const latestProposal = state.proposals[0] ?? null;
  const champion = state.champion.champion_package_id
    ? {
        model_version: state.champion.champion_package_id,
        run_id: state.champion.active_proposal_id
      }
    : { model_version: null, run_id: null };
  const topCandidate = latestProposal ? proposalCandidate(latestProposal) : null;
  const winner = latestProposal?.status === "approved"
    ? {
        run_id: latestProposal.proposal_id,
        model_version: latestProposal.candidate.package_id
      }
    : null;

  return {
    payload: {
      registered_model_name: state.champion.registered_model_name,
      aliases: {
        champion,
        challenger: topCandidate ? { model_version: topCandidate.model_version, run_id: topCandidate.run_id } : { model_version: null, run_id: null },
        candidate: topCandidate ? { model_version: topCandidate.model_version, run_id: topCandidate.run_id } : { model_version: null, run_id: null }
      },
      latest_proposal: latestProposal
        ? {
            proposal_id: latestProposal.proposal_id,
            status: latestProposal.status,
            created_at_utc: latestProposal.created_at_utc,
            expires_at_utc: latestProposal.expires_at_utc,
            champion: {
              model_version: latestProposal.champion.package_id,
              run_id: latestProposal.proposal_id
            },
            winner,
            candidates_ranked: [proposalCandidate(latestProposal)]
          }
        : {
            proposal_id: "none",
            status: "unavailable",
            created_at_utc: new Date(0).toISOString(),
            expires_at_utc: new Date(0).toISOString(),
            champion,
            winner: null,
            candidates_ranked: []
          },
      gate_results: latestProposal ? proposalGates(latestProposal) : [],
      status_reason: latestProposal
        ? `Latest governance proposal ${latestProposal.proposal_id} is ${latestProposal.status}.`
        : "No release proposal exists in the governed proposal registry.",
      actions_enabled: false
    },
    sourceContext: {
      source_id: latestProposal?.artifact_paths.release_proposal ?? "reports/governance/proposals",
      source_type: "other"
    }
  };
}

function proposalCandidate(proposal: ReleaseProposal): ProposalCandidate {
  return {
    run_id: proposal.experiment_id,
    model_version: proposal.candidate.package_id,
    gate_pass: proposal.gate_results.every((gate) => gate.status !== "fail"),
    weighted_segment_mdape_improvement: round(proposal.champion.metrics.mdape - proposal.candidate.metrics.mdape),
    overall_ppe10_lift: round(proposal.candidate.metrics.ppe10 - proposal.champion.metrics.ppe10),
    max_major_segment_ppe10_drop: round(Math.max(0, proposal.champion.metrics.ppe10 - proposal.candidate.metrics.ppe10)),
    min_major_segment_ppe10: round(proposal.candidate.metrics.ppe10),
    drift_alert_delta: null,
    fairness_alert_delta: null
  };
}

function proposalGates(proposal: ReleaseProposal) {
  return proposal.gate_results.map((gate) => ({
    gate_key: gate.name,
    label: gate.name.replaceAll("_", " "),
    status: gate.status === "fail" ? "fail" as const : "pass" as const,
    threshold: gate.detail,
    actual: gate.status === "fail" ? 0 : 1
  }));
}

function round(value: number) {
  return Math.round(value * 10000) / 10000;
}
