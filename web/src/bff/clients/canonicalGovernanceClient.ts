import type { SourceContext } from "@/src/bff/types/baseContracts";
import { latestArtifactPath, readJsonArtifact } from "@/src/bff/clients/artifactStore";

type ProposalCandidate = {
  run_id: string;
  model_version: string;
  gate_pass: boolean;
  weighted_segment_mdape_improvement: number;
  overall_ppe10_lift: number;
  max_major_segment_ppe10_drop: number;
  min_major_segment_ppe10: number;
  drift_alert_delta: number;
  fairness_alert_delta: number;
};

type ProposalPayload = {
  proposal_id: string;
  status: string;
  created_at_utc: string;
  expires_at_utc: string;
  champion: { run_id: string; model_version: string };
  winner: { run_id: string; model_version: string } | null;
  candidates_ranked: ProposalCandidate[];
};

const GATE_THRESHOLDS = {
  weighted_segment_mdape_improvement: 0.05,
  max_major_segment_ppe10_drop: 0.02,
  major_segment_ppe10_floor: 0.24,
  no_new_drift_alerts: 0,
  no_new_fairness_alerts: 0
} as const;

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
  const latestProposalPath = await latestArtifactPath("reports/arena", (name) =>
    /^proposal_.*\.json$/.test(name)
  );

  const proposal = latestProposalPath
    ? await readJsonArtifact<ProposalPayload>(latestProposalPath)
    : null;

  const candidatesRanked = (proposal?.candidates_ranked ?? []).map((candidate) => ({
    run_id: candidate.run_id,
    model_version: candidate.model_version,
    gate_pass: candidate.gate_pass,
    weighted_segment_mdape_improvement: candidate.weighted_segment_mdape_improvement,
    overall_ppe10_lift: candidate.overall_ppe10_lift,
    max_major_segment_ppe10_drop: candidate.max_major_segment_ppe10_drop,
    min_major_segment_ppe10: candidate.min_major_segment_ppe10,
    drift_alert_delta: candidate.drift_alert_delta,
    fairness_alert_delta: candidate.fairness_alert_delta
  }));

  const candidate = candidatesRanked[0] ?? null;

  const gateResults: Array<{
    gate_key: string;
    label: string;
    status: "pass" | "fail";
    threshold: string;
    actual: number;
  }> = candidate
    ? [
        {
          gate_key: "weighted_segment_mdape_improvement",
          label: "Weighted segment MdAPE improvement",
          status:
            candidate.weighted_segment_mdape_improvement >=
            GATE_THRESHOLDS.weighted_segment_mdape_improvement
              ? "pass"
              : "fail",
          threshold: `>= ${GATE_THRESHOLDS.weighted_segment_mdape_improvement}`,
          actual: candidate.weighted_segment_mdape_improvement
        },
        {
          gate_key: "max_major_segment_ppe10_drop",
          label: "Max major segment PPE10 drop",
          status:
            candidate.max_major_segment_ppe10_drop <= GATE_THRESHOLDS.max_major_segment_ppe10_drop
              ? "pass"
              : "fail",
          threshold: `<= ${GATE_THRESHOLDS.max_major_segment_ppe10_drop}`,
          actual: candidate.max_major_segment_ppe10_drop
        },
        {
          gate_key: "major_segment_ppe10_floor",
          label: "Major segment PPE10 floor",
          status:
            candidate.min_major_segment_ppe10 >= GATE_THRESHOLDS.major_segment_ppe10_floor
              ? "pass"
              : "fail",
          threshold: `>= ${GATE_THRESHOLDS.major_segment_ppe10_floor}`,
          actual: candidate.min_major_segment_ppe10
        },
        {
          gate_key: "no_new_drift_alerts",
          label: "No new drift alerts",
          status: candidate.drift_alert_delta <= GATE_THRESHOLDS.no_new_drift_alerts ? "pass" : "fail",
          threshold: `<= ${GATE_THRESHOLDS.no_new_drift_alerts}`,
          actual: candidate.drift_alert_delta
        },
        {
          gate_key: "no_new_fairness_alerts",
          label: "No new fairness alerts",
          status:
            candidate.fairness_alert_delta <= GATE_THRESHOLDS.no_new_fairness_alerts ? "pass" : "fail",
          threshold: `<= ${GATE_THRESHOLDS.no_new_fairness_alerts}`,
          actual: candidate.fairness_alert_delta
        }
      ]
    : [];

  const championAlias = proposal?.champion
    ? { model_version: proposal.champion.model_version, run_id: proposal.champion.run_id }
    : { model_version: null, run_id: null };

  const winner = proposal?.winner
    ? { run_id: proposal.winner.run_id, model_version: proposal.winner.model_version }
    : null;

  const topCandidateAlias = candidate
    ? { model_version: candidate.model_version, run_id: candidate.run_id }
    : { model_version: null, run_id: null };

  const statusReason = proposal
    ? proposal.status === "approved"
      ? "Latest proposal is approved."
      : `Latest proposal status is '${proposal.status}'. Governance actions remain read-only in no-auth mode.`
    : "No proposal artifact found; showing empty governance state.";

  return {
    payload: {
      registered_model_name: "spec-nyc-avm",
      aliases: {
        champion: championAlias,
        challenger: { model_version: null, run_id: null },
        candidate: topCandidateAlias
      },
      latest_proposal: {
        proposal_id: proposal?.proposal_id ?? "none",
        status: proposal?.status ?? "unavailable",
        created_at_utc: proposal?.created_at_utc ?? new Date(0).toISOString(),
        expires_at_utc: proposal?.expires_at_utc ?? new Date(0).toISOString(),
        champion: championAlias,
        winner,
        candidates_ranked: candidatesRanked
      },
      gate_results: gateResults,
      status_reason: statusReason,
      actions_enabled: false
    },
    sourceContext: {
      source_id: latestProposalPath ?? "reports/arena/proposal_*.json",
      source_type: "other"
    }
  };
}
