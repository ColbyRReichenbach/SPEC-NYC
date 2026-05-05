"use client";

import { useCallback, useEffect, useState } from "react";
import { CheckCircle2, GitCompareArrows, PackageCheck, ShieldCheck, XCircle } from "lucide-react";

import type { ExperimentRunBundle } from "@/src/features/platform/experimentRegistry";
import type { PlatformData } from "@/src/features/platform/data";
import { DEFAULT_PLATFORM_OPTIONS } from "@/src/features/platform/platformOptions";
import type { ChampionAlias, ReleaseProposal } from "@/src/features/platform/releaseRegistry";

type GovernanceState = {
  proposals: ReleaseProposal[];
  eligibleExperiments: ExperimentRunBundle[];
  champion: ChampionAlias;
};

export function ReleaseGovernancePanel({ data }: { data: PlatformData }) {
  const options = data.options ?? DEFAULT_PLATFORM_OPTIONS;
  const [state, setState] = useState<GovernanceState | null>(null);
  const [selectedExperimentId, setSelectedExperimentId] = useState("");
  const [selectedProposalId, setSelectedProposalId] = useState("");
  const [decisionReason, setDecisionReason] = useState("Approved after reviewing same-dataset challenger evidence.");
  const [status, setStatus] = useState<"loading" | "idle" | "running" | "error">("loading");
  const [message, setMessage] = useState<string | null>(null);

  const applyState = useCallback((nextState: GovernanceState) => {
    setState(nextState);
    const firstEligible = nextState.eligibleExperiments[0]?.id ?? "";
    const firstPending = nextState.proposals.find((proposal) => proposal.status === "pending")?.proposal_id ?? "";
    setSelectedExperimentId((current) => current || firstEligible);
    setSelectedProposalId((current) => {
      if (current && nextState.proposals.some((proposal) => proposal.proposal_id === current)) {
        return current;
      }
      return firstPending || nextState.proposals[0]?.proposal_id || "";
    });
  }, []);

  const loadState = useCallback(async (signal?: AbortSignal) => {
    setStatus("loading");
    try {
      const response = await fetch("/api/v1/governance/proposals", { signal });
      const payload = (await response.json()) as GovernanceState & { error?: string };
      if (!response.ok) {
        throw new Error(payload.error ?? "Unable to load release proposals.");
      }
      applyState(payload);
      setMessage(null);
      setStatus("idle");
    } catch (error) {
      if (!signal?.aborted) {
        setMessage(error instanceof Error ? error.message : "Unable to load release proposals.");
        setStatus("error");
      }
    }
  }, [applyState]);

  useEffect(() => {
    const controller = new AbortController();
    void loadState(controller.signal);
    return () => controller.abort();
  }, [loadState]);

  async function mutate(endpoint: string, body: Record<string, unknown>, successMessage: string) {
    setStatus("running");
    setMessage(null);
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      const payload = (await response.json()) as GovernanceState & { error?: string; proposal?: ReleaseProposal };
      if (!response.ok) {
        throw new Error(payload.error ?? "Governance action failed.");
      }
      applyState(payload);
      if (payload.proposal?.proposal_id) {
        setSelectedProposalId(payload.proposal.proposal_id);
      }
      setMessage(successMessage);
      setStatus("idle");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Governance action failed.");
      setStatus("error");
    }
  }

  const pendingProposal = state?.proposals.find((proposal) => proposal.proposal_id === selectedProposalId);
  const isRunning = status === "running" || status === "loading";

  return (
    <section className="panel release-governance-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Champion Registry</span>
          <h2>Release proposal workflow</h2>
        </div>
        <span className={`status-pill ${state?.champion.champion_package_id ? "pass" : "warn"}`}>
          <ShieldCheck size={14} aria-hidden="true" />
          {state?.champion.champion_package_id ? "champion set" : "no champion"}
        </span>
      </div>

      <div className="release-registry-grid">
        <div>
          <span>Current champion</span>
          <code>{state?.champion.champion_package_id ?? "not_set"}</code>
        </div>
        <div>
          <span>Rollback target</span>
          <code>{state?.champion.rollback_package_id ?? "not_set"}</code>
        </div>
        <div>
          <span>Active proposal</span>
          <code>{state?.champion.active_proposal_id ?? "not_set"}</code>
        </div>
      </div>

      <div className="release-action-grid">
        <div className="release-action-card">
          <div>
            <PackageCheck size={18} aria-hidden="true" />
            <strong>Create proposal from completed experiment</strong>
          </div>
          <select
            aria-label="Eligible completed experiment"
            value={selectedExperimentId}
            onChange={(event) => setSelectedExperimentId(event.target.value)}
            disabled={isRunning || !state?.eligibleExperiments.length}
          >
            {state?.eligibleExperiments.length ? (
              state.eligibleExperiments.map((experiment) => (
                <option key={experiment.id} value={experiment.id}>
                  {experiment.id} · {experiment.run_plan.challenger_package_id}
                </option>
              ))
            ) : (
              <option>No completed passed experiments</option>
            )}
          </select>
          <button
            className="command-button secondary"
            type="button"
            disabled={isRunning || !selectedExperimentId}
            onClick={() =>
              mutate(
                "/api/v1/governance/proposals",
                { experimentId: selectedExperimentId },
                "Release proposal created."
              )
            }
          >
            <GitCompareArrows size={15} aria-hidden="true" />
            Create Release Proposal
          </button>
        </div>

        <div className="release-action-card">
          <div>
            <CheckCircle2 size={18} aria-hidden="true" />
            <strong>Approve or reject pending proposal</strong>
          </div>
          <select
            aria-label="Release proposal"
            value={selectedProposalId}
            onChange={(event) => setSelectedProposalId(event.target.value)}
            disabled={isRunning || !state?.proposals.length}
          >
            {state?.proposals.length ? (
              state.proposals.map((proposal) => (
                <option key={proposal.proposal_id} value={proposal.proposal_id}>
                  {proposal.proposal_id} · {proposal.status}
                </option>
              ))
            ) : (
              <option>No release proposals</option>
            )}
          </select>
          <input
            aria-label="Decision reason"
            value={decisionReason}
            onChange={(event) => setDecisionReason(event.target.value)}
            disabled={isRunning}
          />
          <div className="release-button-row">
            <button
              className="command-button secondary"
              type="button"
              disabled={isRunning || !pendingProposal || pendingProposal.status !== "pending"}
              onClick={() =>
                mutate(
                  `/api/v1/governance/proposals/${selectedProposalId}/approve`,
                  { reason: decisionReason, decidedBy: options.identity.release_owner },
                  "Release proposal approved and champion registry updated."
                )
              }
            >
              <CheckCircle2 size={15} aria-hidden="true" />
              Approve
            </button>
            <button
              className="command-button danger"
              type="button"
              disabled={isRunning || !pendingProposal || !["pending", "blocked"].includes(pendingProposal.status)}
              onClick={() =>
                mutate(
                  `/api/v1/governance/proposals/${selectedProposalId}/reject`,
                  { reason: decisionReason || "Rejected by release owner.", decidedBy: options.identity.release_owner },
                  "Release proposal rejected."
                )
              }
            >
              <XCircle size={15} aria-hidden="true" />
              Reject
            </button>
          </div>
        </div>
      </div>

      {message ? <p className={status === "error" ? "form-error" : "form-success"}>{message}</p> : null}

      <div className="proposal-list" aria-label="Release proposals">
        {state?.proposals.length ? (
          state.proposals.slice(0, 6).map((proposal) => <ProposalCard key={proposal.proposal_id} proposal={proposal} />)
        ) : (
          <p className="quiet">No release proposals have been generated yet.</p>
        )}
      </div>
    </section>
  );
}

function ProposalCard({ proposal }: { proposal: ReleaseProposal }) {
  return (
    <article className="proposal-card">
      <div>
        <strong>{proposal.proposal_id}</strong>
        <span className={`status-pill ${proposal.status === "approved" ? "pass" : proposal.status === "rejected" || proposal.status === "blocked" ? "fail" : "warn"}`}>
          {proposal.status}
        </span>
      </div>
      <dl>
        <div><dt>Candidate</dt><dd><code>{proposal.candidate.package_id}</code></dd></div>
        <div><dt>Champion</dt><dd><code>{proposal.champion.package_id}</code></dd></div>
        <div><dt>MdAPE delta</dt><dd>{formatDelta(proposal.comparison.metric_deltas.mdape)}</dd></div>
        <div><dt>PPE10 delta</dt><dd>{formatDelta(proposal.comparison.metric_deltas.ppe10)}</dd></div>
      </dl>
      <div className="proposal-gates">
        {proposal.gate_results.map((gate) => (
          <span key={gate.name} className={gate.status}>
            {gate.name}
          </span>
        ))}
      </div>
    </article>
  );
}

function formatDelta(value: number) {
  const rounded = Math.round(value * 10000) / 100;
  return `${rounded > 0 ? "+" : ""}${rounded}%`;
}
