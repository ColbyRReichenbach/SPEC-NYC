import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from src.mlops.arena import (
    approve,
    evaluate_candidate,
    is_eligible_run,
    proposal_is_expired,
    reject,
    validate_change_note,
)


class _FakeModelVersion:
    def __init__(self, version: str, run_id: str):
        self.version = version
        self.run_id = run_id


class _FakeClient:
    def __init__(self):
        self.alias_sets = []

    def get_model_version_by_alias(self, model_name, alias):
        return _FakeModelVersion("3", "old_run")

    def set_registered_model_alias(self, model_name, alias, version):
        self.alias_sets.append((model_name, alias, version))


class TestArena(unittest.TestCase):
    def setUp(self):
        self.policy = {
            "major_segment_min_n": 2,
            "gates": {
                "weighted_segment_mdape_improvement": 0.05,
                "max_major_segment_ppe10_drop": 0.02,
                "major_segment_ppe10_floor": 0.24,
            },
            "fairness": {"max_segment_ppe10_gap": 0.20},
            "scoring": {"mdape_weight": 0.5, "ppe10_weight": 0.3, "stability_weight": 0.2},
        }
        self.champion = {
            "overall": {"ppe10": 0.30, "mdape": 0.20},
            "per_segment": {
                "A": {"n": 10, "ppe10": 0.32, "mdape": 0.19},
                "B": {"n": 12, "ppe10": 0.28, "mdape": 0.21},
            },
            "metadata": {"drift_alerts": 0},
        }

    def test_evaluate_candidate_passes_composite_gate(self):
        challenger = {
            "overall": {"ppe10": 0.34, "mdape": 0.17},
            "per_segment": {
                "A": {"n": 10, "ppe10": 0.35, "mdape": 0.17},
                "B": {"n": 12, "ppe10": 0.30, "mdape": 0.18},
            },
            "metadata": {"drift_alerts": 0},
        }
        decision = evaluate_candidate(
            champion_metrics=self.champion,
            challenger_metrics=challenger,
            challenger_run_id="run_x",
            challenger_model_version="v2",
            policy=self.policy,
        )
        self.assertTrue(decision.gate_pass)
        self.assertGreater(decision.score, 0)

    def test_evaluate_candidate_fails_segment_drop_gate(self):
        challenger = {
            "overall": {"ppe10": 0.31, "mdape": 0.17},
            "per_segment": {
                "A": {"n": 10, "ppe10": 0.10, "mdape": 0.17},  # Large drop from 0.32.
                "B": {"n": 12, "ppe10": 0.31, "mdape": 0.18},
            },
            "metadata": {"drift_alerts": 0},
        }
        decision = evaluate_candidate(
            champion_metrics=self.champion,
            challenger_metrics=challenger,
            challenger_run_id="run_bad",
            challenger_model_version="v2",
            policy=self.policy,
        )
        self.assertFalse(decision.gate_pass)
        self.assertGreater(decision.max_major_segment_ppe10_drop, 0.02)

    def test_is_eligible_run_enforces_required_tags_and_no_smoke(self):
        good = {
            "status": "FINISHED",
            "run_name": "train-v2",
            "tags": {
                "run_kind": "train",
                "hypothesis_id": "H-1",
                "change_type": "feature",
                "change_summary": "added features",
                "owner": "ml-eng",
                "feature_set_version": "v2.1",
                "dataset_version": "20260211",
            },
        }
        self.assertTrue(is_eligible_run(good))

        smoke = dict(good)
        smoke["run_name"] = "v1_smoke"
        self.assertFalse(is_eligible_run(smoke))

    def test_proposal_expiry(self):
        now = datetime(2026, 2, 11, tzinfo=timezone.utc)
        proposal = {"expires_at_utc": (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")}
        self.assertTrue(proposal_is_expired(proposal, now=now))

    def test_validate_change_note_schema(self):
        valid_note = {
            "problem_statement": "underperforming",
            "change_rationale": "segment split",
            "change_details": "new features",
            "before_after_metrics": {"before": {}, "after": {}},
            "risk_callouts": "tail risk",
            "rollback_pointer": "champion:v1",
        }
        is_valid, missing = validate_change_note(valid_note)
        self.assertTrue(is_valid)
        self.assertEqual(missing, [])

    def test_approve_mutates_champion_alias_and_updates_proposal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            arena_dir = Path(tmpdir)
            proposal_path = arena_dir / "proposal_abc123.json"
            proposal = {
                "proposal_id": "abc123",
                "status": "pending",
                "expires_at_utc": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
                "registered_model_name": "spec-nyc-avm",
                "winner": {"run_id": "new_run", "model_version": "9", "score": 0.9},
                "champion": {"run_id": "old_run", "model_version": "3"},
            }
            proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
            fake_client = _FakeClient()

            with patch("src.mlops.arena._mlflow_client", return_value=(object(), fake_client)):
                result = approve(
                    proposal_path=proposal_path,
                    arena_dir=arena_dir,
                    approved_by="tester",
                )

            self.assertEqual(result["status"], "approved")
            self.assertIn(("spec-nyc-avm", "champion", "9"), fake_client.alias_sets)
            updated = json.loads(proposal_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["status"], "approved")
            self.assertEqual(updated["approved_by"], "tester")

    def test_reject_updates_proposal_without_alias_mutation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            arena_dir = Path(tmpdir)
            proposal_path = arena_dir / "proposal_abc124.json"
            proposal = {
                "proposal_id": "abc124",
                "status": "pending",
                "expires_at_utc": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
                "winner": {"run_id": "new_run", "model_version": "10"},
                "champion": {"run_id": "old_run", "model_version": "3"},
            }
            proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
            result = reject(
                reason="insufficient uplift",
                proposal_path=proposal_path,
                arena_dir=arena_dir,
                rejected_by="reviewer",
            )
            self.assertEqual(result["status"], "rejected")
            updated = json.loads(proposal_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["status"], "rejected")
            self.assertEqual(updated["rejection_reason"], "insufficient uplift")


if __name__ == "__main__":
    unittest.main()
