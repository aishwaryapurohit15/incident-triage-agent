"""Fast unit tests for the incident triage project."""

import numpy as np
import pandas as pd
import pytest

from triage import risk
from triage.agent import DEFAULT_POLICY, PermissionDenied, TriageAgent
from triage.config import SEVERITIES, TEAMS
from triage.generate import generate
from triage.models import TextClassifier, cost_per_ticket
from triage.quality import audit, clean, normalize_text


@pytest.fixture(scope="module")
def small():
    raw, gt = generate(n_days=90, seed=3)
    df = clean(raw).merge(gt, on="incident_id", how="left")
    return raw, gt, df


@pytest.fixture(scope="module")
def agent(small):
    _, _, df = small
    train = df.dropna(subset=["assigned_team"])
    sev = TextClassifier().fit(train, train["severity"])
    team = TextClassifier().fit(train, train["assigned_team"])
    return TriageAgent(sev, team, train.assign(cluster=0), dict(DEFAULT_POLICY), "test-v1")


def test_generator_shapes(small):
    raw, gt, _ = small
    assert len(raw) > 1000
    assert set(raw["severity"]) <= set(SEVERITIES)
    assert set(raw["assigned_team"].dropna()) <= set(TEAMS)
    assert gt["incident_id"].is_unique


def test_audit_finds_planted_defects(small):
    raw, _, _ = small
    a = audit(raw)
    assert a["exact_duplicate_rows"] > 0
    assert a["missing_resolution_notes"] > 0
    assert a["raw_category_labels"] > a["clean_category_labels"]


def test_cleaning_removes_impossible_timestamps(small):
    _, _, df = small
    assert (df["hours_to_resolve"].dropna() >= 0).all()
    assert not df.drop(columns=["root_cause", "true_severity", "true_team", "duplicate_of"]).duplicated().any()


def test_normalize_text():
    assert normalize_text("Pls check DB on prod, CHG12345 failed 3 times!") == "please check database on production changeid failed num times"


def test_cost_matrix_penalizes_under_triage_more():
    over = cost_per_ticket(["SEV-3"], ["SEV-1"])
    under = cost_per_ticket(["SEV-1"], ["SEV-3"])
    assert under > over


def test_agent_refuses_write_actions(agent):
    for tool in ["close_ticket", "set_severity", "reassign_ticket"]:
        with pytest.raises(PermissionDenied):
            agent.call(tool, "INC0000001")
    assert any(not e.allowed for e in agent.audit)


def test_agent_suggestion_contract(agent):
    ticket = {"incident_id": "INC9999999", "opened_at": "2025-01-15 10:30:00", "service": "Payments Gateway",
              "channel": "Phone", "short_description": "Payments Gateway down",
              "description": "Complete outage, no transactions are going through. All customers are affected."}
    s = agent.triage(ticket)
    assert s["severity"] in SEVERITIES
    assert 0 <= s["severity_confidence"] <= 1
    assert s["review_route"] in {"fast_path", "full_review", "escalate_now"}
    assert abs(sum(s["severity_probabilities"].values()) - 1) < 1e-3


def test_high_severity_never_fast_path(agent):
    p = {"SEV-1": 0.05, "SEV-2": 0.6, "SEV-3": 0.3, "SEV-4": 0.05}
    assert agent._route("SEV-2", 0.6, 0.99, p) != "fast_path"
    p = {"SEV-1": 0.4, "SEV-2": 0.1, "SEV-3": 0.4, "SEV-4": 0.1}
    assert agent._route("SEV-3", 0.4, 0.99, p) == "escalate_now"


def test_psi_is_zero_for_identical_distributions():
    s = pd.Series({"a": 10, "b": 20})
    assert risk.psi(s, s) == pytest.approx(0, abs=1e-9)
    assert risk.psi(s, pd.Series({"a": 20, "b": 10})) > 0.1


def test_loss_distribution_ordering(small):
    _, _, df = small
    ld = risk.loss_distribution(df, sims=2000)
    assert ld["expected_annual_loss"] <= ld["var_99"] <= ld["var_99_9"] <= ld["es_99_9"]
    assert np.isfinite(ld["expected_annual_loss"])
