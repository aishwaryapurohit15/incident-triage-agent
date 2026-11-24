"""Incident triage agent: tools, schemas, permission boundaries and audit trail.

The agent may read tickets and history and may write *suggestions*. It has no
tool that closes a ticket, changes severity or reassigns work. Those actions
belong to a human triage coordinator, and the permission layer refuses them
even if a model or a prompt asks for them.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from .config import SEVERITIES
from .quality import normalize_text


class Permission(str, Enum):
    READ = "read"  # read tickets, history and reference data
    SUGGEST = "suggest"  # write a suggestion record for a human to review
    WRITE = "write"  # change the system of record (never granted to the agent)


AGENT_GRANTS = {Permission.READ, Permission.SUGGEST}


class PermissionDenied(Exception):
    pass


# ---------------------------------------------------------------------------
# Schemas (JSON Schema, shared with the LLM adapter and the website)
# ---------------------------------------------------------------------------
TICKET_SCHEMA = {
    "type": "object",
    "required": ["incident_id", "opened_at", "service", "short_description", "description"],
    "properties": {
        "incident_id": {"type": "string", "pattern": "^INC[0-9]{7}$"},
        "opened_at": {"type": "string", "format": "date-time"},
        "service": {"type": "string"},
        "channel": {"type": "string", "enum": ["Monitoring Alert", "Service Portal", "Email", "Phone"]},
        "short_description": {"type": "string", "maxLength": 200},
        "description": {"type": "string", "maxLength": 4000},
    },
}

SUGGESTION_SCHEMA = {
    "type": "object",
    "required": ["incident_id", "severity", "severity_confidence", "team", "team_confidence", "similar_incidents",
                 "possible_duplicate_of", "review_route", "rationale", "model_version"],
    "properties": {
        "incident_id": {"type": "string"},
        "severity": {"type": "string", "enum": SEVERITIES},
        "severity_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "team": {"type": "string"},
        "team_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "similar_incidents": {"type": "array", "maxItems": 5, "items": {"type": "object", "properties": {
            "incident_id": {"type": "string"}, "similarity": {"type": "number"}, "short_description": {"type": "string"}}}},
        "possible_duplicate_of": {"type": ["string", "null"]},
        "recurring_cluster": {"type": ["integer", "null"]},
        "review_route": {"type": "string", "enum": ["fast_path", "full_review", "escalate_now"]},
        "rationale": {"type": "string", "maxLength": 500},
        "model_version": {"type": "string"},
    },
}


@dataclass
class Tool:
    name: str
    description: str
    permission: Permission
    input_schema: dict
    fn: Callable[..., Any]


@dataclass
class AuditEvent:
    event_id: str
    timestamp: float
    tool: str
    permission: str
    allowed: bool
    incident_id: str | None
    detail: dict = field(default_factory=dict)


class TriageAgent:
    """Deterministic agent loop over a fixed tool set.

    The same tool contract works with an LLM planner (see `llm.py`); this
    implementation runs the tools in a fixed order so that results are
    reproducible and auditable.
    """

    def __init__(self, severity_model, team_model, history: pd.DataFrame, policy: dict, model_version: str,
                 audit_path: Path | None = None):
        self.sev_model = severity_model
        self.team_model = team_model
        self.history = history.reset_index(drop=True)
        self.policy = policy
        self.model_version = model_version
        self.audit_path = audit_path
        self.audit: list[AuditEvent] = []
        from .similarity import similarity_vectorizer

        self.sim_vec = similarity_vectorizer(self.history["text"])
        self._hist_X = normalize(self.sim_vec.transform(self.history["text"]))
        self.tools = {t.name: t for t in self._build_tools()}

    # -- tool registry ------------------------------------------------------
    def _build_tools(self) -> list[Tool]:
        id_only = {"type": "object", "required": ["incident_id"], "properties": {"incident_id": {"type": "string"}}}
        return [
            Tool("classify_severity", "Suggest a severity (SEV-1 to SEV-4) with class probabilities.", Permission.READ, TICKET_SCHEMA, self._classify_severity),
            Tool("suggest_owner", "Suggest the resolver team with class probabilities.", Permission.READ, TICKET_SCHEMA, self._suggest_owner),
            Tool("find_similar_incidents", "Return the most similar past incidents and a possible duplicate.", Permission.READ, TICKET_SCHEMA, self._find_similar),
            Tool("submit_suggestion", "Write a suggestion record for human review.", Permission.SUGGEST, SUGGESTION_SCHEMA, self._submit),
            # Registered so that refusals are explicit and audited, never silently absent.
            Tool("set_severity", "Change the severity in the system of record.", Permission.WRITE, id_only, None),
            Tool("close_ticket", "Close the ticket in the system of record.", Permission.WRITE, id_only, None),
            Tool("reassign_ticket", "Reassign the ticket to another team.", Permission.WRITE, id_only, None),
        ]

    def call(self, name: str, incident_id: str | None = None, **kwargs):
        tool = self.tools[name]
        allowed = tool.permission in AGENT_GRANTS
        self._log(name, tool.permission, allowed, incident_id, {k: v for k, v in kwargs.items() if k != "ticket"})
        if not allowed:
            raise PermissionDenied(f"The agent is not permitted to call '{name}' ({tool.permission.value}). A human must do this.")
        return tool.fn(**kwargs)

    # -- tools --------------------------------------------------------------
    @staticmethod
    def _frame(ticket: dict) -> pd.DataFrame:
        t = dict(ticket)
        t["text"] = normalize_text(f"{t.get('short_description', '')}. {t.get('description', '')}")
        ts = pd.Timestamp(t.get("opened_at", pd.Timestamp.now()))
        t["business_hours"] = int(8 <= ts.hour <= 18 and ts.dayofweek < 5)
        t.setdefault("channel", "Service Portal")
        return pd.DataFrame([t])

    def _classify_severity(self, ticket: dict) -> dict:
        p = self.sev_model.predict_proba(self._frame(ticket))[0]
        return dict(zip(self.sev_model.classes_, np.round(p, 4).tolist()))

    def _suggest_owner(self, ticket: dict) -> dict:
        p = self.team_model.predict_proba(self._frame(ticket))[0]
        return dict(zip(self.team_model.classes_, np.round(p, 4).tolist()))

    def _find_similar(self, ticket: dict, k: int = 3) -> dict:
        f = self._frame(ticket)
        q = normalize(self.sim_vec.transform(f["text"]))
        sims = (self._hist_X @ q.T).toarray().ravel()
        # Small bonus for the same service, so like-for-like incidents rank first.
        sims = sims + 0.05 * (self.history["service"].to_numpy() == ticket["service"])
        top = sims.argsort()[::-1][:k]
        similar = [{"incident_id": self.history.at[i, "incident_id"], "similarity": round(float(sims[i]), 3),
                    "short_description": self.history.at[i, "short_description"],
                    "cluster": int(self.history.at[i, "cluster"]) if "cluster" in self.history else None} for i in top]
        # Duplicate: same service, opened within the window, above threshold.
        dup = None
        opened = pd.Timestamp(ticket["opened_at"])
        win = pd.Timedelta(hours=self.policy["duplicate_window_hours"])
        recent = self.history[(self.history["service"] == ticket["service"]) & (self.history["opened_at"] <= opened)
                              & (self.history["opened_at"] >= opened - win)]
        if len(recent):
            s = sims[recent.index]
            j = int(s.argmax())
            if s[j] >= self.policy["duplicate_threshold"]:
                dup = recent.iloc[j]["incident_id"]
        return {"similar": similar, "possible_duplicate_of": dup}

    def _submit(self, suggestion: dict) -> dict:
        missing = [k for k in SUGGESTION_SCHEMA["required"] if k not in suggestion]
        if missing:
            raise ValueError(f"Suggestion is missing required fields: {missing}")
        if suggestion["severity"] not in SEVERITIES:
            raise ValueError("Invalid severity value")
        return {"status": "queued_for_review", "incident_id": suggestion["incident_id"]}

    # -- orchestration ------------------------------------------------------
    def triage(self, ticket: dict) -> dict:
        iid = ticket["incident_id"]
        sev_p = self.call("classify_severity", iid, ticket=ticket)
        team_p = self.call("suggest_owner", iid, ticket=ticket)
        sim = self.call("find_similar_incidents", iid, ticket=ticket)
        severity, sev_conf = max(sev_p.items(), key=lambda kv: kv[1])
        team, team_conf = max(team_p.items(), key=lambda kv: kv[1])
        route = self._route(severity, sev_conf, team_conf, sev_p)
        cluster = sim["similar"][0]["cluster"] if sim["similar"] and sim["similar"][0]["similarity"] >= 0.5 else None
        rationale = (f"Most likely {severity} ({sev_conf:.0%}) for {ticket['service']}; owner {team} ({team_conf:.0%}). "
                     + (f"Possible duplicate of {sim['possible_duplicate_of']}. " if sim["possible_duplicate_of"] else "")
                     + f"Routed to {route.replace('_', ' ')} under policy {self.policy['version']}.")
        suggestion = {
            "incident_id": iid, "severity": severity, "severity_confidence": sev_conf, "severity_probabilities": sev_p,
            "team": team, "team_confidence": team_conf, "similar_incidents": sim["similar"],
            "possible_duplicate_of": sim["possible_duplicate_of"], "recurring_cluster": cluster,
            "review_route": route, "rationale": rationale, "model_version": self.model_version,
        }
        self.call("submit_suggestion", iid, suggestion=suggestion)
        return suggestion

    def _route(self, severity: str, sev_conf: float, team_conf: float, sev_p: dict) -> str:
        p = self.policy
        # Any meaningful chance of SEV-1 goes straight to a human, whatever the top class is.
        if sev_p.get("SEV-1", 0) >= p["escalate_sev1_probability"]:
            return "escalate_now"
        if severity in p["always_review"]:
            return "full_review"
        # Fast path only when a high-severity outcome is unlikely and the owner is clear.
        p_high = sev_p.get("SEV-1", 0) + sev_p.get("SEV-2", 0)
        if p_high <= p["fast_path_max_high_severity_probability"] and team_conf >= p["fast_path_team_confidence"]:
            return "fast_path"
        return "full_review"

    # -- audit --------------------------------------------------------------
    def _log(self, tool, permission, allowed, incident_id, detail):
        ev = AuditEvent(str(uuid.uuid4()), time.time(), tool, permission.value, allowed, incident_id, detail)
        self.audit.append(ev)
        if self.audit_path:
            with open(self.audit_path, "a") as fh:
                fh.write(json.dumps(asdict(ev)) + "\n")


DEFAULT_POLICY = {
    "version": "policy-1.0",
    "fast_path_max_high_severity_probability": 0.10,
    "fast_path_team_confidence": 0.85,
    "always_review": ["SEV-1", "SEV-2"],
    "escalate_sev1_probability": 0.30,
    "duplicate_threshold": 0.55,
    "duplicate_window_hours": 6,
}
