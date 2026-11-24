"""LLM classifier adapter (the challenger in the method comparison).

This module sends a ticket to a large language model and asks for a JSON
answer that follows the same contract as the TF-IDF models. It needs the
`anthropic` package and an `ANTHROPIC_API_KEY` environment variable. No LLM
results are included in the repository; run `scripts/run_llm_eval.py` to
produce them on a sample of the test set.
"""

from __future__ import annotations

import json
import os
import time

from .config import SEVERITIES, SEVERITY_NAMES, TEAMS

PROMPT = """You are an IT incident triage assistant at a bank. Read the ticket and suggest a severity and an owning team.

Severity definitions:
{severities}

Teams: {teams}

Ticket
Service: {service}
Channel: {channel}
Opened: {opened_at}
Short description: {short_description}
Description: {description}

Reply with JSON only, in this exact form:
{{"severity": "SEV-1|SEV-2|SEV-3|SEV-4", "team": "<one team from the list>", "confidence": <0 to 1>, "reason": "<one sentence>"}}"""

SEVERITY_GUIDE = {
    "SEV-1": "customer-facing critical service unavailable or major financial exposure",
    "SEV-2": "significant degradation or partial outage of an important service",
    "SEV-3": "limited impact, workaround available, or internal service affected",
    "SEV-4": "single user or cosmetic issue, service request",
}


def build_prompt(ticket: dict) -> str:
    sev = "\n".join(f"- {s} ({SEVERITY_NAMES[s]}): {SEVERITY_GUIDE[s]}" for s in SEVERITIES)
    return PROMPT.format(severities=sev, teams=", ".join(TEAMS), channel=ticket.get("channel", "Unknown"), **{
        k: ticket[k] for k in ["service", "opened_at", "short_description", "description"]})


def parse_response(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1])
    if data.get("severity") not in SEVERITIES:
        raise ValueError(f"Invalid severity: {data.get('severity')}")
    if data.get("team") not in TEAMS:
        raise ValueError(f"Invalid team: {data.get('team')}")
    return data


class LLMClassifier:
    def __init__(self, model: str | None = None, max_retries: int = 2):
        import anthropic  # imported lazily so the rest of the project runs without it

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY to run the LLM classifier.")
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("TRIAGE_LLM_MODEL", "claude-haiku-5-5")
        self.max_retries = max_retries

    def classify(self, ticket: dict) -> dict:
        last_error = None
        for _ in range(self.max_retries + 1):
            t0 = time.time()
            msg = self.client.messages.create(model=self.model, max_tokens=200,
                                              messages=[{"role": "user", "content": build_prompt(ticket)}])
            try:
                out = parse_response(msg.content[0].text)
                out["latency_s"] = round(time.time() - t0, 3)
                out["input_tokens"] = msg.usage.input_tokens
                out["output_tokens"] = msg.usage.output_tokens
                return out
            except (ValueError, json.JSONDecodeError) as e:  # schema violation: retry, then fail closed
                last_error = e
        return {"severity": None, "team": None, "error": str(last_error)}
