"""Triage one ticket from the command line with the trained champion model.

Usage (from the repository root, after running the pipeline):

    PYTHONPATH=src python scripts/triage_ticket.py \\
        --service "Payments Gateway" \\
        --short "Card payments failing" \\
        --description "About 40% of transactions are failing. Many users are affected."
"""

import argparse
import json
import pickle

import pandas as pd

from triage.agent import TriageAgent
from triage.config import DATA_PROCESSED, MODELS, TEST_START
from triage.quality import normalize_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--service", required=True)
    ap.add_argument("--short", required=True)
    ap.add_argument("--description", required=True)
    ap.add_argument("--channel", default="Service Portal")
    ap.add_argument("--opened-at", default=str(pd.Timestamp.now().floor("min")))
    args = ap.parse_args()

    registry = json.loads((MODELS / "registry.json").read_text())
    if registry.get("kill_switch"):
        print("Kill switch is on: suggestions are disabled and tickets go to manual triage.")
        return
    bundle = pickle.loads((MODELS / "champion.pkl").read_bytes())
    hist = pd.read_csv(DATA_PROCESSED / "incidents_clean.csv", parse_dates=["opened_at"])
    hist = hist[hist["opened_at"] < TEST_START].copy()
    hist["text"] = (hist["short_description"].fillna("") + ". " + hist["description"].fillna("")).map(normalize_text)
    agent = TriageAgent(bundle["severity"], bundle["team"], hist, bundle["policy"], bundle["version"])
    ticket = {"incident_id": "INC0000000", "opened_at": args.opened_at, "service": args.service, "channel": args.channel,
              "short_description": args.short, "description": args.description}
    print(json.dumps(agent.triage(ticket), indent=2, default=str))


if __name__ == "__main__":
    main()
