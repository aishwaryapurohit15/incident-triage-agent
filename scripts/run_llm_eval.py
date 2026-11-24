"""Score the LLM classifier on a random sample of the test period.

Usage (from the repository root):

    export ANTHROPIC_API_KEY=...
    PYTHONPATH=src python scripts/run_llm_eval.py --n 300

Writes reports/llm_eval.json with accuracy, cost-weighted error, latency and
token usage, so the LLM can be compared with the TF-IDF models on the same
tickets. This script is not run by the main pipeline because it calls a paid API.
"""

import argparse
import json

import numpy as np
import pandas as pd

from triage.config import DATA_PROCESSED, REPORTS, TEST_START
from triage.llm import LLMClassifier
from triage.models import severity_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    df = pd.read_csv(DATA_PROCESSED / "incidents_clean.csv", parse_dates=["opened_at"])
    test = df[df["opened_at"] >= TEST_START].sample(args.n, random_state=11)
    llm = LLMClassifier(model=args.model)
    rows = []
    for _, r in test.iterrows():
        ticket = {k: str(r[k]) for k in ["service", "channel", "opened_at", "short_description", "description"]}
        rows.append(llm.classify(ticket))
    res = pd.DataFrame(rows)
    ok = res["severity"].notna().to_numpy()
    out = {
        "model": llm.model,
        "tickets": int(len(test)),
        "schema_failures": int((~ok).sum()),
        "severity": severity_metrics(test["severity"].to_numpy()[ok], res.loc[ok, "severity"].to_numpy()),
        "team_accuracy": round(float((res.loc[ok, "team"].to_numpy() == test["assigned_team"].to_numpy()[ok]).mean()), 4),
        "median_latency_s": round(float(np.median(res.loc[ok, "latency_s"])), 3),
        "mean_input_tokens": round(float(res.loc[ok, "input_tokens"].mean()), 1),
        "mean_output_tokens": round(float(res.loc[ok, "output_tokens"].mean()), 1),
    }
    (REPORTS / "llm_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
