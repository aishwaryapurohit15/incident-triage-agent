"""End-to-end pipeline: audit, clean, model, analyze, monitor and export.

Run from the repository root:

    PYTHONPATH=src python -m triage.pipeline
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from . import risk
from .agent import DEFAULT_POLICY, PermissionDenied, TriageAgent
from .config import DATA_PROCESSED, DATA_RAW, DOCS, MODELS, REPORTS, SEVERITIES, SLA_HOURS, TEAMS, TEST_START, VALID_START, VENDOR_SHIFT
from .models import RULES_SEVERITY, RULES_TEAM, TextClassifier, expected_calibration_error, review_curve, rules_predict, severity_metrics, team_metrics
from .quality import audit, clean, load_raw
from .similarity import similarity_vectorizer, cluster_incidents, cluster_quality, find_duplicate_candidates, repeat_rate, score_duplicates


# Readable names for clusters, as an analyst would label them after reading
# the tickets. Here the label is checked against the hidden root cause.
ROOT_CAUSE_LABELS = {
    "certificate_expiry": "Expired TLS certificates", "db_connection_pool": "Database connection exhaustion",
    "failed_deployment": "Failed releases and changes", "vpn_gateway": "VPN gateway failures", "disk_full": "Disk space exhaustion",
    "third_party_api": "Vendor API outages", "batch_failure": "Overnight batch failures", "dns_failure": "DNS resolution failures",
    "sso_outage": "Single sign-on and MFA failures", "memory_leak": "Memory leaks and crashes", "data_feed": "Late or bad data feeds",
    "phishing_alert": "Phishing and account compromise", "end_user": "End-user device and access requests",
    "network_latency": "Network latency and packet loss", "storage_replication": "Storage and replication lag",
    "maintenance_overrun": "Maintenance overruns",
}


def _round(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def run() -> dict:
    for p in (DATA_PROCESSED, REPORTS, MODELS, DOCS / "data"):
        p.mkdir(parents=True, exist_ok=True)
    out: dict = {}

    # 1. Audit and clean --------------------------------------------------
    raw = load_raw(DATA_RAW / "incidents.csv")
    out["quality"] = audit(raw)
    df = clean(raw)
    gt = pd.read_csv(DATA_RAW / "ground_truth.csv")
    df = df.merge(gt, on="incident_id", how="left")
    df.drop(columns=["root_cause", "true_severity", "true_team", "duplicate_of"]).to_csv(DATA_PROCESSED / "incidents_clean.csv", index=False)
    out["quality"]["rows_after_cleaning"] = int(len(df))
    print(f"Cleaned {len(df):,} rows")

    # 2. Time-based split ---------------------------------------------------
    train_all = df[df["opened_at"] < TEST_START]
    core = df[df["opened_at"] < VALID_START]
    valid = df[(df["opened_at"] >= VALID_START) & (df["opened_at"] < TEST_START)]
    test = df[df["opened_at"] >= TEST_START].copy()
    out["split"] = {"train_rows": int(len(train_all)), "validation_rows": int(len(valid)), "test_rows": int(len(test)),
                    "train_period": f"{df['opened_at'].min():%b %Y} to Apr 2026",
                    "validation_period": "Feb 2026 to Apr 2026", "test_period": f"May 2026 to {df['opened_at'].max():%b %Y}"}

    # 3. Model selection on validation -------------------------------------
    candidates = {"tfidf_baseline": dict(rich=False), "tfidf_challenger": dict(rich=True)}
    val_scores = {}
    for name, kw in candidates.items():
        m = TextClassifier(**kw).fit(core, core["severity"])
        val_scores[name] = severity_metrics(valid["severity"], m.predict(valid))
    champion_name = min(val_scores, key=lambda n: val_scores[n]["cost_per_ticket"])
    out["validation"] = {"severity": val_scores, "champion": champion_name, "selection_rule": "lowest misclassification cost per ticket"}
    print("Validation:", val_scores, "->", champion_name)

    # 4. Refit on all training data and score on the untouched test set ----
    team_train = train_all.dropna(subset=["assigned_team"])
    team_test = test.dropna(subset=["assigned_team"])
    fitted = {}
    sev_results, team_results = {}, {}
    sev_results["keyword_rules"] = severity_metrics(test["severity"], rules_predict(test["text"], RULES_SEVERITY, "SEV-3"), test["true_severity"])
    team_results["keyword_rules"] = team_metrics(team_test["assigned_team"], rules_predict(team_test["text"], RULES_TEAM, "End-User Computing"))
    sev_results["majority_class"] = severity_metrics(test["severity"], np.full(len(test), train_all["severity"].mode()[0]), test["true_severity"])
    for name, kw in candidates.items():
        sm = TextClassifier(**kw).fit(train_all, train_all["severity"])
        tm = TextClassifier(**kw).fit(team_train, team_train["assigned_team"])
        fitted[name] = (sm, tm)
        proba = sm.predict_proba(test)
        pred = np.array(sm.classes_)[proba.argmax(1)]
        r = severity_metrics(test["severity"], pred, test["true_severity"])
        r["calibration_error"] = _round(expected_calibration_error(proba, test["severity"], sm.classes_))
        sev_results[name] = r
        team_results[name] = team_metrics(team_test["assigned_team"], tm.predict(team_test))
        team_results[name]["accuracy_vs_true_team"] = _round(float((tm.predict(team_test) == team_test["true_team"]).mean()))
    out["test"] = {"severity": sev_results, "team": team_results}
    print("Test severity:", sev_results)
    print("Test team:", team_results)

    sev_model, team_model = fitted[champion_name]
    web_sev, web_team = fitted["tfidf_baseline"]
    test["pred_severity"] = sev_model.predict(test)
    test["pred_team"] = team_model.predict(test)
    sev_proba = sev_model.predict_proba(test)
    team_proba = team_model.predict_proba(test)
    out["confusion_matrix"] = {"labels": SEVERITIES, "matrix": confusion_matrix(test["severity"], test["pred_severity"], labels=SEVERITIES).tolist()}
    team_labels = [t for t in TEAMS]
    out["team_confusion"] = {"labels": team_labels, "matrix": confusion_matrix(team_test["assigned_team"], team_model.predict(team_test), labels=team_labels).tolist()}

    # 5. Human-review routing ----------------------------------------------
    curve = review_curve(sev_proba, test["severity"], sev_model.classes_)
    out["review_curve"] = curve
    # Fast-path policy: choose the largest "probability of SEV-1 or SEV-2" cap for
    # which, on validation, at most 3% of fast-path tickets were really SEV-1 or SEV-2.
    vm = TextClassifier(**candidates[champion_name]).fit(core, core["severity"])
    vproba = vm.predict_proba(valid)
    vtm = TextClassifier(**candidates[champion_name]).fit(core.dropna(subset=["assigned_team"]), core.dropna(subset=["assigned_team"])["assigned_team"])
    vteam_conf = vtm.predict_proba(valid).max(1)
    vcls = list(vm.classes_)
    v_high = vproba[:, vcls.index("SEV-1")] + vproba[:, vcls.index("SEV-2")]
    v_true_high = valid["severity"].isin(["SEV-1", "SEV-2"]).to_numpy()
    policy = dict(DEFAULT_POLICY)
    cap_search = []
    for cap in np.round(np.arange(0.05, 0.41, 0.025), 3):
        m = (v_high <= cap) & (vteam_conf >= policy["fast_path_team_confidence"])
        cap_search.append({"cap": float(cap), "fast_path_share": _round(m.mean()), "high_severity_leakage": _round(v_true_high[m].mean() if m.any() else 0.0)})
    ok = [c for c in cap_search if c["high_severity_leakage"] <= 0.03]
    policy["fast_path_max_high_severity_probability"] = ok[-1]["cap"] if ok else 0.05
    out["policy_search"] = cap_search
    valid_override = float((np.array(vm.classes_)[vproba.argmax(1)] != valid["severity"]).mean())

    # 6. Duplicate detection: threshold chosen on validation, scored on test --
    sim_vec = similarity_vectorizer(train_all["text"])
    vcands = find_duplicate_candidates(valid, sim_vec, window_hours=policy["duplicate_window_hours"])
    vbest = max(score_duplicates(vcands, gt), key=lambda r: r["f1"])
    policy["duplicate_threshold"] = vbest["threshold"]
    cands = find_duplicate_candidates(test, sim_vec, window_hours=policy["duplicate_window_hours"])
    dup_curve = score_duplicates(cands, gt)
    chosen = next(r for r in dup_curve if r["threshold"] == vbest["threshold"])
    # Staff baseline: tickets whose notes already say "Duplicate of".
    true_dup = test["duplicate_of"].notna()
    staff_flag = test["is_flagged_duplicate"]
    staff = {"precision": _round(float((staff_flag & true_dup).sum() / max(staff_flag.sum(), 1))),
             "recall": _round(float((staff_flag & true_dup).sum() / max(true_dup.sum(), 1)))}
    agent_flag = cands.set_index("incident_id").loc[test["incident_id"], "similarity"].to_numpy() >= vbest["threshold"]
    missed = true_dup & ~staff_flag
    staff["agent_catches_staff_missed"] = int((missed & agent_flag).sum())
    staff["staff_missed"] = int(missed.sum())
    out["duplicates"] = {"curve": dup_curve, "chosen": chosen, "true_duplicates_in_test": int(true_dup.sum()),
                         "staff_flagging": staff}
    out["policy"] = policy

    # Route every test ticket with the same policy the agent uses.
    classes = np.array(sev_model.classes_)
    sev_conf = sev_proba.max(1)
    team_conf = team_proba.max(1)
    p_sev1 = sev_proba[:, list(classes).index("SEV-1")]
    p_high = p_sev1 + sev_proba[:, list(classes).index("SEV-2")]
    route = np.where(p_sev1 >= policy["escalate_sev1_probability"], "escalate_now",
                     np.where(np.isin(test["pred_severity"], policy["always_review"]), "full_review",
                              np.where((p_high <= policy["fast_path_max_high_severity_probability"]) & (team_conf >= policy["fast_path_team_confidence"]), "fast_path", "full_review")))
    test["route"] = route
    fp = test["route"] == "fast_path"
    true_sev1 = test["severity"] == "SEV-1"
    out["routing"] = {
        "share": {r: _round(float((route == r).mean())) for r in ["fast_path", "full_review", "escalate_now"]},
        "fast_path_severity_accuracy": _round(float((test.loc[fp, "pred_severity"] == test.loc[fp, "severity"]).mean())),
        "fast_path_high_severity_leakage": _round(float(test.loc[fp, "severity"].isin(["SEV-1", "SEV-2"]).mean())),
        "fast_path_team_accuracy": _round(float((test.loc[fp, "pred_team"] == test.loc[fp, "assigned_team"]).mean())),
        "sev1_caught_by_escalation_or_review": _round(float((test.loc[true_sev1, "route"] != "fast_path").mean())),
        "sev1_escalated_now": _round(float((test.loc[true_sev1, "route"] == "escalate_now").mean())),
    }
    # Triage effort: assume 6 minutes for a full manual triage, 1.5 to confirm a fast-path suggestion,
    # and 4 for a full review that starts from a suggestion.
    minutes_manual = 6 * len(test)
    minutes_agent = 1.5 * fp.sum() + 4 * (~fp).sum()
    out["routing"]["triage_minutes_saved_pct"] = _round(1 - minutes_agent / minutes_manual)
    out["routing"]["triage_hours_saved_test_period"] = round((minutes_manual - minutes_agent) / 60, 0)

    # 7. Recurring-issue clustering ----------------------------------------
    cl = cluster_incidents(df)
    df["cluster"] = cl["labels"]
    test["cluster"] = df.set_index("incident_id").loc[test["incident_id"], "cluster"].to_numpy()
    out["clustering"] = {"k_search": cl["k_search"], "best_k": cl["best_k"], **cluster_quality(cl["labels"], df["root_cause"])}
    rep = repeat_rate(df)
    clusters = []
    for c, g in df.groupby("cluster"):
        monthly = g.groupby("month").size()
        first, last = monthly.iloc[:6].mean(), monthly.iloc[-6:].mean()
        dominant = g["root_cause"].value_counts(normalize=True)
        clusters.append({
            "cluster": int(c), "top_terms": cl["top_terms"][c], "incidents": int(len(g)),
            "sev1_sev2_share": _round(g["severity"].isin(["SEV-1", "SEV-2"]).mean()),
            "total_loss_usd": round(float(g["financial_loss_usd"].sum()), 0),
            "repeat_within_7_days": _round(rep.get(c, np.nan)),
            "trend_ratio_last6_vs_first6": _round(last / first if first else np.nan, 3),
            "top_service": g["service"].mode()[0], "top_team": g["assigned_team"].mode()[0],
            "dominant_root_cause": dominant.index[0], "label": ROOT_CAUSE_LABELS[dominant.index[0]], "purity": _round(dominant.iloc[0]),
        })
    clusters.sort(key=lambda x: -x["total_loss_usd"])
    out["clusters"] = clusters

    # 8. Operational risk analytics ----------------------------------------
    months = df["month"].nunique()
    daily = risk.volume_with_anomalies(df)
    sla = risk.sla_summary(df)
    out["risk"] = {
        "months": int(months),
        "incidents_per_month": round(len(df) / months, 1),
        "sla_by_severity": sla.reset_index(names="severity").to_dict("records"),
        "sla_by_team": risk.team_sla(df).reset_index(names="team").to_dict("records"),
        "overall_sla_breach_rate": _round(float((df["hours_to_resolve"] > df["severity"].map(SLA_HOURS))[df["hours_to_resolve"].notna()].mean())),
        "service_risk": risk.service_risk(df, months).reset_index(names="service").to_dict("records"),
        "heatmap": {"services": list(risk.service_heatmap(df).index), "severities": SEVERITIES,
                    "counts": risk.service_heatmap(df).to_numpy().tolist()},
        "loss_distribution": risk.loss_distribution(df),
        "change_risk": risk.change_risk(df),
        "category_shift": risk.category_shift(df, VENDOR_SHIFT),
        "total_loss_usd": round(float(df["financial_loss_usd"].sum()), 0),
        "customer_impacting_share": _round(df["customer_impact"].mean()),
        "reopen_rate": _round(df["reopened"].mean()),
        "outage_check": _outage_check(df, daily),
        "anomaly_days": [{"date": str(d.date()), "incidents": int(r.incidents), "limit": int(r.upper_limit)}
                         for d, r in daily[daily["anomaly"]].iterrows()],
        "daily": {"dates": [str(d.date()) for d in daily.index], "incidents": daily["incidents"].astype(int).tolist(),
                  "limit": [None if np.isnan(v) else int(v) for v in daily["upper_limit"]], "anomaly": daily["anomaly"].tolist()},
        "monthly_by_severity": {s: df[df["severity"] == s].groupby("month").size().reindex(sorted(df["month"].unique()), fill_value=0).tolist() for s in SEVERITIES},
        "months_list": sorted(df["month"].unique()),
        "basel_mix": df.groupby("basel_event_type").agg(incidents=("incident_id", "size"), loss=("financial_loss_usd", "sum")).round(0).reset_index().to_dict("records"),
    }

    # 9. Monitoring over the test months -----------------------------------
    ref_pred = pd.Series(sev_model.predict(train_all.iloc[-3000:])).value_counts()
    ref_cat = train_all["category_clean"].value_counts()
    mon = []
    for month, g in test.groupby("month"):
        mon.append({
            "month": month, "tickets": int(len(g)),
            "override_rate": _round(float((g["pred_severity"] != g["severity"]).mean())),
            "team_override_rate": _round(float((g["pred_team"] != g["assigned_team"]).dropna().mean())),
            "psi_predicted_severity": _round(risk.psi(ref_pred, g["pred_severity"].value_counts())),
            "psi_category_mix": _round(risk.psi(ref_cat, g["category_clean"].value_counts())),
            "fast_path_share": _round(float((g["route"] == "fast_path").mean())),
        })
    # Override alert: validation override rate plus 5 points; watch at plus 3.
    thresholds = {"validation_override_rate": _round(valid_override), "override_watch": _round(valid_override + 0.03),
                  "override_alert": _round(valid_override + 0.05), "psi_watch": 0.10, "psi_alert": 0.25}
    for m in mon:
        flags, level = [], 0
        psi_max = max(m["psi_predicted_severity"], m["psi_category_mix"])
        if m["override_rate"] > thresholds["override_alert"]:
            flags.append("Severity override rate above the alert limit"); level = 2
        elif m["override_rate"] > thresholds["override_watch"]:
            flags.append("Severity override rate above the watch limit"); level = max(level, 1)
        if psi_max > thresholds["psi_alert"]:
            flags.append("Population shift: PSI above 0.25"); level = 2
        elif psi_max > thresholds["psi_watch"]:
            flags.append("Drift watch: PSI above 0.10"); level = max(level, 1)
        m["status"] = ["ok", "watch", "alert"][level]
        m["flags"] = flags
    out["monitoring"] = {"months": mon, "thresholds": thresholds}

    # 10. Agent run on sample tickets, including a refused write ----------
    history = train_all.merge(df[["incident_id", "cluster"]], on="incident_id", how="left")
    version = f"{champion_name}-v1"
    agent = TriageAgent(sev_model, team_model, history, policy, version, audit_path=REPORTS / "agent_audit.jsonl")
    (REPORTS / "agent_audit.jsonl").unlink(missing_ok=True)
    sample = pd.concat([test[test["severity"] == s].sample(3, random_state=7) for s in SEVERITIES])
    examples = []
    for _, r in sample.iterrows():
        ticket = {k: (str(r[k]) if k == "opened_at" else r[k]) for k in ["incident_id", "opened_at", "service", "channel", "short_description", "description"]}
        s = agent.triage(ticket)
        s["human_recorded_severity"], s["human_recorded_team"] = r["severity"], r["assigned_team"] if isinstance(r["assigned_team"], str) else None
        s["ticket"] = ticket
        examples.append(s)
    try:
        agent.call("close_ticket", sample.iloc[0]["incident_id"])
        refusal = None
    except PermissionDenied as e:
        refusal = str(e)
    out["agent"] = {"examples": examples, "refusal_example": refusal, "audit_events": len(agent.audit),
                    "tools": [{"name": t.name, "permission": t.permission.value, "description": t.description,
                               "granted": t.permission.value in ("read", "suggest")} for t in agent.tools.values()]}

    # 11. Persist models and exports ----------------------------------------
    with open(MODELS / "champion.pkl", "wb") as fh:
        pickle.dump({"severity": sev_model, "team": team_model, "policy": policy, "version": version}, fh)
    with open(MODELS / "registry.json", "w") as fh:
        json.dump({"champion": version, "fallback": "keyword_rules", "policy": policy["version"], "kill_switch": False,
                   "notes": "Set kill_switch to true to stop suggestions and fall back to manual triage."}, fh, indent=2)
    out["web_model"] = export_web_model(web_sev, team_train, team_test, history, policy)
    with open(REPORTS / "metrics.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    with open(DOCS / "data" / "site_data.js", "w") as fh:
        fh.write("window.SITE_DATA = " + json.dumps(out, default=str, separators=(",", ":")) + ";\n")
    print("Wrote reports/metrics.json, docs/data/site_data.js and docs/data/web_model.js")
    return out


def _outage_check(df: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """How many planted major-outage days the control chart caught (uses ground truth)."""
    parents = df["duplicate_of"].value_counts()
    big = parents[parents >= 8].index
    outage_days = set(df.loc[df["incident_id"].isin(big), "opened_at"].dt.normalize())
    flagged = set(daily.index[daily["anomaly"]])
    month_start = {d for d in flagged if d.day <= 2}
    return {"planted_outage_days": len(outage_days), "caught": len(outage_days & flagged),
            "flagged_days": len(flagged), "flagged_month_start": len(month_start - outage_days)}


def export_web_model(sev_model, team_train: pd.DataFrame, team_test: pd.DataFrame, history: pd.DataFrame, policy: dict) -> dict:
    """Export the TF-IDF baseline so the website can run it in the browser."""
    vocab = sev_model.word.vocabulary_
    terms = [None] * len(vocab)
    for t, i in vocab.items():
        terms[i] = t
    from sklearn.preprocessing import normalize

    from .similarity import similarity_vectorizer

    sim_vec = similarity_vectorizer(history["text"])
    sim_terms = [None] * len(sim_vec.vocabulary_)
    for t, i in sim_vec.vocabulary_.items():
        sim_terms[i] = t
    sample = history.sample(1500, random_state=1)
    X = sim_vec.transform(sample["text"])

    X = normalize(X).tocsr()
    hist = []
    for i, (_, r) in enumerate(sample.iterrows()):
        row = X.getrow(i)
        hist.append({"id": r["incident_id"], "s": r["service"], "d": r["short_description"], "v": r["severity"],
                     "t": r["assigned_team"] if isinstance(r["assigned_team"], str) else "", "c": int(r["cluster"]),
                     "k": row.indices.tolist(), "w": [round(float(x), 3) for x in row.data]})
    # Refit the team classifier on the severity model's vocabulary so the browser
    # computes one TF-IDF vector for both predictions.
    from sklearn.linear_model import LogisticRegression

    web_team = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced", random_state=1)
    web_team.fit(sev_model.word.transform(team_train["text"]), team_train["assigned_team"])
    web_team_acc = float((web_team.predict(sev_model.word.transform(team_test["text"])) == team_test["assigned_team"]).mean())
    web = {
        "terms": terms,
        "idf": [round(float(x), 4) for x in sev_model.word.idf_],
        "severity": {"classes": sev_model.classes_, "coef": np.round(sev_model.clf.coef_, 3).tolist(),
                     "intercept": np.round(sev_model.clf.intercept_, 4).tolist()},
        "team": {"classes": list(web_team.classes_), "coef": np.round(web_team.coef_, 3).tolist(),
                 "intercept": np.round(web_team.intercept_, 4).tolist()},
        "sim_terms": sim_terms,
        "stop_words": sorted(__import__("sklearn.feature_extraction.text", fromlist=["ENGLISH_STOP_WORDS"]).ENGLISH_STOP_WORDS),
        "sim_idf": [round(float(x), 4) for x in sim_vec.idf_],
        "history": hist,
        "policy": policy,
    }
    with open(DOCS / "data" / "web_model.js", "w") as fh:
        fh.write("window.WEB_MODEL = " + json.dumps(web, separators=(",", ":")) + ";\n")
    return {"team_accuracy": round(web_team_acc, 4), "vocabulary_size": len(terms), "history_sample": len(hist)}


if __name__ == "__main__":
    run()
