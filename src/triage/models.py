"""Severity and owner classification: rules baseline, TF-IDF baseline and challenger."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import OneHotEncoder

from .config import MISCLASSIFICATION_COST, RANDOM_SEED, SEVERITIES, TEAMS

META_COLS = ["service", "channel"]

# ---------------------------------------------------------------------------
# Rules baseline: the kind of keyword logic a service desk writes by hand.
# ---------------------------------------------------------------------------
RULES_SEVERITY = [
    (r"outage|down|unavailable|all customers|crash|503|500 errors|cannot complete", "SEV-2"),
    (r"slow|latency|delay|intermittent|lag|timeout", "SEV-3"),
    (r"password|printer|laptop|headset|monitor|access to|outlook|camera|payslip", "SEV-4"),
]
RULES_TEAM = [
    (r"vpn|dns|latency|packet|wan|network|circuit|switch", "Network Operations"),
    (r"database|connection|replica|replication|standby|sessions", "Database Administration"),
    (r"certificate|cert|tls|ssl|sso|login|mfa|saml|phishing|credential|suspicious", "Identity and Security Operations"),
    (r"vendor|third-party|provider|cardnet|payflow|marketlink|televox|cloudmail|findata", "Third-Party Vendor Management"),
    (r"batch|etl|feed|disk|filesystem|space|job|reconciliation", "Data Platform Engineering"),
    (r"release|deploy|rollback|feature flag|health checks", "Digital Channels Support"),
    (r"memory|jvm|heap|oom|maintenance|patching|crash", "Payments Application Support"),
]


def rules_predict(texts: pd.Series, rules: list[tuple[str, str]], default: str) -> np.ndarray:
    out = []
    for t in texts:
        label = default
        for pattern, value in rules:
            if re.search(pattern, t):
                label = value
                break
        out.append(label)
    return np.array(out)


# ---------------------------------------------------------------------------
# Learned models
# ---------------------------------------------------------------------------
@dataclass
class TextClassifier:
    """TF-IDF + logistic regression. `rich=True` adds character n-grams and metadata."""

    rich: bool = False
    max_features: int = 6000
    C: float = 4.0

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "TextClassifier":
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=self.max_features, sublinear_tf=True)
        mats = [self.word.fit_transform(df["text"])]
        if self.rich:
            self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=30000, sublinear_tf=True)
            self.ohe = OneHotEncoder(handle_unknown="ignore")
            mats.append(self.char.fit_transform(df["text"]))
            mats.append(self.ohe.fit_transform(df[META_COLS]))
            mats.append(sparse.csr_matrix(df[["business_hours"]].to_numpy(dtype=float)))
        X = sparse.hstack(mats).tocsr()
        self.clf = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced", random_state=RANDOM_SEED)
        self.clf.fit(X, y)
        self.classes_ = list(self.clf.classes_)
        return self

    def _X(self, df: pd.DataFrame):
        mats = [self.word.transform(df["text"])]
        if self.rich:
            mats += [self.char.transform(df["text"]), self.ohe.transform(df[META_COLS]),
                     sparse.csr_matrix(df[["business_hours"]].to_numpy(dtype=float))]
        return sparse.hstack(mats).tocsr()

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self._X(df))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array(self.classes_)[self.predict_proba(df).argmax(1)]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def cost_per_ticket(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=SEVERITIES)
    return float((cm * np.array(MISCLASSIFICATION_COST)).sum() / cm.sum())


def expected_calibration_error(proba: np.ndarray, y_true, classes, bins: int = 10) -> float:
    conf = proba.max(1)
    pred = np.array(classes)[proba.argmax(1)]
    correct = (pred == np.asarray(y_true)).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def severity_metrics(y_true, y_pred, y_true_hidden=None) -> dict:
    out = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "sev1_recall": round(recall_score(y_true, y_pred, labels=["SEV-1"], average="macro", zero_division=0), 4),
        "under_triage_rate": round(float(np.mean([SEVERITIES.index(p) > SEVERITIES.index(t) for t, p in zip(y_true, y_pred)])), 4),
        "cost_per_ticket": round(cost_per_ticket(y_true, y_pred), 3),
    }
    if y_true_hidden is not None:
        out["accuracy_vs_true_severity"] = round(accuracy_score(y_true_hidden, y_pred), 4)
    return out


def team_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", labels=TEAMS, zero_division=0), 4),
    }


def review_curve(proba: np.ndarray, y_true, classes, thresholds=None) -> list[dict]:
    """Coverage versus accuracy when only confident suggestions take the fast path."""
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.30, 0.96, 0.05), 2)
    conf = proba.max(1)
    pred = np.array(classes)[proba.argmax(1)]
    y = np.asarray(y_true)
    rows = []
    for t in thresholds:
        m = conf >= t
        rows.append({
            "threshold": float(t),
            "fast_path_share": round(float(m.mean()), 4),
            "fast_path_accuracy": round(float((pred[m] == y[m]).mean()) if m.any() else float("nan"), 4),
        })
    return rows
