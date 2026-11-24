"""Data-quality audit and cleaning for the raw incident export."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Map messy reporter labels onto a controlled category list.
CATEGORY_MAP = {
    "security": "Security", "sec": "Security", "certificates": "Security", "phishing": "Security",
    "database": "Database", "db": "Database", "databse": "Database",
    "application": "Application", "app": "Application", "software": "Application", "performance": "Application",
    "network": "Network", "nw": "Network", "netwrk": "Network", "connectivity": "Network", "dns": "Network", "telecom": "Network",
    "infrastructure": "Infrastructure", "infra": "Infrastructure", "storage": "Infrastructure", "server": "Infrastructure",
    "vendor": "Third Party", "third party": "Third Party", "3rd party": "Third Party", "external": "Third Party",
    "batch": "Data", "batch job": "Data", "data": "Data", "feeds": "Data",
    "access": "Access", "login": "Access",
    "change": "Change", "maintenance": "Change",
    "hardware": "End-User", "desktop": "End-User", "email": "End-User", "other": "End-User",
}

ABBREVIATIONS = {
    r"\bdb\b": "database", r"\bprod\b": "production", r"\bpls\b": "please", r"\bapp\b": "application",
    r"\basap\b": "as soon as possible", r"\bdc\b": "data center",
}


def load_raw(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["opened_at", "resolved_at"])


def audit(df: pd.DataFrame) -> dict:
    """Return a dictionary of data-quality findings on the raw export."""
    n = len(df)
    exact_dupes = int(df.duplicated().sum())
    neg = int((df["resolved_at"] < df["opened_at"]).sum())
    open_ = int(df["resolved_at"].isna().sum())
    raw_categories = int(df["category"].nunique())
    clean_categories = int(df["category"].str.strip().str.lower().map(CATEGORY_MAP).nunique())
    vague = int(df["description"].str.len().lt(35).sum())
    return {
        "rows": n,
        "columns": int(df.shape[1]),
        "first_incident": str(df["opened_at"].min().date()),
        "last_incident": str(df["opened_at"].max().date()),
        "exact_duplicate_rows": exact_dupes,
        "missing_resolution_notes": int(df["resolution_notes"].isna().sum()),
        "missing_resolution_notes_pct": round(100 * df["resolution_notes"].isna().mean(), 1),
        "unassigned_tickets": int(df["assigned_team"].isna().sum()),
        "resolved_before_opened": neg,
        "never_resolved": open_,
        "raw_category_labels": raw_categories,
        "clean_category_labels": clean_categories,
        "vague_descriptions": vague,
        "vague_descriptions_pct": round(100 * vague / n, 1),
        "duplicate_note_rows": int(df["resolution_notes"].fillna("").str.startswith("Duplicate of").sum()),
    }


def normalize_text(text: str) -> str:
    text = str(text).lower()
    for pattern, repl in ABBREVIATIONS.items():
        text = re.sub(pattern, repl, text)
    text = re.sub(r"\b(chg|inc)\d+\b", " changeid ", text)
    text = re.sub(r"\b\d+(\.\d+)*\b", " num ", text)
    text = re.sub(r"[^a-z0-9_\-\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning rules documented in the README."""
    out = df.drop_duplicates().copy()
    out["category_clean"] = out["category"].str.strip().str.lower().map(CATEGORY_MAP).fillna("Other")
    bad = out["resolved_at"] < out["opened_at"]
    out.loc[bad, "resolved_at"] = pd.NaT  # treat impossible timestamps as unknown
    out["hours_to_resolve"] = (out["resolved_at"] - out["opened_at"]).dt.total_seconds() / 3600
    out["text"] = (out["short_description"].fillna("") + ". " + out["description"].fillna("")).map(normalize_text)
    out["hour"] = out["opened_at"].dt.hour
    out["weekday"] = out["opened_at"].dt.dayofweek
    out["business_hours"] = ((out["hour"].between(8, 18)) & (out["weekday"] < 5)).astype(int)
    out["month"] = out["opened_at"].dt.to_period("M").astype(str)
    out["is_flagged_duplicate"] = out["resolution_notes"].fillna("").str.startswith("Duplicate of")
    return out.reset_index(drop=True)


def sla_breach(df: pd.DataFrame, sla_hours: dict) -> pd.Series:
    target = df["severity"].map(sla_hours)
    return np.where(df["hours_to_resolve"].isna(), np.nan, (df["hours_to_resolve"] > target).astype(float))
