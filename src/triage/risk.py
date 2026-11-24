"""Operational risk analytics on the cleaned incident log."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import RANDOM_SEED, SERVICES, SEVERITIES, SLA_HOURS


def volume_with_anomalies(df: pd.DataFrame, alpha: float = 0.001) -> pd.DataFrame:
    """Daily incident counts with a Poisson control limit from a trailing baseline.

    The baseline is the median of the previous 20 weekdays (or 8 weekend
    days) and excludes the current day. The limit is the (1 - alpha)
    Poisson quantile of the trailing mean, so a day is flagged only when its
    volume is very unlikely under recent normal conditions.
    """
    daily = df.set_index("opened_at").resample("D").size().rename("incidents").to_frame()
    weekday = daily.index.dayofweek < 5
    # Separate baselines for weekdays and weekends because volumes differ.
    base = pd.Series(np.nan, index=daily.index)
    for mask, w in ((weekday, 20), (~weekday, 8)):
        s = daily.loc[mask, "incidents"]
        base.loc[mask] = s.shift(1).rolling(w, min_periods=6).median()
    daily["baseline"] = base
    daily["upper_limit"] = stats.poisson.ppf(1 - alpha, daily["baseline"].fillna(daily["incidents"].median()))
    daily["anomaly"] = daily["incidents"] > daily["upper_limit"]
    return daily


def sla_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["hours_to_resolve"]).copy()
    d["breach"] = d["hours_to_resolve"] > d["severity"].map(SLA_HOURS)
    g = d.groupby("severity")
    return pd.DataFrame({
        "incidents": g.size(),
        "median_hours": g["hours_to_resolve"].median().round(2),
        "p90_hours": g["hours_to_resolve"].quantile(0.9).round(2),
        "sla_hours": pd.Series(SLA_HOURS),
        "breach_rate": g["breach"].mean().round(4),
    }).reindex(SEVERITIES)


def team_sla(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["hours_to_resolve", "assigned_team"]).copy()
    d["breach"] = d["hours_to_resolve"] > d["severity"].map(SLA_HOURS)
    g = d.groupby("assigned_team")
    return pd.DataFrame({"incidents": g.size(), "breach_rate": g["breach"].mean().round(4),
                         "median_hours": g["hours_to_resolve"].median().round(2)}).sort_values("breach_rate", ascending=False)


def service_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df["service"], df["severity"]).reindex(columns=SEVERITIES, fill_value=0)


def service_risk(df: pd.DataFrame, months: float) -> pd.DataFrame:
    """Likelihood (incidents per month) versus impact (mean loss per impacting incident)."""
    g = df.groupby("service")
    impact = df[df["financial_loss_usd"] > 0].groupby("service")["financial_loss_usd"]
    out = pd.DataFrame({
        "incidents_per_month": (g.size() / months).round(1),
        "sev1_sev2_share": g["severity"].apply(lambda s: s.isin(["SEV-1", "SEV-2"]).mean()).round(4),
        "mean_loss_usd": impact.mean().round(0),
        "total_loss_usd": g["financial_loss_usd"].sum().round(0),
        "criticality": pd.Series(SERVICES),
    }).fillna(0)
    return out.sort_values("total_loss_usd", ascending=False)


def loss_distribution(df: pd.DataFrame, sims: int = 50_000, seed: int = RANDOM_SEED) -> dict:
    """Loss distribution approach: Poisson frequency x lognormal severity per Basel event type.

    Simulates annual aggregate operational loss and reports expected loss,
    value at risk and expected shortfall.
    """
    rng = np.random.default_rng(seed)
    losses = df[df["financial_loss_usd"] > 0]
    months = df["opened_at"].dt.to_period("M").nunique()
    by_type, total = {}, np.zeros(sims)
    for event_type, g in losses.groupby("basel_event_type"):
        lam_year = len(g) / months * 12
        shape, loc, scale = stats.lognorm.fit(g["financial_loss_usd"], floc=0)
        mu, sigma = np.log(scale), shape
        counts = rng.poisson(lam_year, size=sims)
        annual = np.array([rng.lognormal(mu, sigma, size=c).sum() for c in counts])
        total += annual
        # Goodness of fit on the log scale.
        ks = stats.kstest(np.log(g["financial_loss_usd"]), stats.norm(loc=mu, scale=sigma).cdf)
        by_type[event_type] = {
            "loss_events": int(len(g)), "annual_frequency": round(lam_year, 1), "lognormal_mu": round(mu, 3),
            "lognormal_sigma": round(sigma, 3), "ks_pvalue": round(float(ks.pvalue), 4),
            "expected_annual_loss": round(float(annual.mean()), 0), "var_99_9": round(float(np.quantile(annual, 0.999)), 0),
        }
    q = lambda p: float(np.quantile(total, p))  # noqa: E731
    var999 = q(0.999)
    # Histogram up to the 99.95th percentile so the body and tail are both readable.
    hist, edges = np.histogram(total[total <= np.quantile(total, 0.9995)], bins=48)
    return {
        "by_event_type": by_type,
        "expected_annual_loss": round(float(total.mean()), 0),
        "var_95": round(q(0.95), 0), "var_99": round(q(0.99), 0), "var_99_9": round(var999, 0),
        "es_99_9": round(float(total[total >= var999].mean()), 0),
        "unexpected_loss_99_9": round(var999 - float(total.mean()), 0),
        "observed_annual_loss": round(float(losses["financial_loss_usd"].sum() / months * 12), 0),
        "histogram": {"counts": hist.tolist(), "edges": [round(float(e), 0) for e in edges]},
        "simulations": sims,
    }


def change_risk(df: pd.DataFrame) -> dict:
    c = df["change_related"]
    return {
        "change_related_share": round(float(c.mean()), 4),
        "change_related_loss_share": round(float(df.loc[c, "financial_loss_usd"].sum() / df["financial_loss_usd"].sum()), 4),
        "sev1_rate_change": round(float((df.loc[c, "severity"] == "SEV-1").mean()), 4),
        "sev1_rate_other": round(float((df.loc[~c, "severity"] == "SEV-1").mean()), 4),
    }


def psi(expected: pd.Series, actual: pd.Series, eps: float = 1e-4) -> float:
    """Population stability index between two categorical distributions."""
    cats = expected.index.union(actual.index)
    e = expected.reindex(cats, fill_value=0) / expected.sum() + eps
    a = actual.reindex(cats, fill_value=0) / actual.sum() + eps
    return float(((a - e) * np.log(a / e)).sum())


def category_shift(df: pd.DataFrame, cutoff: str) -> dict:
    """Detect the shift in incident mix after a cutoff date (vendor regime change)."""
    before = df[df["opened_at"] < cutoff]["category_clean"].value_counts()
    after = df[df["opened_at"] >= cutoff]["category_clean"].value_counts()
    monthly = (df.groupby("month")["category_clean"].apply(lambda s: (s == "Third Party").mean()).round(4))
    tp = df["category_clean"] == "Third Party"
    return {"psi": round(psi(before, after), 4), "third_party_share_by_month": monthly.to_dict(),
            "third_party_share_before": round(float(tp[df["opened_at"] < cutoff].mean()), 4),
            "third_party_share_after": round(float(tp[df["opened_at"] >= cutoff].mean()), 4)}
