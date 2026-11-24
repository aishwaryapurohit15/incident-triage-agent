"""Draw the README figures from reports/metrics.json.

Usage (from the repository root): python scripts/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
INK, MUTED, LINE, ACCENT, FAINT, RED = "#1c1d1f", "#5d6066", "#d9d9d5", "#39434d", "#a3a6a0", "#8c3b32"
SEV = {"SEV-1": "#8c3b32", "SEV-2": "#b07434", "SEV-3": "#6d7869", "SEV-4": "#a3a6a0"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": LINE, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150,
})


def main() -> None:
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Model comparison on cost per ticket.
    names = {"majority_class": "Majority class", "keyword_rules": "Keyword rules", "tfidf_challenger": "TF-IDF challenger", "tfidf_baseline": "TF-IDF baseline (champion)"}
    order = ["majority_class", "keyword_rules", "tfidf_challenger", "tfidf_baseline"]
    vals = [m["test"]["severity"][k]["cost_per_ticket"] for k in order]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh([names[k] for k in order], vals, color=[FAINT, FAINT, FAINT, ACCENT], height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 0.1, i, f"{v:.2f}", va="center", color=INK)
    ax.set_xlabel("Misclassification cost per ticket (lower is better)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Severity models on the test period", loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png")
    plt.close(fig)

    # 2. Monthly incidents by severity.
    r = m["risk"]
    months = r["months_list"]
    fig, ax = plt.subplots(figsize=(7, 3))
    bottom = [0] * len(months)
    for s, c in SEV.items():
        v = r["monthly_by_severity"][s]
        ax.bar(range(len(months)), v, bottom=bottom, color=c, width=0.75, label=s)
        bottom = [a + b for a, b in zip(bottom, v)]
    ax.set_xticks(range(0, len(months), 3), [months[i] for i in range(0, len(months), 3)])
    ax.legend(ncols=4, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.16))
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Incidents")
    fig.tight_layout()
    fig.savefig(OUT / "monthly_severity.png")
    plt.close(fig)

    # 3. Annual loss distribution.
    ld = r["loss_distribution"]
    edges, counts = ld["histogram"]["edges"], ld["histogram"]["counts"]
    mids = [(a + b) / 2 / 1e6 for a, b in zip(edges[:-1], edges[1:])]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(mids, counts, width=(edges[1] - edges[0]) / 1e6 * 0.92, color=FAINT)
    ax.set_yscale("log")
    for val, col, lab in [(ld["expected_annual_loss"], ACCENT, "Expected"), (ld["var_99_9"], RED, "VaR 99.9%")]:
        ax.axvline(val / 1e6, color=col, ls="--", lw=1.2)
        ax.text(val / 1e6, max(counts) * 0.6, f" {lab} ${val / 1e6:.1f}M", color=col, fontsize=9, fontweight="bold")
    ax.set_xlabel("Annual operational loss (USD millions)")
    ax.set_ylabel("Simulated years (log)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "loss_distribution.png")
    plt.close(fig)

    # 4. Vendor share and the NovaPay shift.
    vs = r["category_shift"]["third_party_share_by_month"]
    keys = sorted(vs)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(range(len(keys)), [vs[k] * 100 for k in keys], color=ACCENT, marker="o", ms=3)
    ax.axvline(keys.index("2026-07"), color=RED, ls="--", lw=1.2)
    ax.text(keys.index("2026-07") - 0.3, max(vs.values()) * 100, "NovaPay live ", color=RED, ha="right", va="top", fontsize=9, fontweight="bold")
    ax.set_xticks(range(0, len(keys), 3), [keys[i] for i in range(0, len(keys), 3)])
    ax.set_ylabel("Third-party share (%)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "vendor_shift.png")
    plt.close(fig)
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
