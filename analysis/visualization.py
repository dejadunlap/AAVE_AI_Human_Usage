"""
Visualization utilities for AAVE feature analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Any, Dict

def plot_feature_rates(
    rates_df: pd.DataFrame,
    figsize=(11, 4.8),
    value_col: str = "rate_per_1k",
    annotate: bool = True,
    title: str = "Feature rates per 1,000 sentences",
    ylabel: str = "Rate per 1,000",
    bar_width: float = 0.2,
):
    """Draw a bar chart along feature and corpus dimensions."""
    features = rates_df["feature"].cat.categories.tolist()
    corpora = rates_df["corpus"].cat.categories.tolist()
    x = np.arange(len(features))

    fig, ax = plt.subplots(figsize=figsize)

    for i, corp in enumerate(corpora):
        vals = rates_df[rates_df["corpus"] == corp].set_index("feature")[value_col].reindex(features).values
        ax.bar(x + (i - (len(corpora)-1)/2)*bar_width, vals, width=bar_width, label=str(corp).title())

        if annotate:
            for xi, v in zip(x + (i - (len(corpora)-1)/2)*bar_width, vals):
                ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=len(corpora), frameon=False)
    ax.margins(x=0.02)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_density_bars(
    df_multi: pd.DataFrame,
    models_order=("meta", "openai", "google"),
    display_names=None,
    rate_scale: float = 1000.0,
    decimals: int = 1,
    title: str = "Feature rates per 1,000 sentences",
    bar_width: float = 0.18,
    yscale: str = "linear",
):
    """bar plot: Human + each model per feature."""
    display_names = display_names or {m: m.title() for m in models_order}
    present = set(df_multi["model"].unique())
    missing = [m for m in models_order if m not in present]
    if missing:
        raise ValueError(f"Models not found in df: {missing}. Present: {sorted(present)}")

    feats = sorted(df_multi["feature"].unique().tolist())

    # human once per feature
    human_rates = df_multi.drop_duplicates("feature").set_index("feature")["human_rate"]
    x = np.arange(len(feats))

    fig, ax = plt.subplots(figsize=(11, 5.2))

    # Human bars
    human_vals = (human_rates.reindex(feats).fillna(0).to_numpy()) * rate_scale
    base_off = -(len(models_order)+1)/2 * bar_width
    ax.bar(x + base_off, human_vals, width=bar_width, label="Human")

    for xi, v in zip(x + base_off, human_vals):
        ax.text(xi, max(v, 1e-8), f"{v:.{decimals}f}", ha="center", va="bottom", fontsize=8)

    # Model bars
    for i, name in enumerate(models_order):
        sub = df_multi[df_multi["model"] == name].set_index("feature")
        vals = (sub.reindex(feats)["model_rate"].fillna(0).to_numpy()) * rate_scale
        off = -(len(models_order)-1)/2*bar_width + i*bar_width
        ax.bar(x + off, vals, width=bar_width, label=display_names.get(name, name))
        for xi, v, feat in zip(x + off, vals, feats):
            ax.text(xi, max(v, 1e-8), f"{v:.{decimals}f}", ha="center", va="bottom", fontsize=8)
            row = sub.loc[[feat]]
            if "q_bh" in row.columns:
                q = float(row["q_bh"].values[0])
                if np.isfinite(q) and q < 0.05:
                    ax.text(xi, max(v, 1e-8), "*", ha="center", va="bottom", fontsize=13)

    ax.set_xticks(x)
    ax.set_xticklabels(feats, rotation=30, ha="right")
    ax.set_ylabel(f"Rate per {int(rate_scale):,}")
    ax.set_title(title)
    ax.legend(ncol=len(models_order)+1, frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    if yscale == "log":
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-2)
    plt.tight_layout()
    plt.show()


def plot_sentiment(
    results: Dict[str, Dict[str, Any]],
    corpora_order: Sequence[str] = ("human", "llama", "GPT-4o", "Gemma"),
    show_counts: bool = False,
    title: str = "Sentiment by corpus",
):
    """Plot sentiment distribution across corpora."""
    cats = ["negative", "neutral", "positive"]
    vals = []
    labels = []
    for c in corpora_order:
        if c not in results:
            continue
        r = results[c]
        if (not show_counts) and ("proportions" in r):
            v = [r["proportions"].get(k, 0.0) * 100 for k in cats]
            ylab = "Percentage of sentences"
        else:
            cnts = r.get("counts", {k: 0 for k in cats})
            v = [cnts.get(k, 0) for k in cats]
            ylab = "Sentence count"
        vals.append(v)
        labels.append(c.title())
    vals = np.array(vals)

    x = np.arange(len(cats))
    w = 0.8 / max(1, len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, name in enumerate(labels):
        ax.bar(x + (i - (len(labels)-1)/2)*w, vals[i], width=w, label=name)
        for xi, v in zip(x + (i - (len(labels)-1)/2)*w, vals[i]):
            ax.text(xi, v, f"{v:.0f}" if show_counts else f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in cats])
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(ncol=len(labels), frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.margins(x=0.02)
    plt.tight_layout()
    plt.show()
