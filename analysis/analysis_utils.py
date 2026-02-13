"""
Analysis utility functions for feature comparison.
"""
import pandas as pd
from typing import List, Sequence


def _pretty_feat(name: str) -> str:
    """Turn column names like 'feat_multi_modals' into 'multi modals'."""
    return name.replace("feat_", "").replace("_", " ").replace("aint", "ain't")

def compute_per_thousand_rates(
    df: pd.DataFrame,
    feature_cols: List[str],
    corpus_col: str = "corpus",
    corpora_order: Sequence[str] = ("human", "meta", "openai", "google"),
) -> pd.DataFrame:
    """
    Returns a long DataFrame with per-thousand rates for each feature x corpus.
    Columns: feature, corpus, hits, n, rate_per_1k
    """
    rows = []
    # Ensure boolean
    for c in feature_cols:
        df[c] = df[c].astype(bool)

    for feat in feature_cols:
        grp = df.groupby(corpus_col)[feat].agg(hits="sum", n="count")
        for corp in corpora_order:
            if corp not in grp.index:
                rows.append({"feature": _pretty_feat(feat), "corpus": corp,
                             "hits": 0, "n": 0, "rate_per_1k": 0.0})
            else:
                h, n = int(grp.loc[corp, "hits"]), int(grp.loc[corp, "n"])
                rate = (h / n * 1000.0) if n else 0.0
                rows.append({"feature": _pretty_feat(feat), "corpus": corp,
                             "hits": h, "n": n, "rate_per_1k": rate})
    out = pd.DataFrame(rows)
    # keep feature order as given
    out["feature"] = pd.Categorical(out["feature"], [_pretty_feat(f) for f in feature_cols], ordered=True)
    out["corpus"] = pd.Categorical(out["corpus"], corpora_order, ordered=True)
    return out.sort_values(["feature", "corpus"]).reset_index(drop=True)
