"""
Contextual significance testing for AAVE linguistic features.
Tests feature rate differences within specific linguistic contexts.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy.stats import fisher_exact, chi2_contingency

from statistical import wilson_ci, newcombe_diff_ci, two_prop_ztest, _bh_fdr


def test_incontext_significance(
    human_rates: Dict[str, Dict[str, float]],
    model_rates_by_name: Dict[str, Dict[str, Dict[str, float]]],
    human_denoms: Dict[str, Dict[str, int]],
    model_denoms_by_name: Dict[str, Dict[str, Dict[str, int]]],
    alpha: float = 0.05,
    use_fdr: bool = True
) -> pd.DataFrame:
    """
    Test contextual significance of AAVE features.
    
    Args:
        human_rates: feature -> context -> rate
        model_rates_by_name: model -> (feature -> context -> rate)
        human_denoms: feature -> context -> n_opportunities
        model_denoms_by_name: model -> (feature -> context -> n_opportunities)
        alpha: significance level
        use_fdr: apply Benjamini-Hochberg FDR correction
        
    Returns:
        DataFrame with statistical test results per model/feature/context
    """
    rows = []
    for model, model_rates in model_rates_by_name.items():
        for feat, h_ctx_map in human_rates.items():
            m_ctx_map = model_rates.get(feat, {})
            # union of contexts seen in either human or model
            contexts = sorted(set(h_ctx_map) | set(m_ctx_map))
            for ctx in contexts:
                r_h = float(h_ctx_map.get(ctx, 0.0))
                r_m = float(m_ctx_map.get(ctx, 0.0))
                n_h = int(human_denoms.get(feat, {}).get(ctx, 0))
                n_m = int(model_denoms_by_name.get(model, {}).get(feat, {}).get(ctx, 0))
                # reconstruct integer "hits"
                x_h = int(round(r_h * n_h))
                x_m = int(round(r_m * n_m))

                # choose Fisher when small/zero expected
                test, stat, p = "two-prop z (≈χ²)", np.nan, np.nan
                # valid table?
                if (n_h>0 and n_m>0) and not (x_h==0 and x_m==0):
                    table = np.array([[x_h, n_h-x_h],
                                      [x_m, n_m-x_m]], dtype=float)
                    # chi-square expected counts
                    chi2, chi_p, dof, exp = chi2_contingency(table, correction=True)
                    if (exp < 5).any():
                        test = "Fisher exact"
                        _, p = fisher_exact(table, alternative="two-sided")
                    else:
                        z, p = two_prop_ztest(x_h, n_h, x_m, n_m, continuity=True)
                        stat = z
                diff = (r_m - r_h) if (n_h>0 and n_m>0) else np.nan
                ci_low, ci_high = (np.nan, np.nan)
                if n_h>0 and n_m>0:
                    ci_low, ci_high = newcombe_diff_ci(x_h, n_h, x_m, n_m, alpha)

                rows.append({
                    "model": model, "feature": feat, "context": ctx,
                    "human_rate": r_h, "human_n": n_h, "human_x": x_h,
                    "model_rate": r_m, "model_n": n_m, "model_x": x_m,
                    "diff": diff, "ci_low": ci_low, "ci_high": ci_high,
                    "test": test, "stat": stat, "p": p
                })

    out = pd.DataFrame(rows)
    # FDR per (feature) across contexts × models
    if use_fdr and not out.empty:
        out["q_bh"] = (
            out.groupby("feature", group_keys=False)["p"]
               .apply(lambda s: pd.Series(_bh_fdr(s.fillna(1.0).to_numpy()), index=s.index))
        )
        out["sig@0.05(BH)"] = out["q_bh"] < 0.05
    return out.sort_values(["feature","model","context"]).reset_index(drop=True)
