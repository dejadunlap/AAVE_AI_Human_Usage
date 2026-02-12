"""
Statistical testing utilities for comparing feature distributions.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, norm
from math import sqrt
from typing import Tuple, Dict, Mapping, Optional, List


class StatisticalTests:
    """Statistical tests for comparing linguistic features."""
    
    @staticmethod
    def chi_square_significance(human_total: int,
                               model_total: int,
                               human_density: float,
                               model_density: float) -> Tuple[float, float]:
        """
        Calculate Chi-Square test for independence to measure statistical significance
        of differences between feature densities.
        
        Args:
            human_total: Total sentences in human dataset
            model_total: Total sentences in model dataset
            human_density: Feature density in human data
            model_density: Feature density in model data
            
        Returns:
            Tuple of (chi2_statistic, p_value)
        """
        human_count = round(human_density * human_total)
        model_count = round(model_density * model_total)
        
        table = np.array([
            [human_count, human_total - human_count],
            [model_count, model_total - model_count]
        ])
        
        chi2, p, dof, expected = chi2_contingency(table)
        return chi2, p


# ============================================================================
# Statistical Helper Functions
# ============================================================================

def wilson_ci(x: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - alpha/2)
    p = x / n
    z2 = z*z
    denom = 1 + z2/n
    center = (p + z2/(2*n)) / denom
    half = z * sqrt((p*(1-p)/n) + (z2/(4*n*n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def newcombe_diff_ci(x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Newcombe's method for CI on difference of proportions."""
    l1, u1 = wilson_ci(x1, n1, alpha)
    l2, u2 = wilson_ci(x2, n2, alpha)
    return (l2 - u1, u2 - l1)


def two_prop_ztest(x1: int, n1: int, x2: int, n2: int, continuity: bool = True) -> Tuple[float, float]:
    """Two-proportion z-test."""
    p1, p2 = x1/n1, x2/n2
    p = (x1 + x2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1/n1 + 1/n2))
    if se == 0:
        return (0.0, 1.0)
    diff = p2 - p1
    if continuity:
        cc = 0.5 * (1/n1 + 1/n2)
        diff = np.sign(diff) * max(0.0, abs(diff) - cc)
    z = diff / se
    pval = 2 * (1 - norm.cdf(abs(z)))
    return (z, pval)


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q-values (monotone, two-sided)"""
    q = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if mask.sum() < 2:
        return q
    m = mask.sum()
    order = np.argsort(p[mask])
    ranks = np.arange(1, m+1)
    qmask = p[mask][order] * m / ranks
    # make non-increasing from the end
    qmask = np.minimum.accumulate(qmask[::-1])[::-1]
    q[mask] = np.clip(qmask[np.argsort(order)], 0, 1)
    return q


def test_density_difference(h_total: int, m_total: int,
                           h_density: float, m_density: float,
                           alpha: float = 0.05) -> Dict[str, float]:
    """Test density difference between human and model using appropriate statistical test."""
    if h_total <= 0 or m_total <= 0:
        raise ValueError("Totals must be > 0.")

    # Convert densities to counts
    h_hits = int(round(h_density * h_total))
    m_hits = int(round(m_density * m_total))
    h_hits = max(0, min(h_hits, h_total))
    m_hits = max(0, min(m_hits, m_total))

    table = np.array([[h_hits, h_total - h_hits],
                      [m_hits, m_total - m_hits]], dtype=float)

    # Degenerate? (no variation)
    if (table.sum(axis=1) == 0).any() or (table.sum(axis=0) == 0).any():
        return {"test": "degenerate", "p": np.nan, "stat": np.nan,
                "human_rate": h_hits/h_total if h_total else np.nan,
                "model_rate": m_hits/m_total if m_total else np.nan,
                "diff": (m_hits/m_total - h_hits/h_total) if (h_total and m_total) else np.nan,
                "ci_low": np.nan, "ci_high": np.nan,
                "human_hits": int(h_hits), "model_hits": int(m_hits)}

    # Chi-square with continuity correction; Fisher if small expected
    chi2, chi_p, dof, exp = chi2_contingency(table, correction=True)
    use_fisher = (exp < 5).any()
    if use_fisher:
        _, p = fisher_exact(table, alternative="two-sided")
        stat = np.nan
        test = "fisher"
    else:
        z, p = two_prop_ztest(h_hits, h_total, m_hits, m_total, continuity=True)
        stat = z
        test = "two-prop z"

    p1, p2 = h_hits/h_total, m_hits/m_total
    ci_l, ci_u = newcombe_diff_ci(h_hits, h_total, m_hits, m_total, alpha)
    return {"test": test, "stat": float(stat), "p": float(p),
            "human_rate": float(p1), "model_rate": float(p2),
            "diff": float(p2 - p1), "ci_low": float(ci_l), "ci_high": float(ci_u),
            "human_hits": int(h_hits), "model_hits": int(m_hits)}


def compare_feature_densities(
    h_total: int,
    human_density: Mapping[str, float],
    model_densities: Mapping[str, Mapping[str, float]],
    model_totals: Optional[Mapping[str, int]] = None,
    alpha: float = 0.05,
    fdr: bool = True,
    fdr_scope: str = "per_model",
) -> pd.DataFrame:
    """
    Compare Human vs Models.
    Returns a DataFrame with model vs human feature density comparisons.
    """
    if h_total <= 0:
        raise ValueError("h_total must be > 0.")
    model_totals = model_totals or {name: h_total for name in model_densities.keys()}

    rows: List[Dict] = []
    for name, md in model_densities.items():
        m_total = int(model_totals.get(name, 0))
        if m_total <= 0:
            raise ValueError(f"Total for model '{name}' must be > 0.")
        feats = sorted(set(human_density) & set(md))
        if not feats:
            raise ValueError(f"No overlapping feature keys for model '{name}'.")
        for feat in feats:
            out = test_density_difference(h_total, m_total, float(human_density[feat]), float(md[feat]), alpha)
            out.update({"model": name, "feature": feat})
            rows.append(out)

    df = pd.DataFrame(rows)
    # FDR
    if fdr:
        if fdr_scope == "per_model":
            df["q_bh"] = np.nan
            for name, sub_idx in df.groupby("model").groups.items():
                q = _bh_fdr(df.loc[sub_idx, "p"].to_numpy())
                df.loc[sub_idx, "q_bh"] = q
        elif fdr_scope == "global":
            df["q_bh"] = _bh_fdr(df["p"].to_numpy())
        else:
            raise ValueError("fdr_scope must be 'per_model' or 'global'.")
        df["sig@0.05(BH)"] = df["q_bh"] < 0.05
    return df.sort_values(["model", "p"]).reset_index(drop=True)
