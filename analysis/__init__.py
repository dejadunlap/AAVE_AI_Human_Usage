"""
Analysis module for AAVE feature comparison.
Includes statistical testing, visualization, and utility functions.
"""

from .feature_analysis import (
    StatisticalTests,
    wilson_ci,
    newcombe_diff_ci,
    two_prop_ztest,
    _bh_fdr,
    test_density_difference,
    compare_feature_densities,
)
from .visualization import (
    plot_feature_rates,
    plot_density_bars,
    plot_sentiment,
)
from .analysis_utils import (
    _pretty_feat,
    compute_per_thousand_rates,
)

__all__ = [
    "StatisticalTests",
    "wilson_ci",
    "newcombe_diff_ci",
    "two_prop_ztest",
    "_bh_fdr",
    "test_density_difference",
    "compare_feature_densities",
    "plot_grouped_feature_rates",
    "plot_density_bars",
    "plot_density_bars_multi",
    "plot_sentiment_multi",
    "_pretty_feat",
    "compute_per_thousand_rates",
]
