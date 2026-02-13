"""
Linguistic module for AAVE feature detection.
Contains linguistic feature analyzer and feature detection utilities.
"""

from .linguistic_features import LinguisticFeatureDetector
from .feature_analyzer import AAVEFeatureComparison

__all__ = [
    "LinguisticFeatureDetector",
    "AAVEFeatureComparison",
]
