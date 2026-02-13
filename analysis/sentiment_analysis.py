"""
Sentiment analysis and embedding-based text analysis.
"""
import math
import numpy as np
import nltk
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import mannwhitneyu, ks_2samp
from spacy.language import Language


class SentimentAnalyzer:
    """Performs sentiment analysis on text."""
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        try:
            _ = SentimentIntensityAnalyzer()
        except Exception:
            nltk.download("vader_lexicon")
        self._vader = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self,
                         sentences: List[str],
                         plot_path: Optional[str] = None,
                         show_plot: bool = False) -> Dict[str, Any]:
        """
        Analyze sentiment of sentences using VADER.
        
        Args:
            sentences: List of sentences to analyze
            plot_path: Path to save sentiment distribution plot
            return_scores: If True, include individual scores in result
            show_plot: If True, display the plot
            
        Returns:
            Dictionary with sentiment statistics and distributions
        """
        if not sentences:
            return {
                'n_sentences': 0,
                'mean': math.nan,
                'median': math.nan,
                'stdev': math.nan,
                'min': math.nan,
                'max': math.nan,
                'counts': {'negative': 0, 'neutral': 0, 'positive': 0},
                'proportions': {'negative': 0.0, 'neutral': 0.0, 'positive': 0.0},
                'top_positive_examples': [],
                'top_negative_examples': []            
                }
        
        scores, labels = [], []
        for sent in sentences:
            v = self._vader.polarity_scores(sent)
            c = v["compound"]
            scores.append(c)
            if c >= 0.05:
                labels.append("positive")
            elif c <= -0.05:
                labels.append("negative")
            else:
                labels.append("neutral")
        
        n = len(scores)
        mean = sum(scores) / n
        median = sorted(scores)[n // 2]
        stdev = (sum((x - mean) ** 2 for x in scores) / n) ** 0.5 if n > 1 else 0.0
        mn, mx = min(scores), max(scores)
        
        counts = {
            'negative': labels.count("negative"),
            'neutral': labels.count("neutral"),
            'positive': labels.count("positive")
        }
        proportions = {k: (v / n) for k, v in counts.items()}
        
        top_pos_idx = sorted(range(n), key=lambda i: scores[i], reverse=True)[:3]
        top_neg_idx = sorted(range(n), key=lambda i: scores[i])[:3]
        top_positive_examples = [sentences[i] for i in top_pos_idx]
        top_negative_examples = [sentences[i] for i in top_neg_idx]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(6, 4))
        cats = ["negative", "neutral", "positive"]
        vals = [counts[c] for c in cats]
        ax.bar(cats, vals)
        ax.set_title("VADER sentiment distribution (by sentence)")
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v}\n({proportions[cats[i]]:.0%})", ha="center", va="bottom")
        plt.tight_layout()
        if plot_path:
            fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close(fig)
        
        result = {
            'n_sentences': n,
            'mean': mean,
            'median': median,
            'stdev': stdev,
            'min': mn,
            'max': mx,
            'counts': counts,
            'proportions': proportions,
        }
        
        result['scores'] = scores
        
        return result