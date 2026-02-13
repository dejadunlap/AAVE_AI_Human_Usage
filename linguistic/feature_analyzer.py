"""
Main AAVE Feature Comparison class that orchestrates analysis.
Combines data loading, linguistic feature detection, sentiment analysis, and topic modeling.
Optimized for batch processing performance.
"""
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Optional
from spacy.language import Language
from spacy.tokens import Doc
import spacy

from data_handling import DataLoader
from .linguistic_features import LinguisticFeatureDetector
from analysis.feature_analysis import StatisticalTests
from analysis.sentiment_analysis import SentimentAnalyzer


class AAVEFeatureComparison:
    """
    Comprehensive AAVE feature analysis and comparison tool.
    Analyzes linguistic features, sentiment, topics in text data.
    """
    
    def __init__(self, path: str, data_type: str, human: bool = True, nlp: Optional[Language] = None):
        """
        Initialize the AAVE feature comparison analyzer.
        
        Args:
            path: file or directory to read from
            data_type: "interview" | "tweet"
            human: if True, restrict interview lines to human speaker (se)
            nlp: optional preloaded spaCy pipeline (saves reload time across runs)
        """
        
        self.files = path
        self.data_type = data_type
        self.human = human
        
        # lazy-load spaCy if not provided
        self.nlp = nlp if nlp is not None else spacy.load("en_core_web_sm")
        self.nlp.max_length = 5000000
        
        # Data storage
        self._data_loader = DataLoader(path, data_type, human)
        self.dataset: str = ""
        self.total_sentences: int = 0
        
        # Caches
        self._sentences: Optional[List[str]] = None
        self._sentence_docs: Optional[List[Doc]] = None  # Optimized: cache parsed Doc objects
        self._overall_counts: Optional[Counter] = None
        self._word_list: Optional[List[str]] = None  # Cache word tokenization for n_grams
        
        # Feature analyzers
        self.feature_detector = LinguisticFeatureDetector(self.nlp)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.stats = StatisticalTests()
        
        # Feature storage
        self.feature_prob: Dict[str, Dict[str, float]] = {
            "be": {},
            "aint": {},
            "perf_done": {},
            "null_copula": {}
        }
        self.feature_density: Dict[str, float] = {
            "be": 0,
            "negative": 0,
            "aint": 0,
            "double_comp": 0,
            "multi_modals": 0,
            "perf_done": 0,
            "null_copula": 0,
        }
    
    # ========== Data Loading ==========
    
    def load_data(self) -> None:
        """Load and clean data from files."""
        self._data_loader.load()
        self.dataset = self._data_loader.dataset
        self.total_sentences = self._data_loader.total_sentences
        
    def sentences(self) -> List[str]:
        """Get cached sentence tokenization."""
        if self._sentences is None:
            self._sentences = nltk.tokenize.sent_tokenize(self.dataset)
        return self._sentences
    
    def sentence_docs(self) -> List[Doc]:
        """Get cached parsed spaCy Doc objects for sentences. Uses batch processing for performance."""
        if self._sentence_docs is None:
            sents = self.sentences()
            # Use nlp.pipe() for efficient batch processing
            self._sentence_docs = list(self.nlp.pipe(sents, batch_size=32, n_process=1))
        return self._sentence_docs
    
    def word_tokens(self) -> List[str]:
        """Get cached word list for n-gram generation."""
        if self._word_list is None:
            self._word_list = [w for w in self.dataset.split() if w]
        return self._word_list
    
    def n_grams(self, window: int = 2) -> List[List[str]]:
        """Return word n-grams of given window size using cached word tokenization."""
        words = self.word_tokens()
        if window <= 0 or len(words) < window:
            return []
        return [words[i:i + window] for i in range(len(words) - window + 1)]
    
    # ========== Feature-Specific Methods ==========
    # ain't in this file due to simplicity to finding usage
    def aint_feature(self) -> Dict[str, int]:
        """Count words that immediately precede "ain't" in bigrams."""
        preceding: Dict[str, int] = {}
        for left, right in self.n_grams(2):
            if right in {"ain't", "ain't", "aint"}:
                preceding[left] = preceding.get(left, 0) + 1
        return preceding
    
    def find_feature_appearances(self) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
        """
        Iterate sentences and collect subject counts for:
          - habitual 'be', null copula, perfective 'done'
        Also increments densities for negative concord, double comp, multi-modals.
        
        OPTIMIZED: Uses pre-parsed spaCy Doc objects and batch processing.
        
        Returns:
            (be_subj_counts, null_subj_counts, done_subj_counts)
        """
        # Storing the preceding subjects for contextual analysis
        null_preceding: Dict[str, int] = defaultdict(int)
        done_preceding: Dict[str, int] = defaultdict(int)
        be_preceding: Dict[str, int] = defaultdict(int)
        aint_preceding: Dict[str, int] = defaultdict(int)
        
        # Get pre-parsed Doc objects (uses batch processing internally)
        sentence_docs = self.sentence_docs()
        
        # Process each sentence with its pre-parsed Doc
        for doc in sentence_docs:
            # Check all features using pre-parsed Doc (no additional parsing)
            if self.feature_detector.has_null_copula(doc):
                subj = self.feature_detector.get_null_copula_subject(doc)
                self.feature_density["null_copula"] += 1
                if subj:
                    null_preceding[subj] += 1
            
            if self.feature_detector.has_double_comparative(doc):
                self.feature_density["double_comp"] += 1
            
            if self.feature_detector.has_multiple_modals(doc):
                self.feature_density["multi_modals"] += 1
            
            if self.feature_detector.has_perfective_done(doc):
                self.feature_density["perf_done"] += 1
                subj = self.feature_detector.get_perfective_done_subject(doc)
                if subj:
                    done_preceding[subj] += 1
            
            if self.feature_detector.has_negative_concord(doc):
                self.feature_density["negative"] += 1
            
            if self.feature_detector.has_habitual_be(doc):
                self.feature_density["be"] += 1
                subj = self.feature_detector.get_habitual_be_subject(doc)
                if subj:
                    be_preceding[subj] += 1
        
        return dict(be_preceding), dict(null_preceding), dict(done_preceding)
    
    # ========== Density & Probability Calculations ==========
    
    _WORD_RE = re.compile(r"[A-Za-z]+'\w+|[A-Za-z]+|\d+")
    
    def _token_counts(self) -> Counter:
        """Cached token counts over the dataset (lowercased)."""
        if self._overall_counts is None:
            toks = [t.lower() for t in self._WORD_RE.findall(self.dataset)]
            self._overall_counts = Counter(toks)
        return self._overall_counts
    
    def _count_aint_tokens(self) -> int:
        """Count occurrences of ain't/ain't/aint as standalone tokens."""
        return len(re.findall(r"\b(?:ain't|ain't|aint)\b", self.dataset))

        # This should be a count of sentences containing "ain't", not
        # a count of how many times "ain't" occurs overall
        count_aint = 0
        for sentence in self.sentences():
            words = sentence.split()
            if "ain't" in words or "aint" in words:
                count_aint += 1                
        return count_aint

    
    def feature_densities(self) -> None:
        """Normalize feature counts by sentence count and compute ain't rate."""
        n = max(1, self.total_sentences)
        self.feature_density["aint"] = self._count_aint_tokens() / n
        for k in ("negative", "be", "double_comp", "multi_modals", "perf_done", "null_copula"):
            self.feature_density[k] = self.feature_density[k] / n
    
    def _top_k(self, d: Dict[str, int], k: int, allowed: Optional[Dict[str, int]] = None) -> Dict[str, int]:
        """
        Get top-k items from dictionary.
        
        Args:
            d: Dictionary to extract top-k from
            k: Number of top items to return
            allowed: (Deprecated) Human keys for comparison - NOT used for filtering
                    Returns top-k from model regardless of human data presence
        
        Returns:
            Top k items sorted by frequency (descending)
        """
        # Always return top-k from the provided dictionary, regardless of human keys
        # The comparison between human and model data happens at the analysis level
        return dict(sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k])
    
    def _context_prob(self, key: str, numerator_counts: Dict[str, int]) -> float:
        """Compute P(feature|key) ≈ count(feature with key)/count(key)."""
        denom = self._token_counts().get(key.lower(), 0)
        num = numerator_counts.get(key.lower(), 0)
        return (num / denom) if denom else 0.0
    
    def lexical_feature(self, human_keys: Optional[Dict[str, int]] = None) -> Tuple[Dict, Dict, Dict, Dict]:
        """
          Finds top preceding subjects/words for each feature, context probabilities P(feature|key), normalized feature densities
        
        Args:
            human_keys: Optional dict of human feature keys for comparison with model outputs
            
        Returns:
            Tuple of (be_top, null_top, done_top, aint_top) dicts
        """
        be_pre, null_pre, done_pre = self.find_feature_appearances()
        self.feature_densities()
        aint_pre = self.aint_feature()
        
        # normalize key case
        be_pre = {k.lower(): v for k, v in be_pre.items()}
        null_pre = {k.lower(): v for k, v in null_pre.items()}
        done_pre = {k.lower(): v for k, v in done_pre.items()}
        aint_pre = {k.lower(): v for k, v in aint_pre.items()}
        
        # select top-k per feature (k=10 default)
        top_be = self._top_k(be_pre, 10, human_keys)
        top_null = self._top_k(null_pre, 10, human_keys)
        top_done = self._top_k(done_pre, 10, human_keys)
        top_aint = self._top_k(aint_pre, 10, human_keys)
        
        # fill conditional maps
        self.feature_prob["be"] = {k: self._context_prob(key=k, numerator_counts=be_pre) for k in top_be}
        self.feature_prob["null_copula"] = {k: self._context_prob(key=k, numerator_counts=null_pre) for k in top_null}
        self.feature_prob["perf_done"] = {k: self._context_prob(key=k, numerator_counts=done_pre) for k in top_done}
        self.feature_prob["aint"] = {k: self._context_prob(key=k, numerator_counts=aint_pre) for k in top_aint}
        
        print(self.feature_prob)
        print(self.feature_density)
        
        return top_be, top_null, top_done, top_aint
