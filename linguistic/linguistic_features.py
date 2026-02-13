"""
Linguistic feature detection for AAVE patterns.
Includes detection methods for various phonological and grammatical features.
"""
import re
import nltk
from typing import Optional
from spacy.language import Language
nltk.download('punkt_tab')


class LinguisticFeatureDetector:
    """Detects and analyzes AAVE linguistic features."""
    
    def __init__(self, nlp: Language):
        """Initialize with a spaCy language model."""
        self.nlp = nlp
    
    # ===== Ain't Feature =====
    
    def get_aint_preceding_words(self, dataset: str) -> dict:
        """Count words that immediately precede "ain't" in bigrams."""
        preceding = {}
        words = [w for w in dataset.split() if w]
        for i in range(len(words) - 1):
            if words[i + 1] in {"ain't", "ain't", "aint"}:
                preceding[words[i]] = preceding.get(words[i], 0) + 1
        return preceding
    
    # ===== Negative Concord =====
    
    def has_negative_concord(self, sent: str) -> bool:
        """True if sentence likely exhibits negative concord."""
        doc = self.nlp(sent)
        neg_count = 0
        for tok in doc:
            if tok.dep_ == "neg":
                neg_count += 1
            if tok.text.lower() in {"no", "nothing", "nobody", "never", "none", "nowhere"}:
                neg_count += 1
        return neg_count >= 2
    
    # ===== Habitual 'be' =====
    
    def has_habitual_be(self, sent: str) -> bool:
        """Detect habitual 'be' (exclude future "will/gonna/going to be")."""
        if re.search(r"\b(will|gonna|going to|gon|gwine)\s+be\b", sent.lower()):
            return False
        doc = self.nlp(sent)
        for s in doc.sents:
            toks = list(s)
            for i, tok in enumerate(toks):
                if tok.lemma_ != "be":
                    continue
                prog = next((t for t in toks[i + 1:i + 5] if t.tag_ == "VBG" or t.text.lower().endswith("ing")), None)
                if prog is None:
                    continue
                subj_ok = any(ch.dep_.startswith("nsubj") for ch in tok.children) or \
                          any(ch.dep_.startswith("nsubj") for ch in prog.children)
                if subj_ok:
                    return True
        return False

    def get_habitual_be_subject(self, sent: str) -> Optional[str]:
        """Return subject (lowercased) for the first habitual 'be' in a sentence."""
        if re.search(r"\b(will|gonna|going to|gon|gwine)\s+be\b", sent.lower()):
            return None
        doc = self.nlp(sent)
        for s in doc.sents:
            toks = list(s)
            for i, tok in enumerate(toks):
                if tok.lemma_ != "be":
                    continue
                prog = next((t for t in toks[i + 1:i + 5] if t.tag_ == "VBG" or t.text.lower().endswith("ing")), None)
                if prog is None:
                    continue
                subj_tok = next((ch for ch in tok.children if ch.dep_.startswith("nsubj")), None)
                if subj_tok is None:
                    subj_tok = next((ch for ch in prog.children if ch.dep_.startswith("nsubj")), None)
                if subj_tok is None and tok.head != tok and tok.head.dep_.startswith("nsubj"):
                    subj_tok = tok.head
                if subj_tok is not None and subj_tok.pos_ in {"PRON", "NOUN", "PROPN"}:
                    return subj_tok.text.lower()
        return None
    
    # ===== Double Comparative/Superlative =====
    
    def has_double_comparative(self, sent: str) -> bool:
        """
        True if:
          - 'more/less' + comparative, or
          - 'most/least' + superlative
        """
        _IRREG_COMPS = {"better", "worse", "farther", "further", "less"}
        _IRREG_SUPS = {"best", "worst", "furthest", "farthest", "least"}
        _DEG_SKIP = {"much", "far", "way", "even", "very", "really", "so", "real", "kinda", "sorta", "a", "lot"}

        doc = self.nlp(sent)
        for s in doc.sents:
            toks = list(s)
            n = len(toks)
            for i, tok in enumerate(toks):
                low = tok.text.lower()

                # more/less + (skip adv) + comparative
                if low in {"more", "less"}:
                    k, hops = i + 1, 0
                    while k < n and hops < 2 and (toks[k].pos_ == "ADV" or toks[k].text.lower() in _DEG_SKIP):
                        k += 1; hops += 1
                    if k < n:
                        nxt = toks[k]
                        if (nxt.tag_ in {"JJR", "RBR"}) or \
                           (nxt.pos_ in {"ADJ", "ADV"} and nxt.text.lower().endswith("er")) or \
                           (nxt.lemma_.lower() in _IRREG_COMPS):
                            return True

                # most/least + (skip adv) + superlative
                if low in {"most", "least"}:
                    k, hops = i + 1, 0
                    while k < n and hops < 2 and (toks[k].pos_ == "ADV" or toks[k].text.lower() in _DEG_SKIP):
                        k += 1; hops += 1
                    if k < n:
                        nxt = toks[k]
                        if (nxt.tag_ in {"JJS", "RBS"}) or \
                           (nxt.pos_ in {"ADJ", "ADV"} and nxt.text.lower().endswith("est")) or \
                           (nxt.lemma_.lower() in _IRREG_SUPS):
                            return True
        return False
    
    # ===== Multi-Modal Verbs =====
    
    def _is_modal_token(self, tok) -> bool:
        """Check if token is a modal verb."""
        _MODAL_LEMMAS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would", "ought"}
        _MODAL_TEXTS = {
            "can", "could", "may", "might", "must", "shall", "should", "will", "would",
            "oughta", "gonna", "gon", "finna", "hafta", "woulda", "coulda", "shoulda", "mighta"
        }
        return tok.tag_ == "MD" or tok.lemma_.lower() in _MODAL_LEMMAS or tok.text.lower() in _MODAL_TEXTS

    def _verb_head(self, tok):
        """Get the main verb head, following auxiliary chain."""
        h = tok
        steps = 0
        while steps < 5 and (h.pos_ == "AUX" or h.dep_ in {"aux", "auxpass"}):
            if h.head == h:
                break
            h = h.head
            steps += 1
        return h

    def has_multiple_modals(self, sent: str, max_gap_tokens: int = 5) -> bool:
        """Detect multiple modal verbs in close succession."""
        _SKIP_BETWEEN = {
            "not", "n't", "really", "just", "probably", "maybe", "kinda", "sorta", "still",
            "always", "usually", "often", "ever", "even", "real", "so", "too", "very", "pretty", "right"
        }
        doc = self.nlp(sent)
        for s in doc.sents:
            toks = list(s)
            mods = [t for t in toks if self._is_modal_token(t)]
            if len(mods) < 2:
                continue
            for i in range(len(mods) - 1):
                m1, m2 = mods[i], mods[i + 1]
                between = [t for t in toks if m1.i < t.i < m2.i]
                if any(t.pos_ == "CCONJ" or t.is_punct for t in between):
                    continue
                eff_gap = sum(1 for t in between if t.text.lower() not in _SKIP_BETWEEN and not t.is_space)
                if eff_gap > max_gap_tokens:
                    continue
                if self._verb_head(m1) == self._verb_head(m2):
                    return True
        return False
    
    # ===== Perfective 'done' =====
    
    def has_perfective_done(self, sent: str, max_gap_tokens: int = 3) -> bool:
        """Detect perfective 'done' (e.g., 'I done did it')."""
        _DONE_SKIP = {"already", "just", "really", "right", "kinda", "sorta", "still", "always", "usually",
                      "often", "even", "ever", "so", "very", "pretty", "real"}
        doc = self.nlp(sent)
        for s in doc.sents:
            toks = list(s)
            n = len(toks)
            for i, tok in enumerate(toks):
                if tok.text.lower() != "done":
                    continue

                # Exclusion 1: HAVE + done
                if i > 0 and (toks[i - 1].lemma_.lower() == "have" or toks[i - 1].text.lower() in {"'ve", "'ve"}):
                    continue

                # Exclusion 2: BE + done + ADJ/VBG
                if i > 0 and toks[i - 1].lemma_.lower() == "be":
                    nxt = toks[i + 1] if i + 1 < n else None
                    if nxt is not None and (nxt.tag_ == "VBG" or nxt.pos_ == "ADJ"):
                        continue

                # A) done + been + VBG
                if i + 2 < n:
                    t1, t2 = toks[i + 1], toks[i + 2]
                    if t1.lemma_.lower() == "be" and t1.tag_ in {"VBN", "VBD"} and \
                       (t2.tag_ == "VBG" or t2.text.lower().endswith("ing")):
                        return True

                # B) done + (<= gap advs) + VERB (VBD/VBN/VB)
                k, hops = i + 1, 0
                while k < n and hops < max_gap_tokens and (toks[k].pos_ == "ADV" or toks[k].text.lower() in _DONE_SKIP):
                    k += 1; hops += 1
                if k < n and (toks[k].pos_ == "VERB" or toks[k].tag_ in {"VBD", "VBN", "VB"}):
                    return True
        return False

    def get_perfective_done_subject(self, sent: str, max_gap_tokens: int = 3) -> Optional[str]:
        """Get the subject of a perfective 'done' construction."""
        _DONE_SKIP = {"already", "just", "really", "right", "kinda", "sorta", "still", "always", "usually",
                      "often", "even", "ever", "so", "very", "pretty", "real"}
        doc = self.nlp(sent)
        for s in doc.sents:
            toks = list(s); n = len(toks)
            for i, tok in enumerate(toks):
                if tok.text.lower() != "done":
                    continue
                prev = toks[i - 1] if i > 0 else None
                if prev is not None and (prev.lemma_.lower() == "have" or prev.text.lower() in {"'ve", "'ve"}):
                    continue
                if prev is not None and prev.lemma_.lower() == "be":
                    nxt = toks[i + 1] if i + 1 < n else None
                    if nxt is not None and (nxt.tag_ == "VBG" or nxt.pos_ == "ADJ"):
                        continue

                verb = None
                if i + 2 < n:
                    t1, t2 = toks[i + 1], toks[i + 2]
                    if t1.lemma_.lower() == "be" and t1.tag_ in {"VBN", "VBD"} and \
                       (t2.tag_ == "VBG" or t2.text.lower().endswith("ing")):
                        verb = t2
                if verb is None:
                    k, hops = i + 1, 0
                    while k < n and hops < max_gap_tokens and (toks[k].pos_ == "ADV" or toks[k].text.lower() in _DONE_SKIP):
                        k += 1; hops += 1
                    if k < n and (toks[k].pos_ == "VERB" or toks[k].tag_ in {"VBD", "VBN", "VB"}):
                        verb = toks[k]
                if verb is None:
                    continue

                subj_tok = next((ch for ch in verb.children if ch.dep_.startswith("nsubj")), None)
                if subj_tok is None:
                    subj_tok = next((ch for ch in tok.children if ch.dep_.startswith("nsubj")), None)
                if subj_tok is None:
                    head, steps = verb, 0
                    while steps < 3 and head.head != head:
                        head = head.head; steps += 1
                        cand = next((ch for ch in head.children if ch.dep_.startswith("nsubj")), None)
                        if cand is not None:
                            subj_tok = cand; break

                if subj_tok is not None and subj_tok.pos_ in {"PRON", "NOUN", "PROPN"}:
                    return subj_tok.text.lower()
        return None
    
    # ===== Null Copula =====
    
    def has_null_copula(self, sent: str, max_gap_tokens: int = 2, block_I: bool = True) -> bool:
        """Detect null copula (missing 'be' between subject and predicate)."""
        def is_be_like(t):
            lo = t.text.lower()
            return ((t.lemma_ == "be" and t.pos_ in {"AUX", "VERB"}) or
                    lo in {"'s", "'s", "'re", "'re", "'m", "'m", "ain't", "aint", "ain't"})

        doc = self.nlp(sent)
        for s in doc.sents:
            toks = list(s)
            for head in toks:
                if not (head.pos_ in {"ADJ", "NOUN", "PROPN"} or head.tag_ == "VBG"):
                    continue
                subj = next((ch for ch in head.children if ch.dep_.startswith("nsubj")), None)
                if subj is None:
                    continue
                if block_I and subj.text.lower() == "i":
                    continue
                if head.dep_ in {"obj", "dobj", "pobj", "obl", "nmod"}:
                    continue
                if any(ch.dep_ in {"cop", "aux", "auxpass"} and ch.lemma_ == "be" for ch in head.children):
                    continue
                i0, i1 = sorted([subj.i, head.i])
                between = [t for t in doc if i0 < t.i < i1]
                if any(is_be_like(t) for t in between):
                    continue
                if any(t.pos_ == "VERB" and t.lemma_ != "be" for t in between):
                    continue
                gap = sum(1 for t in between if not t.is_punct)
                if gap <= max_gap_tokens and subj.i < head.i:
                    return True
        return False

    def get_null_copula_subject(self, sent: str, max_gap_tokens: int = 2, block_I: bool = True) -> Optional[str]:
        """Get the subject of a null copula construction."""
        def is_be_like(t):
            lo = t.text.lower()
            return ((t.lemma_ == "be" and t.pos_ in {"AUX", "VERB"}) or
                    lo in {"'s", "'s", "'re", "'re", "'m", "'m", "ain't", "aint", "ain't"})

        doc = self.nlp(sent)
        for s in doc.sents:
            toks = list(s)
            for head in toks:
                if not (head.pos_ in {"ADJ", "NOUN", "PROPN"} or head.tag_ == "VBG"):
                    continue
                subj = next((ch for ch in head.children if ch.dep_.startswith("nsubj")), None)
                if subj is None:
                    continue
                if block_I and subj.text.lower() == "i":
                    continue
                if head.dep_ in {"obj", "dobj", "pobj", "obl", "nmod"}:
                    continue
                if any(ch.dep_ in {"cop", "aux", "auxpass"} and ch.lemma_ == "be" for ch in head.children):
                    continue
                i0, i1 = sorted([subj.i, head.i])
                between = [t for t in doc if i0 < t.i < i1]
                if any(is_be_like(t) for t in between):
                    continue
                if any(t.pos_ == "VERB" and t.lemma_ != "be" for t in between):
                    continue
                gap = sum(1 for t in between if not t.is_punct)
                if subj.i < head.i and gap <= max_gap_tokens:
                    return subj.text.lower()
        return None
