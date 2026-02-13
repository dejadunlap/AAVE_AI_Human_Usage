"""
Data loading and text cleaning utilities for interview and tweet data.
"""
import os
import re
import nltk
from typing import List


class DataLoader:
    """Handles reading and cleaning text data from files."""
    
    def __init__(self, path: str, data_type: str, human: bool = True):
        """
        Args:
            path: file or directory to read from
            data_type: "interview" | "tweet"
            human: if True, restrict interview lines to human speaker (se)
        """
        self.path = path
        self.data_type = data_type
        self.human = human
        self.dataset = ""
        self.total_sentences = 0
    
    def load(self) -> None:
        """Read & clean text from files/directories into self.dataset."""
        paths: List[str] = []
        
        if self.data_type == "interview":
            if os.path.isdir(self.path):
                for root, _, files in os.walk(self.path):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        if "checkpoints" in fpath:
                            continue
                        paths.append(fpath)
            else:
                paths = [self.path]

            chunks: List[str] = []
            for p in paths:
                try:
                    chunks.append(self._clean_interview(p))
                except Exception:
                    continue

            self.dataset = " ".join(chunks).strip()
            self.total_sentences = len(nltk.sent_tokenize(self.dataset))
            
        elif self.data_type == "tweet":
            if not self.human:
                if os.path.isdir(self.path):
                    for root, _, files in os.walk(self.path):
                        for fname in files:
                            fpath = os.path.join(root, fname)
                            with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                                for line in fh:
                                    self.dataset += self._clean_tweet(line)
                                    self.total_sentences += 1
            else:
                with open(self.path, "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        self.dataset += self._clean_tweet(line)
                        self.total_sentences += 1

    @staticmethod
    def _clean_tweet(text: str) -> str:
        
        # Pull out just the tweet, not the metadata
        text = text.split("\t")[5]

        """Clean tweet text by removing mentions, URLs, hashtags, emojis, etc."""
        # remove mentions
        text = re.sub(r"@\w+", " ", text)
        # remove URLs
        text = re.sub(r"http\S+|www\S+", " ", text)
        # remove hashtags symbol (but keep the word)
        text = re.sub(r"#", " ", text)
        # remove emojis and other non-alphanumeric chars except apostrophes and period
        text = re.sub(r"[^\w\s'\u2019.]", " ", text, flags=re.UNICODE)
        # collapse multiple spaces
        text = re.sub(r"\s+", " ", text)

        # Add punctuation so that multiple tweets don't get
        # combined into one "sentence"
        text = text.lower().strip()
        if len(text) > 0:
            if text[-1] in "abcdefghijklmnopqrstuvwxyz0123456789":
                text = text + "."
            text = text + " "

        return text.lower().strip()

    def _clean_interview(self, path: str) -> str:
        """
        For interviews: pull only speaker (se) lines (when human=True) and strip non-word chars.
        Keeps straight and curly apostrophes.
        """
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            content: List[str] = []
            if self.data_type == "interview":
                if self.human:
                    for line in fh:
                        cols = line.split("\t")
                        if len(cols) < 4:
                            continue
                        # speaker content, not a pause
                        if "se" in cols[1] and "(pause " not in cols[3]:
                            content.append(cols[3])
                else:
                    for line in fh:
                        content.append(line)
            else:
                for line in fh:
                    content.append(line)

        text = " ".join(content)
        # keep letters, digits, spaces, and both apostrophe types
        cleaned_text = re.sub(r"[^\w\s'\u2019.]", " ", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        return cleaned_text.lower().strip()
