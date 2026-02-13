#!/usr/bin/env python3
"""
Run script for lexical feature detection on AAVE text datasets.

Modify the configuration variables below to change analysis parameters.
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List

from linguistic import AAVEFeatureComparison

# If you want to do one at a time uncomment the following lines
# Path to data file or directory
#DATA_PATH = "./data/tweets/human_tweets.txt"

# Data type: "interview" or "tweet"
#DATA_TYPE = "tweet"

# If True, restrict to human speakers (for interviews only)
#HUMAN_ONLY = True

# Save results to JSON file (set to None to skip)
#OUTPUT_FILE = "human_tweets_results.json"

# Print detailed output
#VERBOSE = False

def validate_data_path(data_path: str) -> bool:
    """Check if data path exists."""
    if not os.path.exists(data_path):
        print(f"ERROR: Data path does not exist: {data_path}", file=sys.stderr)
        return False
    return True


def run_lexical_feature_detection(data_path: str, data_type: str, human: bool, verbose: bool = False, human_keys: List[str] = None):
    """
    Run lexical feature detection on dataset.
    
    Args:
        data_path: Path to data file or directory
        data_type: "interview" or "tweet"
        human: If True, restrict to human speakers (for interviews)
        verbose: If True, print detailed output
        
    Returns:
        Dictionary containing analysis results
    """
    
    try:

        # analyzer
        print(f"Loading into analyzer for data path {data_path}")
        analyzer = AAVEFeatureComparison(
            path=data_path,
            data_type=data_type,
            human=human
        )
        
        # Load data
        print(f"Loading Data at Path {data_path}")
        analyzer.load_data()
        
        # Run lexical feature analysis
        print(f"Running Feature Analysis")
        top_be, top_null, top_done, top_aint = analyzer.lexical_feature(human_keys=human_keys)
        
        # Compile results
        print(f"Compiling Results")
        results = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "data_path": str(data_path),
                "data_type": data_type,
                "human_speakers_only": human,
                "total_sentences": analyzer.total_sentences,
                "dataset_length": len(analyzer.dataset)
            },
            "feature_densities": analyzer.feature_density,
            "feature_probabilities": {
                "be": analyzer.feature_prob["be"],
                "null_copula": analyzer.feature_prob["null_copula"],
                "perfective_done": analyzer.feature_prob["perf_done"],
                "aint": analyzer.feature_prob["aint"]
            },
            "top_preceding_words": {
                "habitual_be": top_be,
                "null_copula": top_null,
                "perfective_done": top_done,
                "aint": top_aint
            }
        }
        
        return results
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_results(results: dict):
    """Pretty-print analysis results."""
    print(f"\n{'='*70}")
    print("LEXICAL FEATURE ANALYSIS RESULTS")
    print(f"{'='*70}\n")
    
    config = results["configuration"]
    print(f"Dataset: {config['data_type']} data")
    print(f"Sentences analyzed: {config['total_sentences']:,}")
    print(f"Total characters: {config['dataset_length']:,}\n")
    
    print("FEATURE DENSITIES (counts per feature detected):")
    print("-" * 70)
    densities = results["feature_densities"]
    for feature, count in sorted(densities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {count}")
    
    print("\n" + "="*70)
    print("TOP PRECEDING WORDS BY FEATURE:")
    print("="*70)
    
    top_words = results["top_preceding_words"]
    for feature_name, words_dict in top_words.items():
        print(f"\n{feature_name.upper()}:")
        print("-" * 70)
        if words_dict:
            sorted_words = sorted(words_dict.items(), key=lambda x: x[1], reverse=True)
            for i, (word, count) in enumerate(sorted_words, 1):
                prob = results["feature_probabilities"].get(word, 0)
                print(f"  {i}. {word} | Count: {count} | P(feature|word): {prob}")
        else:
            print("No occurrences found")
    
    print(f"\n{'='*70}\n")


def save_results(results: dict, output_path: str):
    """Save results to JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"ERROR: Failed to save results to {output_path}: {e}", file=sys.stderr)


def main():
    """Main entry point."""
    # Validate data path
    for type in ["tweet"]:
        human_keys = None
        for data in ["openai"]:
            if data == "human" and type == "tweet":
               DATA_PATH = "./data/tweet/human_tweet/human_tweet.txt"
            else:
                DATA_PATH = f"./data/{type}/{data}_{type}"
            if not validate_data_path(DATA_PATH):
                sys.exit(1)

            
            HUMAN_ONLY = True if data == "human" else False

            VERBOSE = True
    
            # Run analysis
            results = run_lexical_feature_detection(
            data_path=DATA_PATH,
            data_type=type,
            human=HUMAN_ONLY,
            verbose=VERBOSE,
            human_keys=human_keys
            )

            if HUMAN_ONLY:
                human_keys = results["top_preceding_words"]
    
            # Print results
            print_results(results)
    
            # Save results if requested
            OUTPUT_FILE = f"./results/{data}_{type}_results.json"
            save_results(results, OUTPUT_FILE)
    
    print("Lexical feature detection completed successfully!")


if __name__ == "__main__":
    main()
