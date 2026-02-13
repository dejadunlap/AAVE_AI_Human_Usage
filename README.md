# AAVE Human–Model Comparison

This project is associated with the paper `(2026) D. Dunlap, R.T. McCoy "Evaluating Large Language Model Usage of African American Vernacular English". HCII Cross-Culural Design`.

This is the code we used for comparing model-human usage of African American Vernacular English (AAVE), with feature densities and contextual densities included.
---

## What this project does
- Loads and cleans interview or tweet text data
- Detects AAVE linguistic features
- Computes feature densities and context-conditioned probabilities
- Runs statistical tests comparing human vs. model outputs
- Visualizatoin tools

## Requirements & recommended Python version
run ```pip install -r requirement.txt``` to download the packages for this project.
If running the feature analyzer you should also run ```python -m spacy download en_core_web_sm``` for the tokenizing elements

## Quick start — example usage
The `run.py` file has the necessary set up to complete the feature analysis on the existing dataset for this project or your own dataset (with some tinkering to the dataloader). 

You could also experiements with the below code.

1) Basic end-to-end analysis (interview data):

```python
from linguistic.feature_analyzer import AAVEFeatureComparison

# path -> file or folder containing interview transcripts
an = AAVEFeatureComparison(path='data/interviews/', data_type='interview', human=True)
an.load_data()
print('sentences:', len(an.sentences()))

# compute densities + find lexical contexts
an.feature_densities()
print('feature densities:', an.feature_density)
lex = an.lexical_feature()
print('top preceding words for features:', lex)

# sentiment (per-sentence VADER)
sent = an.sentiment_analyzer.analyze_sentiment(an.sentences(), show_plot=True)
print(sent)
```

2) Compare human vs model feature densities (use `compare_feature_densities`):

```python
from analysis.feature_analysis import compare_feature_densities

human_total = 5000
human_density = {'be': 0.012, 'ain't': 0.001, 'perf_done': 0.003}
model_densities = {
    'gpt': {'be': 0.008, 'ain't': 0.0002, 'perf_done': 0.002},
    'llama': {'be': 0.013, 'ain't': 0.0001, 'perf_done': 0.004}
}

df = compare_feature_densities(human_total, human_density, model_densities)
print(df)
```

3) Plotting helper example (use `analysis.visualization` functions):

```python
from analysis.visualization import plot_feature_rates
import pandas as pd

# small example DataFrame expected by plot functions
df = pd.DataFrame([
    {'feature':'be', 'corpus':'human', 'rate_per_1k': 12.0},
    {'feature':'be', 'corpus':'gpt', 'rate_per_1k': 8.0},
    {'feature":"ain't", 'corpus':'human', 'rate_per_1k': 1.0},
    {'feature":"ain't", 'corpus':'gpt', 'rate_per_1k': 0.2},
])
plot_feature_rates(df)
```

---

## Project layout
- `data_handling/` — text loading & cleaning (DataLoader)
- `linguistic/` — feature detection & analyzer (`AAVEFeatureComparison`, `LinguisticFeatureDetector`)
- `analysis/` — statistical tests, contextual analysis, sentiment, and plotting helpers

Data available on request. Once downloaded the dataset and place into a `data/interview` and `data/tweet` folders.

---

## Notes & helpful tips
- spaCy is loaded lazily by `AAVEFeatureComparison` if not provided; downloading `en_core_web_sm` once speeds repeated runs.
- NLTK `punkt` tokenizer is required for sentence tokenization.
- `SentimentIntensityAnalyzer` (VADER) requires `vader_lexicon` (the code downloads it automatically if missing).
- There is no top-level CLI; use the classes and functions from the modules in scripts or notebooks.
