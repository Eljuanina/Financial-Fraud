# Fraud Detection in Financial Filings

## Overview

This is a university project that explores linguistic and textual features as indicators of fraud in financial filings. Using SEC filings labeled as fraudulent or non-fraudulent, it investigates whether handcrafted linguistic cues, TF-IDF representations, sentiment analysis, and emotion classification can distinguish between the two categories.

The analysis is split across three notebooks, each covering a different aspect of the pipeline.

---

## Repository Structure

```
.
├── fraud_project.ipynb       # Linguistic feature extraction and classification
├── finbert_fraud.ipynb       # Sentiment analysis with FinBERT
├── emotion_fraud.ipynb       # Emotion classification with BERT-Emotions-Classifier
└── Final_Dataset.csv         # Input dataset (170 SEC filings)
```

---

## Dataset

The dataset is the [Financial Statement Fraud Data](https://www.kaggle.com/datasets/amitkedia/financial-statement-fraud-data) by Amit Kedia, available on Kaggle. It contains 170 SEC filings — 85 fraudulent and 85 non-fraudulent — with text extracted from the MD&A sections and financial statements.

The `Fraud` column is binarized (`yes` --> `1`, `no` --> `0`) and recurring boilerplate structures (e.g. `item 14`) are removed via regex before analysis.

> **Note:** `Final_Dataset.csv` is not included due to GitHub's file size limit.  
> Download it from [Kaggle](https://www.kaggle.com/datasets/amitkedia/financial-statement-fraud-data) and place it in the project root.

---

## Notebooks

### `fraud_project.ipynb` — Linguistic Features & Classification

Extracts nine handcrafted linguistic features and trains classifiers on each individually, then on all features combined.

**Features extracted:**

| Feature | Description |
|---|---|
| `length` | Word count per filing |
| `lexical_diversity` | Type-Token Ratio (TTR) |
| `modal_verb_ratio` | Modal verbs relative to total word count |
| `modal_vs_verb_ratio` | Modal verbs relative to all verbs (spaCy POS tagging) |
| `expressivity_emotiveness` | Adjectives + adverbs relative to nouns + verbs |
| `passive_voice_ratio` | Passive constructions (`nsubjpass`) relative to word count |
| `vagueness_ratio` | Vague terms (e.g. *some*, *approximately*, *uncertain*) relative to word count |
| `quantifier_ratio` | Intensifier terms (e.g. *very*, *highly*, *completely*) relative to word count |
| `personal_impersonal_ratio` | Personal pronouns relative to impersonal references |

**Classifiers used:**

- Logistic Regression (LR)
- Gradient Boosting Classifier (GBC)
- Support Vector Machine (SVM)
- Voting Classifier (soft voting)
- Stacking Classifier (LR meta-classifier)

All classifiers use `random_state=39`. A TF-IDF experiment is also included for comparison, run with two different random states (39 and 8).

---

### `finbert_fraud.ipynb` — Sentiment Analysis

Applies [FinBERT](https://huggingface.co/ProsusAI/finbert) to a subset of 30 filings (15 fraudulent, 15 non-fraudulent). Each filing is chunked into 500-word segments; sentiment scores (positive, negative, neutral) are averaged per filing.

---

### `emotion_fraud.ipynb` — Emotion Classification

Applies the [BERT-Emotions-Classifier](https://huggingface.co/ayoubkirouane/BERT-Emotions-Classifier) to a subset of filings, chunked into 500-word segments. Emotion scores across 11 categories are averaged per filing and compared between fraudulent and non-fraudulent groups.

**Emotion categories:** anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust

Results are saved to `emotion_results.csv`.

---

## Results

### Linguistic Features — Classifier Accuracy (%)

| Feature | LR | GBC | SVM | Voting | Stacking | Avg |
|---|---|---|---|---|---|---|
| Text Length | 73.53 | 67.65 | 70.59 | 70.59 | 70.59 | 70.59 |
| Lexical Diversity | 64.71 | 73.53 | 70.59 | 73.53 | 70.59 | 70.59 |
| Modal Verb Ratio | 50.00 | 52.94 | 70.59 | 55.88 | 61.76 | 58.23 |
| Modal Verbs vs Verbs | 47.06 | 44.12 | 70.59 | 41.18 | 55.88 | 51.57 |
| Expressivity Emotiveness | 76.47 | 61.76 | 73.53 | 70.59 | 70.59 | 70.59 |
| Passive Voice | 50.00 | 70.59 | 61.76 | 70.59 | 70.59 | 64.71 |
| Vagueness | 52.94 | 58.82 | 64.71 | 61.76 | 64.71 | 60.59 |
| Modifier Quantity | 50.00 | 55.88 | 67.65 | 52.94 | 58.82 | 57.06 |
| Personal vs. Impersonal | 58.82 | 55.88 | 76.47 | 58.82 | 67.65 | 63.53 |
| **All Features** | 73.53 | 70.59 | 73.53 | 73.53 | 73.53 | 72.94 |
| TF-IDF (seed 39) | 64.71 | 100.00 | 76.47 | 82.35 | 100.00 | 84.71 |
| TF-IDF (seed 8) | 73.53 | 100.00 | 91.18 | 91.18 | 100.00 | 91.18 |

Lexical diversity and expressivity emotiveness were the strongest individual linguistic predictors. Sentiment and emotion analysis did not yield clear distinctions between fraudulent and non-fraudulent filings.

---

## Notes

- The BERT-based experiments (FinBERT, emotion classifier) were run on a subset of 30 filings due to computational constraints.
- The 100% TF-IDF accuracy on GBC and stacking likely reflects overfitting to the specific train/test split rather than genuine generalization.
- The dataset is balanced (85/85), which differs from real-world fraud distributions where fraud is rare.

