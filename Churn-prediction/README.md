# Customer Churn Prediction

A complete machine learning workflow for predicting which telecom customers are likely to cancel, comparing a linear and a tree-based model, and choosing a decision threshold that matches a retention budget.

**Status:** Pipeline complete. Results below are from a synthetic stand-in dataset and must be regenerated on the real Telco file before this repo is treated as final.

## 📑 Table of Contents
- [About This Project](#about-this-project)
- [Problem Statement](#problem-statement)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Next Steps](#next-steps)
- [Technical Specifications](#technical-specifications)
- [How to Reproduce](#how-to-reproduce)
- [Author](#author)

## About This Project

This project builds a churn classifier on subscription data and evaluates it the way a retention team would use it: by how many likely churners it catches, and how many offers it wastes. It demonstrates:

- A leakage-safe `scikit-learn` pipeline in which every preprocessing step is fit inside each cross-validation fold
- Model selection on cross-validated metrics, with the test set used once
- Threshold selection tied to a business constraint (a minimum recall) rather than the default 0.5 cutoff
- Honest reporting of results, including where the test set fell short of the target

## Problem Statement

Losing a customer costs more than the revenue from one month, and retention offers cost money. The practical question is not "will this customer churn?" but "which customers should we spend an offer on?" This project addresses:

1. Which customer attributes (contract type, tenure, payment method, service mix) are most associated with churn?
2. Can a model rank customers by churn risk well enough to be useful?
3. At what decision threshold does the model catch most churners while keeping false alarms manageable?

## Data Sources

| Source | Description | Location |
|---|---|---|
| Telco Customer Churn | 7,000+ customer records with service, contract, billing, and churn labels | Kaggle (IBM sample dataset) |
| Synthetic stand-in | Same column layout, generated with a seeded random process for offline runs | `src/make_synthetic_data.py` |

## Project Structure

```
churn-prediction/
├── README.md                    # project overview, results, and instructions
├── requirements.txt             # pinned library dependencies
├── data/
│   └── churn.csv                # input dataset (not committed; see Data Sources)
├── src/
│   ├── make_synthetic_data.py   # generates a stand-in dataset for offline runs
│   └── train.py                 # preprocessing, model comparison, threshold, evaluation
└── results/
    ├── cv_comparison.json       # cross-validated scores for each candidate model
    └── test_metrics.json        # final held-out metrics and chosen threshold
```

## Methodology

### 1. Data Preparation
- `TotalCharges` is stored as text in the source file. It is converted to numeric, and blank values (customers with zero tenure) are imputed with the median.
- The target is `Churn`, encoded as 1 for Yes and 0 for No.
- The data is split 80/20 with stratification on the target, using a fixed random seed.

### 2. Feature Handling
| Feature type | Columns | Treatment |
|---|---|---|
| Numeric | tenure, MonthlyCharges, TotalCharges, SeniorCitizen | Median imputation, standard scaling |
| Categorical | gender, Partner, Dependents, InternetService, Contract, PaymentMethod | One-hot encoding, unknown categories ignored at prediction time |

All transformations are inside a `Pipeline`, so they are fit only on each training fold.

### 3. Model Comparison
Two candidates are evaluated with 5-fold stratified cross-validation on the training set:

- Logistic regression with class weighting
- Histogram gradient boosting with class weighting

The model with the higher mean CV ROC-AUC is selected. The test set is not used for this choice.

### 4. Threshold Selection
Out-of-fold predictions on the training set are used to find the highest decision threshold that still catches at least 70% of churners. This reflects a rule that a missed churner costs more than a wasted offer. The chosen threshold is then applied once to the test set.

## Key Findings

### Model Comparison (cross-validation, training set)

| Model | CV ROC-AUC | CV PR-AUC |
|---|---|---|
| Logistic regression | 0.790 ± 0.004 | 0.586 |
| Histogram gradient boosting | 0.760 ± 0.009 | 0.531 |

### Held-Out Test Performance (logistic regression, threshold 0.51)

| Metric | Value | Meaning |
|---|---|---|
| ROC-AUC | 0.772 | Ability to rank churners above non-churners |
| PR-AUC | 0.565 | Ranking quality on the churn class specifically |
| Precision | 0.465 | About 47% of flagged customers churned |
| Recall | 0.659 | About 66% of churners were flagged |

### Interpretation
- Logistic regression outperformed gradient boosting. On a small, mostly additive dataset this is plausible, and the ranking should be re-checked on the real data.
- Test recall (0.66) fell short of the 0.70 target. The threshold was chosen on training folds, and the test distribution differs slightly. Thresholds chosen on one sample do not always transfer cleanly.
- Roughly half of flagged customers would churn. Whether that is worth an offer depends on offer cost and customer lifetime value, which this project does not model.

## Limitations

- **Synthetic results:** the numbers above come from generated data. They will change when the real file is used.
- **No time-based validation:** the Telco file is a single snapshot, so the model is not tested on future months.
- **Threshold logic:** a recall floor is a simple rule, not a full cost-benefit optimization.
- **Ecological caution:** feature importance from a linear model shows association, not the effect of changing a customer's contract.

## Technical Specifications

### Pipeline Architecture
```
data/churn.csv
   ↓
train.py: load() → numeric coercion, target encoding
   ↓
train_test_split (80/20, stratified)
   ↓
5-fold CV on training set → candidate models in Pipeline(preprocessor, model)
   ↓
select model by CV ROC-AUC → threshold from out-of-fold predictions (recall ≥ 0.70)
   ↓
single evaluation on held-out test set → results/*.json
```

### Environment
- Python 3.10 or later
- pandas, NumPy, scikit-learn, matplotlib (see `requirements.txt`)

## How to Reproduce

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step 1: Get the Data
Download the Telco Customer Churn CSV from Kaggle and save it as `data/churn.csv`.

For an offline run without the real file:
```bash
python src/make_synthetic_data.py
```

### Step 2: Train and Evaluate
```bash
python src/train.py
```

This prints the cross-validation comparison and the test metrics, and writes both to `results/`.

## Author
GitHub: @karen-2408
