# Credit Card Fraud Detection — Kaggle Playground S6E11

Gradient-boosting ensemble for the Kaggle competition
[Predicting Credit Card Fraud (Playground Series S6E11)](https://www.kaggle.com/competitions/playground-series-s6e11).
The task is to predict the probability that a card transaction is fraudulent. Submissions are scored on **ROC AUC**.

## Results

| Model | Features | 5-fold CV AUC |
|---|---|---|
| LightGBM | raw | 0.94499 |
| CatBoost | raw | 0.94548 |
| XGBoost | raw | 0.94659 |
| XGBoost | raw + original-data lookups | 0.94674 |
| **XGBoost (xgbT)** | raw + original-data lookups + in-fold target encoding | **0.94740** |
| **Rank blend (all four)** | — | see `fraud_blend.py` output |

Public leaderboard: 0.94843

## Data

- **Competition data:** 699,960 training rows and 299,984 test rows. There are 24 features: transaction amount, velocity, merchant risk, device/channel, red-flag indicators and more. Only **1.67%** of rows are fraud.
- **Original dataset:** the competition data was synthetically generated from [Credit Card Fraud Detection 2026](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026), which this project uses as an extra source of features.

The strongest single features (by single-feature AUC) were `merchant_risk_score` (~0.75), `velocity_score` (~0.71), `cvv_retry_count` (~0.71) and `is_new_merchant` (~0.64).

## How it works

```
train.csv ──┬─► raw features ─────────────────────────────► LightGBM ─┐
            ├─► raw features ─────────────────────────────► XGBoost  ─┤
            ├─► raw features ─────────────────────────────► CatBoost ─┼─► rank blend ─► submission
            └─► raw + original lookups + in-fold TE ──────► XGBoost  ─┘   (weights tuned on OOF AUC)
```

1. **Validation:** 5-fold stratified CV. Every model uses the *same* folds, so the out-of-fold (OOF) predictions can be compared and blended fairly. The CV score, not the public leaderboard, was used for every decision.
2. **Original-data lookup features:** for each column, the fraud rate and the row count of that exact value in the original dataset.
3. **In-fold target encoding:** each feature, plus all 28 pairs of the 8 strongest features, is encoded with its fraud rate using scikit-learn's `TargetEncoder`. The encoder is fit only on the training part of each fold, with internal cross-fitting, so validation labels never leak into the features.
4. **Blending:** each model's predictions are converted to ranks, because AUC only depends on ordering. A grid search then picks the blend weights that maximize OOF AUC.
5. **Model persistence:** all 20 fold models (4 models × 5 folds) and their target encoders are pickled to `models/`. The preprocessing state is pickled to `models/artifacts.pkl`: category lists, original-data lookup tables, target-encoding vocabularies and blend weights. `predict.py` loads them to score new transactions without retraining. Values never seen in training map to "unknown", and the encoder falls back to the overall fraud rate.

### Experiments that did *not* help

- **Adding the original dataset as extra training rows:** 0.94655 vs 0.94659, i.e. no gain. Using it as lookup *features* worked better.
- **The three raw-feature models are highly correlated,** so blending them alone gained only about +0.0002 AUC.

## Project structure

```
.
├── fraud_blend.py        # training: features → 4 models × 5 folds → blend → submission + pickles
├── predict.py            # inference: load pickles → fraud probability for any CSV
├── requirements.txt
├── data/                 # competition files (not committed)
├── data_orig/            # original dataset (not committed)
├── preds/                # cached OOF/test predictions (not committed)
└── models/               # pickled fold models + artifacts.pkl (not committed, too large)
```

## How to run

```bash
# 1. environment
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. data (needs a Kaggle API token in ~/.kaggle/kaggle.json and the competition rules accepted)
kaggle competitions download -c playground-series-s6e11 -p data
kaggle datasets download -d uditjain13/credit-card-fraud-detection-2026 -p data_orig
# unzip both archives into their folders

# 3. train + blend
python fraud_blend.py             # uses NVIDIA GPU for XGBoost/CatBoost
python fraud_blend.py --cpu       # CPU only
python fraud_blend.py --retrain   # ignore cached predictions in preds/
```

The script writes `submission_blend2.csv`, which can be submitted with:

```bash
kaggle competitions submit -c playground-series-s6e11 -f submission_blend2.csv -m "blend2"
```

### Score new data with the pickled models

```bash
python predict.py data/test.csv                    # -> predictions.csv
python predict.py new_transactions.csv -o out.csv
```

```python
import pandas as pd
from predict import predict

proba, per_model = predict(pd.read_csv("new_transactions.csv"))
```

`predict.py` blends **probabilities**, not ranks, so it also works on a single transaction. Its ordering matches the Kaggle submission.

## Tech

Python · pandas · scikit-learn · LightGBM · XGBoost (CUDA) · CatBoost (GPU) · Kaggle API

## Next steps

- Train CatBoost and LightGBM on the target-encoded features too
- Optuna hyperparameter tuning
- Multi-seed averaging, and a more diverse model (e.g. a neural net) for the blend
