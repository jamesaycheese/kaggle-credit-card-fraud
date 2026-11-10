"""
Score new transactions with the pickled models from fraud_blend.py.

Usage:
    python predict.py data/test.csv                   # writes predictions.csv
    python predict.py new_transactions.csv -o out.csv

Output: one fraud probability per row (weighted average of the 4 models, each
averaged over its 5 fold models). Unlike the Kaggle submission, which blends
*ranks*, this blends probabilities so it also works for a single transaction.
"""
import argparse

import numpy as np
import pandas as pd

from fraud_blend import (MODEL_DIR, TARGET, base_features, load_pickle, model_path,
                         original_lookup_features, te_columns, to_cat_input)


def predict(df, model_dir=MODEL_DIR):
    art = load_pickle(f"{model_dir}/artifacts.pkl")
    raw_cols, cat_cols = art["raw_cols"], art["cat_cols"]

    X = base_features(df, raw_cols, cat_cols, art["bool_cols"], art["categories"])
    XF = pd.concat([X, original_lookup_features(df, art["orig_stats"], raw_cols)], axis=1)
    K = art["key_builder"].transform(df)
    inputs = {"lgb": X, "xgb": X, "cat": to_cat_input(X, cat_cols)}

    per_model = {}
    for name in art["weights"]:
        fold_preds = []
        for k in range(art["n_folds"]):
            fitted = load_pickle(model_path(name, k))
            if name == "xgbT":
                T = pd.DataFrame(fitted["te"].transform(K), columns=te_columns(K))
                X_in = pd.concat([XF, T], axis=1)
            else:
                X_in = inputs[name]
            fold_preds.append(fitted["model"].predict_proba(X_in)[:, 1])
        per_model[name] = np.mean(fold_preds, axis=0)

    return sum(w * per_model[k] for k, w in art["weights"].items()), per_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV with the same feature columns as train.csv")
    ap.add_argument("-o", "--out", default="predictions.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    proba, _ = predict(df)
    out = pd.DataFrame({"id": df["id"] if "id" in df else np.arange(len(df)), TARGET: proba})
    out.to_csv(args.out, index=False)
    print(f"Scored {len(df):,} rows -> {args.out}")
    print(out[TARGET].describe())


if __name__ == "__main__":
    main()
