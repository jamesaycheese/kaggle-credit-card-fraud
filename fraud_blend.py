"""
Kaggle Playground S6E11 - Predicting Credit Card Fraud
End-to-end pipeline that reproduces the "blend2" submission:

    LightGBM + XGBoost + CatBoost (raw features)
  + XGBoost with original-dataset lookup features and in-fold target encoding
  -> rank-averaged blend with weights chosen on out-of-fold (OOF) AUC

Every fold model is pickled to ./models, together with the preprocessing
artifacts, so predict.py can score new data without retraining.

Usage:
    python fraud_blend.py               # train (re-uses cached preds + models if present)
    python fraud_blend.py --retrain     # ignore cache and retrain all models
    python fraud_blend.py --cpu         # run without an NVIDIA GPU
"""
import argparse
import glob
import itertools
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# ----------------------------------------------------------------------------- config
DATA_DIR = "data"            # competition files: train.csv, test.csv, sample_submission.csv
ORIG_DIR = "data_orig"       # original dataset (uditjain13/credit-card-fraud-detection-2026)
PRED_DIR = "preds"           # cached OOF / test predictions (.npy)
MODEL_DIR = "models"         # pickled fold models + artifacts.pkl
OUT_FILE = "submission_blend2.csv"
TARGET = "is_fraud"
SEED = 42
N_FOLDS = 5

# strongest single features -> all pairwise combinations get target-encoded
TOP_PAIR_FEATURES = [
    "merchant_risk_score", "velocity_score", "cvv_retry_count", "is_new_merchant",
    "customer_age", "txn_count_last_24h", "ip_country_mismatch", "hours_since_last_txn",
]


# ----------------------------------------------------------------------------- pickle helpers
def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def model_path(name, fold):
    return f"{MODEL_DIR}/{name}_fold{fold}.pkl"


# ----------------------------------------------------------------------------- data
def load_data():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
    orig_files = glob.glob(f"{ORIG_DIR}/**/*.csv", recursive=True)
    if not orig_files:
        raise FileNotFoundError(f"No CSV found in {ORIG_DIR}/ - download the original dataset first.")
    orig = pd.read_csv(orig_files[0])
    print(f"train {train.shape} | test {test.shape} | original {orig.shape}")
    return train, test, sub, orig


def column_types(train):
    raw_cols = [c for c in train.columns if c not in ("id", TARGET)]
    bool_cols = [c for c in raw_cols if pd.api.types.is_bool_dtype(train[c])]
    cat_cols = [c for c in raw_cols
                if not pd.api.types.is_numeric_dtype(train[c]) and c not in bool_cols]
    return raw_cols, cat_cols, bool_cols


def norm(s):
    """String key for a column so bools / 0-1 / 'True' all compare equal."""
    s_str = s.astype(str).str.lower()
    if pd.api.types.is_bool_dtype(s) or s_str.isin(["true", "false"]).all():
        return s_str.map({"true": "1", "false": "0", "1": "1", "0": "0", "1.0": "1", "0.0": "0"})
    return s.astype(str)


# ----------------------------------------------------------------------------- features
def base_features(df, raw_cols, cat_cols, bool_cols, categories=None):
    """Raw features: text -> pandas category, True/False -> 0/1."""
    X = df[raw_cols].copy()
    for c in bool_cols:
        X[c] = norm(X[c]).astype(float).astype(int)
    for c in cat_cols:
        cats = None if categories is None else categories[c]
        X[c] = pd.Categorical(X[c].astype(str), categories=cats)
    return X.reset_index(drop=True)


def to_cat_input(X, cat_cols):
    """CatBoost wants plain strings for categorical columns."""
    Xc = X.copy()
    for c in cat_cols:
        Xc[c] = Xc[c].astype(str)
    return Xc


def original_stats(orig, raw_cols):
    """Per column: fraud rate and row count of each value in the original dataset."""
    y_orig = orig[TARGET].astype(int)
    return {c: y_orig.groupby(norm(orig[c])).agg(["mean", "count"]) for c in raw_cols}


def original_lookup_features(df, stats, raw_cols):
    out = pd.DataFrame(index=range(len(df)))
    for c in raw_cols:
        key = norm(df[c]).reset_index(drop=True)
        out[f"te_{c}"] = key.map(stats[c]["mean"]).astype(float)   # NaN if value unseen
        out[f"cnt_{c}"] = key.map(stats[c]["count"]).fillna(0)
    return out


class KeyBuilder:
    """Turns each column (and each pair of top features) into integer category keys.
    Fitted vocabularies are pickled, so new data maps to the same keys (unseen -> -1)."""

    def __init__(self, raw_cols, pair_features):
        self.raw_cols = raw_cols
        top = [c for c in pair_features if c in raw_cols]
        self.pair_list = list(itertools.combinations(top, 2))

    def _single_codes(self, df):
        return {c: self.vocab[c].get_indexer(norm(df[c])).astype(np.int64) for c in self.raw_cols}

    def _combine(self, ca, cb, b):
        return (ca + 1) * (len(self.vocab[b]) + 1) + (cb + 1)

    def fit(self, df):
        self.vocab = {c: pd.Index(norm(df[c]).unique()) for c in self.raw_cols}
        codes = self._single_codes(df)
        self.pair_vocab = {(a, b): pd.Index(np.unique(self._combine(codes[a], codes[b], b)))
                           for a, b in self.pair_list}
        return self

    def transform(self, df):
        codes = self._single_codes(df)
        out = dict(codes)
        for a, b in self.pair_list:
            out[f"{a}__{b}"] = self.pair_vocab[(a, b)].get_indexer(self._combine(codes[a], codes[b], b))
        return pd.DataFrame(out).astype("int32")


# ----------------------------------------------------------------------------- CV runner
def run_cv(name, fit_predict, y, n_test, folds, retrain):
    """fit_predict(tr_idx, va_idx) -> (pred_valid, pred_test, fitted_object).
    Caches predictions in PRED_DIR and pickles each fold's fitted object in MODEL_DIR."""
    oof_path, test_path = f"{PRED_DIR}/oof_{name}.npy", f"{PRED_DIR}/test_{name}.npy"
    cached = (os.path.exists(oof_path) and os.path.exists(test_path)
              and all(os.path.exists(model_path(name, k)) for k in range(len(folds))))
    if cached and not retrain:
        oof, pred = np.load(oof_path), np.load(test_path)
        print(f"[{name}] loaded from cache - CV AUC {roc_auc_score(y, oof):.5f}")
        return oof, pred

    oof, pred = np.zeros(len(y)), np.zeros(n_test)
    for fold, (tr, va) in enumerate(folds):
        p_va, p_te, fitted = fit_predict(tr, va)
        oof[va] = p_va
        pred += p_te / len(folds)
        save_pickle(fitted, model_path(name, fold))
        print(f"[{name}] fold {fold}: AUC {roc_auc_score(y.iloc[va], p_va):.5f}")
    print(f"[{name}] CV AUC {roc_auc_score(y, oof):.5f}")
    np.save(oof_path, oof)
    np.save(test_path, pred)
    return oof, pred


# ----------------------------------------------------------------------------- models
def make_lgb(X, X_test, y):
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  n_estimators=5000, n_jobs=-1, random_state=SEED, verbose=-1)

    def fit_predict(tr, va):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False)])
        return m.predict_proba(X.iloc[va])[:, 1], m.predict_proba(X_test)[:, 1], {"model": m}
    return fit_predict


def xgb_model(device, lr=0.03, colsample=0.7):
    return xgb.XGBClassifier(
        n_estimators=10000, learning_rate=lr, max_depth=6, min_child_weight=5,
        subsample=0.8, colsample_bytree=colsample, reg_lambda=1.0,
        tree_method="hist", device=device, enable_categorical=True, max_cat_to_onehot=1,
        eval_metric="auc", early_stopping_rounds=300, random_state=SEED)


def make_xgb(X, X_test, y, device):
    def fit_predict(tr, va):
        m = xgb_model(device)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
        return m.predict_proba(X.iloc[va])[:, 1], m.predict_proba(X_test)[:, 1], {"model": m}
    return fit_predict


def make_cat(X, X_test, y, cat_cols, use_gpu):
    Xc, Xc_test = to_cat_input(X, cat_cols), to_cat_input(X_test, cat_cols)
    gpu = dict(task_type="GPU", devices="0") if use_gpu else {}

    def fit_predict(tr, va):
        m = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                               eval_metric="AUC", od_type="Iter", od_wait=200,
                               cat_features=cat_cols, verbose=0, random_seed=SEED, **gpu)
        m.fit(Xc.iloc[tr], y.iloc[tr], eval_set=(Xc.iloc[va], y.iloc[va]), use_best_model=True)
        return m.predict_proba(Xc.iloc[va])[:, 1], m.predict_proba(Xc_test)[:, 1], {"model": m}
    return fit_predict


def te_columns(K):
    return [f"TE_{c}" for c in K.columns]


def make_xgb_te(XF, XF_test, y, K_tr, K_te, device):
    """XGBoost on raw + original-lookup features + in-fold target encoding."""
    cols = te_columns(K_tr)

    def fit_predict(tr, va):
        te = TargetEncoder(target_type="binary",
                           cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
        T_tr = pd.DataFrame(te.fit_transform(K_tr.iloc[tr], y.iloc[tr]), columns=cols)
        T_va = pd.DataFrame(te.transform(K_tr.iloc[va]), columns=cols)
        T_te = pd.DataFrame(te.transform(K_te), columns=cols)
        X_tr = pd.concat([XF.iloc[tr].reset_index(drop=True), T_tr], axis=1)
        X_va = pd.concat([XF.iloc[va].reset_index(drop=True), T_va], axis=1)
        X_te = pd.concat([XF_test, T_te], axis=1)
        m = xgb_model(device, colsample=0.5)
        m.fit(X_tr, y.iloc[tr], eval_set=[(X_va, y.iloc[va])], verbose=False)
        return m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1], {"model": m, "te": te}
    return fit_predict


# ----------------------------------------------------------------------------- blend
def rank(p):
    return rankdata(p) / len(p)


def blend_weights(oofs, y, step=0.1):
    """Grid search over weights (sum to 1) on rank-transformed OOF predictions."""
    names = list(oofs)
    ranked = {k: rank(v) for k, v in oofs.items()}
    grid = np.round(np.arange(0, 1 + 1e-9, step), 2)
    best_auc, best_w = 0.0, None
    for w in itertools.product(grid, repeat=len(names) - 1):
        if sum(w) > 1 + 1e-9:
            continue
        weights = dict(zip(names, [float(v) for v in w] + [round(1 - float(sum(w)), 2)]))
        auc = roc_auc_score(y, sum(weights[k] * ranked[k] for k in names))
        if auc > best_auc:
            best_auc, best_w = auc, weights
    return best_auc, best_w


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true", help="ignore cached predictions/models")
    ap.add_argument("--cpu", action="store_true", help="do not use the GPU")
    args = ap.parse_args()
    use_gpu = not args.cpu
    device = "cuda" if use_gpu else "cpu"
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train, test, sub, orig = load_data()
    raw_cols, cat_cols, bool_cols = column_types(train)
    print("categorical:", cat_cols)
    y = train[TARGET].astype(int).reset_index(drop=True)

    # features (everything needed to rebuild them is saved in artifacts.pkl)
    X = base_features(train, raw_cols, cat_cols, bool_cols)
    categories = {c: X[c].cat.categories for c in cat_cols}
    X_test = base_features(test, raw_cols, cat_cols, bool_cols, categories)

    stats = original_stats(orig, raw_cols)
    XF = pd.concat([X, original_lookup_features(train, stats, raw_cols)], axis=1)
    XF_test = pd.concat([X_test, original_lookup_features(test, stats, raw_cols)], axis=1)

    keys = KeyBuilder(raw_cols, TOP_PAIR_FEATURES).fit(pd.concat([train, test], ignore_index=True))
    K_tr, K_te = keys.transform(train), keys.transform(test)

    # same folds for every model -> OOF predictions are comparable and blendable
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))

    models = {
        "lgb": make_lgb(X, X_test, y),
        "xgb": make_xgb(X, X_test, y, device),
        "cat": make_cat(X, X_test, y, cat_cols, use_gpu),
        "xgbT": make_xgb_te(XF, XF_test, y, K_tr, K_te, device),
    }
    oofs, tests = {}, {}
    for name, fit_predict in models.items():
        oofs[name], tests[name] = run_cv(name, fit_predict, y, len(test), folds, args.retrain)

    auc, weights = blend_weights(oofs, y)
    print(f"\nBlend CV AUC {auc:.5f}  weights {weights}")

    save_pickle({
        "raw_cols": raw_cols, "cat_cols": cat_cols, "bool_cols": bool_cols,
        "categories": categories, "orig_stats": stats, "key_builder": keys,
        "weights": weights, "n_folds": N_FOLDS,
        "cv_auc": {k: float(roc_auc_score(y, v)) for k, v in oofs.items()} | {"blend": float(auc)},
    }, f"{MODEL_DIR}/artifacts.pkl")
    print(f"Saved models + {MODEL_DIR}/artifacts.pkl")

    sub[TARGET] = sum(weights[k] * rank(tests[k]) for k in tests)
    sub.to_csv(OUT_FILE, index=False)
    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    # Run through the imported module so pickled classes are saved as
    # fraud_blend.KeyBuilder (not __main__.KeyBuilder) and predict.py can load them.
    import fraud_blend
    fraud_blend.main()
