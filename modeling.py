"""
modeling.py — Predictive Maintenance (Imbalanced) Modeling

What this script does:
- Loads preprocessed.csv
- Creates a realistic TIME-based train/test split (no random leakage)
- Runs TimeSeriesSplit cross-validation on the TRAIN set only
- Trains multiple baseline models (LogReg, RandomForest, HistGB)
- Evaluates on the held-out TEST set with metrics suited for imbalance:
  PR-AUC (main), ROC-AUC, F1, Recall, Precision, Confusion Matrix
- Tunes a decision threshold on the TRAIN set (via CV) and applies it to TEST
- Saves:
  - metrics_summary.csv
  - best_model.joblib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

import joblib


DATA_PATH = r"C:\Users\irene\OneDrive\Υπολογιστής\TuE\projects\predictive_maintenance\datasets\preprocessed.csv"
OUT_DIR = r"C:\Users\irene\OneDrive\Υπολογιστής\TuE\projects\predictive_maintenance\datasets\outputs"

RANDOM_STATE = 42
TEST_QUANTILE = 0.80
N_SPLITS_CV = 5               # TimeSeriesSplit folds on train only for no leak
THRESHOLD_GRID = np.linspace(0.05, 0.95, 19)  # thresholds to try


# 1) Load and sort
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values(["date", "device"]).reset_index(drop=True)  # stable ordering



# Define features/target
y = df["failure"].astype(int)
X = df.drop(columns=["failure"])

# Drop identifiers from features for modeling
DROP_COLS = ["date", "device"]
X_model = X.drop(columns=DROP_COLS, errors="ignore")


# 2) Time-based train/test split
cutoff = df["date"].quantile(TEST_QUANTILE)
train_mask = df["date"] <= cutoff
test_mask = df["date"] > cutoff

X_train, y_train = X_model.loc[train_mask], y.loc[train_mask]
X_test, y_test = X_model.loc[test_mask], y.loc[test_mask]

print("Cutoff date:", cutoff)
print("Train:", X_train.shape, " Test:", X_test.shape)
print("Train failure rate:", y_train.mean())
print("Test  failure rate:", y_test.mean())


# 3) Models
# - LogisticRegression benefits from scaling.
# - RandomForest doesn’t need scaling but works fine with it
# - HistGradientBoosting is strong + fasts

common_preprocess = [
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
]

models = {
    "LogReg_balanced": Pipeline(
        steps=common_preprocess + [
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=None
            ))
        ]
    ),
    "RandomForest_balanced": Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_depth=None
            ))
        ]
    ),
    "HistGB": Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                random_state=RANDOM_STATE,
                max_depth=None,
                learning_rate=0.1
            ))
        ]
    ),
}


# 4) Helpers: CV, threshold tuning, evaluation
def get_proba(model, X):
    """Return P(y=1). Supports models with predict_proba or decision_function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        # convert scores to pseudo-probabilities via logistic transform
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    raise ValueError("Model has neither predict_proba nor decision_function.")


def time_series_cv_scores(model, Xtr, ytr, n_splits=5):
    """TimeSeriesSplit CV on training set returns list of PR-AUC scores and OOF probabilities."""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    pr_auc_scores = []
    oof_proba = pd.Series(index=ytr.index, dtype=float)

    for fold, (idx_tr, idx_val) in enumerate(tscv.split(Xtr)):
        X_fold_tr = Xtr.iloc[idx_tr]
        y_fold_tr = ytr.iloc[idx_tr]
        X_fold_val = Xtr.iloc[idx_val]
        y_fold_val = ytr.iloc[idx_val]

        model.fit(X_fold_tr, y_fold_tr)
        p_val = get_proba(model, X_fold_val)

        pr_auc = average_precision_score(y_fold_val, p_val)
        pr_auc_scores.append(pr_auc)

        oof_proba.iloc[idx_val] = p_val

        print(f"  Fold {fold+1}/{n_splits} PR-AUC: {pr_auc:.4f}")

    return pr_auc_scores, oof_proba


def choose_threshold(y_true, proba, grid=THRESHOLD_GRID, objective="f1"):

    best_t, best_score = 0.5, -1.0

    if objective == "f1":
        for t in grid:
            pred = (proba >= t).astype(int)
            score = f1_score(y_true, pred, zero_division=0)
            if score > best_score:
                best_score, best_t = score, t
        return best_t, best_score

    if objective.startswith("recall_at_precision"):
        # Example: "recall_at_precision>=0.2"
        target_precision = float(objective.split(">=")[-1])
        for t in grid:
            pred = (proba >= t).astype(int)
            prec = precision_score(y_true, pred, zero_division=0)
            rec = recall_score(y_true, pred, zero_division=0)
            # only consider thresholds meeting the precision constraint
            if prec >= target_precision and rec > best_score:
                best_score, best_t = rec, t
        return best_t, best_score

def evaluate_on_test(name, model, Xtr, ytr, Xte, yte, threshold, save_pr_curve=True):
    model.fit(Xtr, ytr)
    p_test = get_proba(model, Xte)

    # Core metrics for imbalance
    pr_auc = average_precision_score(yte, p_test)
    roc_auc = roc_auc_score(yte, p_test)

    pred = (p_test >= threshold).astype(int)
    prec = precision_score(yte, pred, zero_division=0)
    rec = recall_score(yte, pred, zero_division=0)
    f1 = f1_score(yte, pred, zero_division=0)
    cm = confusion_matrix(yte, pred) #

    result = {
        "model": name,
        "threshold": float(threshold),
        "PR_AUC": float(pr_auc),
        "ROC_AUC": float(roc_auc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    if save_pr_curve:
        precision_arr, recall_arr, _ = precision_recall_curve(yte, p_test)
        plt.figure()
        plt.plot(recall_arr, precision_arr)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision–Recall Curve (TEST) — {name}")
        plt.savefig(os.path.join(OUT_DIR, f"pr_curve_{name}.png"), dpi=160, bbox_inches="tight")
        plt.close()

    return result


# 5) Run CV + threshold tuning + final test evaluation
all_results = []
best_by_pr_auc = {"model": None, "mean_cv_pr_auc": -1.0}

for name, model in models.items():
    print(f"\nModel: {name}")
    cv_scores, oof_proba = time_series_cv_scores(model, X_train, y_train, n_splits=N_SPLITS_CV)
    mean_cv = float(np.mean(cv_scores))
    std_cv = float(np.std(cv_scores))

    # Choose threshold using OUT-OF-FOLD proba ->train-only
    chosen_t, chosen_score = choose_threshold(y_train, oof_proba, objective="f1")

    print(f"  Mean CV PR-AUC: {mean_cv:.4f} ± {std_cv:.4f}")
    print(f"  Chosen threshold (train OOF, objective=f1): {chosen_t:.2f} (OOF F1={chosen_score:.4f})")

    # Final evaluation on held-out test
    test_result = evaluate_on_test(
        name=name,
        model=model,
        Xtr=X_train,
        ytr=y_train,
        Xte=X_test,
        yte=y_test,
        threshold=chosen_t,
        save_pr_curve=True
    )
    test_result["mean_cv_PR_AUC"] = mean_cv
    test_result["std_cv_PR_AUC"] = std_cv
    all_results.append(test_result)

    # Track best model by mean CV PR-AUC (train-only selection)
    if mean_cv > best_by_pr_auc["mean_cv_pr_auc"]:
        best_by_pr_auc = {
            "model": name,
            "mean_cv_pr_auc": mean_cv,
            "threshold": chosen_t
        }

results_df = pd.DataFrame(all_results).sort_values(by="mean_cv_PR_AUC", ascending=False)
results_path = os.path.join(OUT_DIR, "metrics_summary.csv")
results_df.to_csv(results_path, index=False)

print("\n=== Summary (sorted by mean CV PR-AUC) ===")
print(results_df[["model", "mean_cv_PR_AUC", "PR_AUC", "precision", "recall", "f1", "threshold"]])
print(f"\nSaved metrics to: {results_path}")

# 6) Train + save best model on full TRAIN
best_name = best_by_pr_auc["model"]
best_threshold = best_by_pr_auc["threshold"]
best_model = models[best_name]

best_model.fit(X_train, y_train)

model_path = os.path.join(OUT_DIR, "best_model.joblib")
joblib.dump({"model": best_model, "threshold": best_threshold, "feature_columns": X_train.columns.tolist()}, model_path)

print(f"\nBest model (by mean CV PR-AUC): {best_name}")
print(f"Saved best model to: {model_path}")

