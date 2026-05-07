"""
Train a LightGBM regressor to predict restaurant rating from the Swiggly dataset.

Uses only the remaining sample (not the representative sample, which is biased
toward extreme ratings by design). Evaluates on a held-out validation split and
on the pre-built city-stratified test sample.
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_DATA_PATH = "data/swiggy_remaining_sample.csv"
TEST_DATA_PATH = "data/swiggy_test_sample.csv"
MODEL_OUTPUT_DIR = "models"
MODEL_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, "rating_predictor.txt")

TARGET = "rating"

RATING_COUNT_ORDER = {
    "20+ ratings": 1,
    "50+ ratings": 2,
    "100+ ratings": 3,
    "500+ ratings": 4,
    "1K+ ratings": 5,
    "5K+ ratings": 6,
    "10K+ ratings": 7,
}

TOP_N_CUISINES = 30  # keep the 30 most frequent individual cuisines



def _extract_individual_cuisines(series: pd.Series) -> list[str]:
    """Return a sorted list of the top-N individual cuisine tokens."""
    from collections import Counter
    counter: Counter[str] = Counter()
    for entry in series.dropna():
        for cuisine in entry.split(","):
            counter[cuisine.strip()] += 1
    return [c for c, _ in counter.most_common(TOP_N_CUISINES)]


def make_features(
    df: pd.DataFrame,
    top_cuisines: list[str] | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Transform raw columns into model-ready features.

    Parameters
    ----------
    df : raw dataframe (must contain cost, rating_count, city, cuisine)
    top_cuisines : pre-computed list of top cuisine tokens (pass from training
                   when transforming test data)
    fit : if True, derive top_cuisines from df; if False, reuse the passed list

    Returns
    -------
    features : DataFrame with engineered columns
    feature_names : list of feature column names
    top_cuisines : the cuisine list (to be reused for test data)
    """
    features = pd.DataFrame(index=df.index)

    # 1. Cost (numeric, fill nulls with median)
    features["cost"] = df["cost"].fillna(df["cost"].median())
    # 2. Rating count: ordinal
    features["rating_count_ordinal"] = df["rating_count"].map(RATING_COUNT_ORDER).fillna(0).astype(int)
    # 3. City: LightGBM categorical (integer codes)
    features["city"] = df["city"].astype("category").cat.codes
    # 4. Cuisine: multi-hot encoding of top-N tokens
    if fit:
        top_cuisines = _extract_individual_cuisines(df["cuisine"])
    assert top_cuisines is not None

    cuisine_filled = df["cuisine"].fillna("")
    for cuisine in top_cuisines:
        col_name = f"cuisine_{cuisine.lower().replace(' ', '_')}"
        features[col_name] = cuisine_filled.str.contains(cuisine, case=False, regex=False).astype(int)

    feature_names = list(features.columns)
    return features, feature_names, top_cuisines


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Print and return regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{'=' * 50}")
    print(f"  {label} Evaluation")
    print(f"{'=' * 50}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    print(f"{'=' * 50}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main():

    print("Loading training data ...")
    df_train_full = pd.read_csv(TRAIN_DATA_PATH)

    # Drop rows where the target itself is missing or non-numeric
    df_train_full = df_train_full[pd.to_numeric(df_train_full[TARGET], errors="coerce").notna()]
    df_train_full[TARGET] = df_train_full[TARGET].astype(float)

    print(f"Rows after cleaning: {len(df_train_full)}")

    X_all, feature_names, top_cuisines = make_features(df_train_full, fit=True)
    y_all = df_train_full[TARGET].values

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.20, random_state=2000,
    )
    print(f"Train size : {len(X_train)}")
    print(f"Val size   : {len(X_val)}")

    print("\nTraining LightGBM regressor ...")

    model_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 2000,
    }

    model = lgb.LGBMRegressor(**model_params, verbose=-1)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(100)],
    )

    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_metrics = evaluate(y_val, y_val_pred, label="Validation")

    # Evaluate on test set
    print("\nLoading test data ...")
    df_test = pd.read_csv(TEST_DATA_PATH)
    df_test = df_test[pd.to_numeric(df_test[TARGET], errors="coerce").notna()]
    df_test[TARGET] = df_test[TARGET].astype(float)

    X_test, _, _ = make_features(df_test, top_cuisines=top_cuisines, fit=False)
    y_test = df_test[TARGET].values

    y_test_pred = model.predict(X_test)
    test_metrics = evaluate(y_test, y_test_pred, label="Test")

    # Feature importance
    importance = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    print("\nTop 15 feature importances (split count):")
    print(importance.head(15).to_string())

    # Save model
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model.booster_.save_model(MODEL_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")

    return {
        "model_params": model_params,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_names": feature_names,
        "model_path": MODEL_OUTPUT_PATH,
        "top_cuisines": top_cuisines,
    }


if __name__ == "__main__":
    main()
