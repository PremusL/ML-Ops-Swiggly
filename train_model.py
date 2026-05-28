
# Train a LightGBM regressor to predict restaurant rating from the Swiggly dataset.

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

TRAIN_DATA_PATH = "data/swiggy_remaining_sample.csv"
TEST_DATA_PATH = "data/swiggy_test_sample.csv"
MODEL_OUTPUT_DIR = "models"

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

TOP_N_CUISINES = 30  # keep the 30 most frequent cuisines



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
    features = pd.DataFrame(index=df.index)

    # Cost (numeric, fill nulls with median)
    features["cost"] = df["cost"].fillna(df["cost"].median())
    # Rating count: ordinal
    features["rating_count_ordinal"] = df["rating_count"].map(RATING_COUNT_ORDER).fillna(0).astype(int)
    # City: LightGBM categorical to integers
    features["city"] = df["city"].astype("category").cat.codes
    # Cuisine: multi-hot encoding of top-N tokens
    if fit:
        top_cuisines = _extract_individual_cuisines(df["cuisine"])
    assert top_cuisines is not None

    cuisine_filled = df["cuisine"].fillna("")
    for cuisine in top_cuisines:
        col_name = f"cuisine_{cuisine.lower().replace(' ', '_')}"
        features[col_name] = cuisine_filled.str.contains(cuisine, case=False, regex=False).astype(int)

    feature_names = list(features.columns)
    return features, feature_names, top_cuisines



def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    # Print and return regression metrics.
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



# Main training 

def main(induce_error: bool = False, model_format: str = "text"):

    df_train_full = pd.read_csv(TRAIN_DATA_PATH)

    if induce_error:
        print("ARTIFICIAL ERROR INDUCED: Subsetting training data to 500 rows...")
        df_train_full = df_train_full.head(500)

    if len(df_train_full) < 1000:
        raise ValueError(f"Insufficient training data: {len(df_train_full)} rows (requires >= 1000)")

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
        "n_estimators": 550,
        "learning_rate": 0.02,
        "max_depth": 6,
        "num_leaves": 32,
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

    # Evaluate on test
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
    if model_format == "onnx":
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType

        num_features = X_train.shape[1]
        initial_types = [('input', FloatTensorType([None, num_features]))]
        
        onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_types)
        
        model_filename = f"rating_predictor_lr{model_params['learning_rate']}_md{model_params['max_depth']}_est{model_params['n_estimators']}.onnx"
        model_output_path = os.path.join(MODEL_OUTPUT_DIR, model_filename)
        
        with open(model_output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"\nModel saved to {model_output_path} (ONNX format)")
    else:
        model_filename = f"rating_predictor_lr{model_params['learning_rate']}_md{model_params['max_depth']}_est{model_params['n_estimators']}.txt"
        model_output_path = os.path.join(MODEL_OUTPUT_DIR, model_filename)
        model.booster_.save_model(model_output_path)
        print(f"\nModel saved to {model_output_path} (native text format)")

    print("\nAvailable models:")
    for f in sorted(os.listdir(MODEL_OUTPUT_DIR)):
        if f.endswith(".txt") or f.endswith(".onnx"):
            print(f"  - {f}")

    return {
        "model_params": model_params,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_names": feature_names,
        "model_path": model_output_path,
        "top_cuisines": top_cuisines,
    }


if __name__ == "__main__":
    main()
