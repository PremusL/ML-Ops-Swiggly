"""
List all versioned model runs tracked by MLflow.

Usage:
    uv run python list_models.py
"""

import mlflow

TRACKING_URI = "sqlite:///mlruns.db"
EXPERIMENT_NAME = "swiggly-rating-predictor"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"No experiment '{EXPERIMENT_NAME}' found.")
        print("Run the pipeline first: uv run python flow.py --environment=pypi run")
        return

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    if runs.empty:
        print("No model runs found.")
        return

    # Display columns of interest
    display_cols = [
        "run_id",
        "start_time",
        "status",
    ]

    # Add metric columns that exist
    metric_cols = [
        "metrics.val_mae",
        "metrics.val_rmse",
        "metrics.val_r2",
        "metrics.test_mae",
        "metrics.test_rmse",
        "metrics.test_r2",
    ]
    display_cols += [c for c in metric_cols if c in runs.columns]

    # Add param columns that exist
    param_cols = [
        "params.n_estimators",
        "params.learning_rate",
        "params.max_depth",
    ]
    display_cols += [c for c in param_cols if c in runs.columns]

    print(f"\n{'=' * 80}")
    print(f"  MLflow Model Registry — {EXPERIMENT_NAME}")
    print(f"  Tracking URI: {TRACKING_URI}")
    print(f"  Total runs: {len(runs)}")
    print(f"{'=' * 80}\n")

    for i, (_, row) in enumerate(runs.iterrows()):
        print(f"  Run {i + 1}:")
        print(f"    Run ID     : {row['run_id']}")
        print(f"    Start time : {row['start_time']}")
        print(f"    Status     : {row['status']}")

        if "metrics.val_mae" in runs.columns:
            print(f"    Val  MAE   : {row.get('metrics.val_mae', 'N/A'):.4f}")
            print(f"    Val  RMSE  : {row.get('metrics.val_rmse', 'N/A'):.4f}")
            print(f"    Val  R²    : {row.get('metrics.val_r2', 'N/A'):.4f}")

        if "metrics.test_mae" in runs.columns:
            print(f"    Test MAE   : {row.get('metrics.test_mae', 'N/A'):.4f}")
            print(f"    Test RMSE  : {row.get('metrics.test_rmse', 'N/A'):.4f}")
            print(f"    Test R²    : {row.get('metrics.test_r2', 'N/A'):.4f}")

        if "params.n_estimators" in runs.columns:
            print(f"    Params     : n_estimators={row.get('params.n_estimators', '?')}, "
                  f"lr={row.get('params.learning_rate', '?')}, "
                  f"max_depth={row.get('params.max_depth', '?')}")

        artifact_uri = row.get("artifact_uri", "")
        print(f"    Artifacts  : {artifact_uri}")
        print()

    print(f"  Tip: Run 'uv run mlflow ui --backend-store-uri {TRACKING_URI}' "
          f"to browse in the web UI.\n")


if __name__ == "__main__":
    main()
