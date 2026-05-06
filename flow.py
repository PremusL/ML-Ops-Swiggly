"""
Swiggly ML Pipeline — Metaflow Flow

Step 1 (data_tests):      Data preprocessing + Great Expectations validation
Step 2 (train_model):     LightGBM model training → serialized model artifact
Step 3 (register_model):  Version the model with MLflow tracking

Run locally:
    uv run python flow.py --environment=pypi run

Per-step dependencies are declared via @pypi decorators and also documented
in requirements_data_tests.txt / requirements_train_model.txt /
requirements_register_model.txt.
"""

from metaflow import FlowSpec, step, pypi


class SwigglyFlow(FlowSpec):
    """
    End-to-end pipeline for the Swiggly restaurant dataset.

    1. data_tests      – reduce, sample, and validate with Great Expectations
    2. train_model     – train a LightGBM regressor and save the model artifact
    3. register_model  – version the model in MLflow (local tracking store)
    """

    # ------------------------------------------------------------------
    # start
    # ------------------------------------------------------------------
    @step
    def start(self):
        """Entry point — kicks off the pipeline."""
        print("Starting Swiggly ML pipeline")
        self.next(self.data_tests)

    # ------------------------------------------------------------------
    # Step 1: Data preprocessing & Great Expectations tests
    # ------------------------------------------------------------------
    @pypi(
        packages={
            "pandas": ">=2.0.0",
            "great-expectations": ">=1.0.0",
        }
    )
    @step
    def data_tests(self):
        """
        Execute the full data validation pipeline from main.py:
        - Reduce the raw dataset (clean, stratified sample)
        - Create representative & remaining samples
        - Run Great Expectations null & threshold tests
        """
        # Import inside the step so isolated env can resolve packages
        from main import (
            reduce_dataset,
            create_representative_sample_2,
            check_null_values,
            run_ge_null_tests,
            get_expected_thresholds_rating,
            get_expected_rating_count_set,
            create_random_test_sample,
            run_ge_threshold_tests,
        )

        reduce_dataset()
        create_representative_sample_2()

        null_val_dict = check_null_values()
        print(null_val_dict)

        run_ge_null_tests()

        min_rating, max_rating = get_expected_thresholds_rating()
        rating_count_set = get_expected_rating_count_set()
        create_random_test_sample()

        run_ge_threshold_tests(min_rating, max_rating, rating_count_set)

        print("\n✅ All data tests passed.")
        self.next(self.train_model)

    # ------------------------------------------------------------------
    # Step 2: Model training
    # ------------------------------------------------------------------
    @pypi(
        packages={
            "pandas": ">=2.0.0",
            "lightgbm": ">=4.0.0",
            "scikit-learn": ">=1.3.0",
        }
    )
    @step
    def train_model(self):
        """
        Train a LightGBM regressor to predict restaurant rating.

        The trained model is serialized in LightGBM's native text format
        (human-readable, language-agnostic, no pickle) and stored at
        models/rating_predictor.txt.
        """
        from train_model import main as run_training

        results = run_training()

        # Store training outputs as Metaflow artifacts for downstream steps
        self.model_params = results["model_params"]
        self.val_metrics = results["val_metrics"]
        self.test_metrics = results["test_metrics"]
        self.feature_names = results["feature_names"]
        self.model_path = results["model_path"]

        with open(self.model_path, "r") as f:
            self.model_artifact = f.read()

        print(f"\n📦 Model artifact saved to: {self.model_path}")
        print(f"   Format: LightGBM native text (no pickle)")
        print(f"   Size: {len(self.model_artifact):,} characters")

        self.next(self.register_model)

    # ------------------------------------------------------------------
    # Step 3: Model versioning with MLflow
    # ------------------------------------------------------------------
    @pypi(
        packages={
            "mlflow": ">=2.0.0",
            "lightgbm": ">=4.0.0",
        }
    )
    @step
    def register_model(self):
        """
        Version the trained model using MLflow local tracking.

        Logs hyperparameters, validation/test metrics, and the serialized
        model artifact (LightGBM native text format) to a local MLflow
        tracking store at mlruns/.
        """
        import mlflow
        import mlflow.lightgbm
        import lightgbm as lgb

        # Local file-based tracking store — no server required
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("swiggly-rating-predictor")

        with mlflow.start_run() as run:
            # Log hyperparameters
            mlflow.log_params(self.model_params)

            # Log validation metrics (prefixed for clarity)
            mlflow.log_metrics({
                f"val_{k}": v for k, v in self.val_metrics.items()
            })

            # Log test metrics
            mlflow.log_metrics({
                f"test_{k}": v for k, v in self.test_metrics.items()
            })

            # Log the serialized model file as an artifact
            mlflow.log_artifact(self.model_path, artifact_path="model")

            # Also log the model via MLflow's LightGBM integration
            # for native model loading support
            booster = lgb.Booster(model_file=self.model_path)
            mlflow.lightgbm.log_model(
                booster,
                artifact_path="lightgbm_model",
                input_example=None,
            )

            self.mlflow_run_id = run.info.run_id
            self.mlflow_experiment_id = run.info.experiment_id

        print(f"\n📋 Model registered in MLflow")
        print(f"   Run ID       : {self.mlflow_run_id}")
        print(f"   Experiment   : swiggly-rating-predictor")
        print(f"   Tracking URI : file:./mlruns")
        print(f"   View all runs: uv run python list_models.py")

        self.next(self.end)

    # ------------------------------------------------------------------
    # end
    # ------------------------------------------------------------------
    @step
    def end(self):
        """Pipeline complete."""
        print(f"\n🏁 Swiggly pipeline complete.")
        print(f"   Model artifact : {self.model_path}")
        print(f"   MLflow run ID  : {self.mlflow_run_id}")


if __name__ == "__main__":
    SwigglyFlow()
