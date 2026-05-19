from metaflow import FlowSpec, step, pypi, catch, Parameter


class SwiggyFlow(FlowSpec):
    """
    End-to-end pipeline for the Swiggly restaurant dataset.
    1. data_tests      - reduce, sample, and validate with Great Expectations
    2. train_model     - train a LightGBM regressor and save the model artifact
    3. register_model  - version the model in MLflow (local tracking store)
    4. test_robustness - evaluate the model for robustness against anomalies/perturbations
    """

    induce_error = Parameter(
        "induce_error",
        help="Simulate a training error (insufficient data)",
        default=False,
    )

    model_format = Parameter(
        "model_format",
        help="Model serialization format (text or onnx)",
        default="text",
    )

    @step
    def start(self):
        print("Starting Swiggy ML pipeline")
        self.next(self.data_tests)

    @pypi(
        packages={
            "pandas": "2.3.3",
            "great-expectations": "1.16.1",
        })
    @step
    def data_tests(self):
        """
        Execute the full data validation pipeline from main.py:
        - Reduce the raw dataset (clean, stratified sample)
        - Create representative & remaining samples
        - Run Great Expectations null & threshold tests
        """

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

        print("\nAll data tests passed.")
        self.next(self.train_model)

    @pypi(
        packages={
            "pandas": "2.3.3",
            "lightgbm": "4.6.0",
            "scikit-learn": "1.8.0",
            "onnxmltools": "1.16.0",
            "onnx": "1.21.0",
        })
    @catch(var="train_error")
    @step
    def train_model(self):
        """
        Train a LightGBM regressor to predict restaurant rating.

        Error Handling Choice:
        We use Metaflow's @catch decorator to gracefully handle unexpected system errors
        or data validation issues during training (e.g., if the training data size
        drops below 1k records). If an exception is raised, the flow does not crash
        immediately. Instead, the error is caught and stored in `self.train_error`.
        Downstream steps then check for this error and can skip execution, ensuring
        we don't deploy or test a broken model.
        """
        from train_model import main as run_training

        results = run_training(induce_error=self.induce_error, model_format=self.model_format)

        # Store training outputs as Metaflow artifacts for downstream steps
        self.model_params = results["model_params"]
        self.val_metrics = results["val_metrics"]
        self.test_metrics = results["test_metrics"]
        self.feature_names = results["feature_names"]
        self.model_path = results["model_path"]
        self.top_cuisines = results["top_cuisines"]
        self.model_format_used = self.model_format

        with open(self.model_path, "rb") as f:
            self.model_artifact = f.read()

        print(f"\nModel artifact saved to: {self.model_path}")
        print(f"   Format: {self.model_format_used}")
        print(f"   Size: {len(self.model_artifact):,} bytes")

        self.next(self.register_model)


    # Should it pick the best performing saved model?
    @pypi(
        packages={
            "pandas": "2.3.3",
            "lightgbm": "4.6.0",
            "numpy": "2.4.4",
            "scikit-learn": "1.8.0",
            "onnxruntime": "1.26.0",
        })
    @step
    def test_robustness(self):
        
        if getattr(self, "train_error", None):
            print(f"Skipping robustness test due to training error: {self.train_error}")
            self.next(self.end)
            return

        import pandas as pd
        import numpy as np
        from train_model import make_features
        
        print("\nTesting Model Robustness...")
        
        # Load test data and engineer baseline features
        df_test = pd.read_csv("data/swiggy_test_sample.csv")
        X_test, _, _ = make_features(df_test, top_cuisines=self.top_cuisines, fit=False)

        if self.model_format_used == "onnx":
            import onnxruntime as rt
            sess = rt.InferenceSession(self.model_path)
            input_name = sess.get_inputs()[0].name
            
            def predict_model(X):
                return sess.run(None, {input_name: X.values.astype(np.float32)})[0].flatten()
        else:
            import lightgbm as lgb
            booster = lgb.Booster(model_file=self.model_path)
            
            def predict_model(X):
                return booster.predict(X)
        
        baseline_preds = predict_model(X_test)
        
        # --------------------------------------------------------------
        # Test A: Extreme Outlier Bounds
        # --------------------------------------------------------------
        print("\n--- Test A: Extreme Outliers ---")
        print("Expectation: Model predictions should remain within the valid [1.0, 5.0]")
        print("range even if the cost is absurdly high (e.g., ₹10,000,000).")
        
        X_extreme = X_test.copy()
        X_extreme["cost"] = 10000000
        extreme_preds = predict_model(X_extreme)
        
        out_of_bounds = np.sum((extreme_preds < 1.0) | (extreme_preds > 5.0))
        print(f"Predictions out of bounds [1.0, 5.0]: {out_of_bounds} / {len(extreme_preds)}")
        
        if out_of_bounds == 0:
            print("Passed outlier robustness test.")
        else:
            print(f"Failed outlier robustness test. Max pred: {np.max(extreme_preds):.2f}, Min pred: {np.min(extreme_preds):.2f}")
            
        # --------------------------------------------------------------
        # Test B: Invariance to Price Inflation
        # --------------------------------------------------------------
        print("\n--- Test B: Invariance to Minor Perturbations ---")
        print("Expectation: Increasing the cost uniformly by 20% should not shift")
        print("the predicted rating by more than 0.2 stars on average.")
        X_perturbed = X_test.copy()
        X_perturbed["cost"] = X_perturbed["cost"] * 1.20
        perturbed_preds = predict_model(X_perturbed)
        
        mean_diff = np.mean(np.abs(perturbed_preds - baseline_preds))
        print(f"Mean absolute difference in predictions: {mean_diff:.4f} stars")
        
        if mean_diff < 0.2:
            print("Passed perturbation robustness test.")
        else:
            print("Failed perturbation robustness test.")
            
        self.next(self.end)

    # ------------------------------------------------------------------
    # Step 3: Model versioning with MLflow
    # ------------------------------------------------------------------
    @pypi(
        packages={
            "mlflow": "3.12.0",
            "lightgbm": "4.6.0",
            "onnx": "1.21.0",
            "onnxruntime": "1.26.0",
        })
    @step
    def register_model(self):
        """
        Version the trained model using MLflow local tracking.

        Logs hyperparameters, validation/test metrics, and the serialized
        model artifact (LightGBM native text format or ONNX) to a local MLflow
        tracking store at sqlite:///mlruns.db.
        """
        if getattr(self, "train_error", None):
            print(f"Skipping model registration due to training error: {self.train_error}")
            self.next(self.test_robustness)
            return

        import mlflow
        import mlflow.lightgbm
        import mlflow.onnx
        import onnx
        import lightgbm as lgb
        import list_models

        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        mlflow.set_experiment("swiggly-rating-predictor")

        with mlflow.start_run() as run:
            mlflow.log_params(self.model_params)

            mlflow.log_metrics({
                f"val_{k}": v for k, v in self.val_metrics.items()
            })

            # Log test metrics
            mlflow.log_metrics({
                f"test_{k}": v for k, v in self.test_metrics.items()
            })

            # Log the serialized model file as an artifact
            mlflow.log_artifact(self.model_path, artifact_path="model")

            # Also log the model via MLflow's native integration
            if self.model_format_used == "onnx":
                onnx_model = onnx.load(self.model_path)
                mlflow.onnx.log_model(
                    onnx_model,
                    artifact_path="onnx_model",
                )
            else:
                booster = lgb.Booster(model_file=self.model_path)
                mlflow.lightgbm.log_model(
                    booster,
                    artifact_path="lightgbm_model",
                    input_example=None,
                )

            self.mlflow_run_id = run.info.run_id
            self.mlflow_experiment_id = run.info.experiment_id


        print(f"\nModel registered in MLflow")
        print(f"Run ID       : {self.mlflow_run_id}")
        print(f"Experiment   : swiggly-rating-predictor")
        print(f"Tracking URI : sqlite:///mlruns.db")
        print(f"View all runs: uv run python list_models.py")

        self.next(self.test_robustness)

    @step
    def end(self):
        """Pipeline complete."""
        print(f"\nSwiggly pipeline complete.")
        print(f"   Model artifact : {self.model_path}")
        print(f"   MLflow run ID  : {self.mlflow_run_id}")


if __name__ == "__main__":
    SwiggyFlow()
