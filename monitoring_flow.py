from metaflow import FlowSpec, step, pypi, Parameter

class SwiggyMonitoringFlow(FlowSpec):
    simulate_drift = Parameter(
        "simulate_drift",
        help="Simulate data drift by inflating cost values",
        default=False,
    )

    @step
    def start(self):
        """
        Start the monitoring pipeline.
        Initialize data paths and log entry.
        """
        print("Starting Swiggy ML Monitoring pipeline")
        self.next(self.detect_drift)

    @pypi(
        packages={
            "pandas": "2.3.3",
            "scipy": "1.17.1",
            "numpy": "2.4.4",
        }
    )
    @step
    def detect_drift(self):
        """
        Perform Data Drift (specifically Covariate Shift) detection on the 'cost' feature.
        
        Drift Type Monitored:
        --------------------
        - Data Drift (Covariate Shift): Detects whether the distribution of the independent 
          feature 'cost' (the cost of eating at a restaurant) has shifted between training 
          time and current/production serving time.
          
        Expected Behavior:
        -----------------
        - The current dataset ('data/swiggy_test_sample.csv') represents incoming production/test 
          data. The baseline dataset ('data/swiggy_remaining_sample.csv') represents the original 
          training data distribution.
        - Under normal operations, the incoming data is expected to be drawn from the same 
          underlying population as the training dataset. Therefore, the statistical test should 
          FAIL to reject the null hypothesis of identical distributions (i.e. KS p-value >= 0.05).
          
        Source of Expectation:
        ----------------------
        - Sourced directly from the training dataset ('data/swiggy_remaining_sample.csv'). The model
          was trained on this data and expects similar statistical properties to remain valid.
          Note: We avoid using the representative sample ('data/swiggy_representative_sample.csv') 
          as a baseline here because it is deliberately biased toward city/rating extremes 
          and thus does not represent the standard feature distribution.
          
        Statistical Tests Performed:
        ---------------------------
        1. Two-sample Kolmogorov-Smirnov (KS) Test (continuous numeric):
           - Measures the maximum vertical distance (statistic D) between the cumulative empirical 
             distribution functions (CDFs) of baseline and current cost samples.
             H0 (Null Hypothesis): The two samples are drawn from the identical continuous distribution.
             Ha (Alternative): They are from different distributions.
             Significance level: alpha = 0.05. If p-value < 0.05, we reject H0 and flag drift.
             
        2. Kullback-Leibler (KL) Divergence (binned probabilities):
           - Measures the expected relative entropy or information loss when using the current 
             distribution Q to approximate the baseline distribution P.
             A value close to 0 indicates identical distributions. Higher values indicate drift.
             Approximate continuous distributions using 20 percentile-based bins with Laplace 
             smoothing (1e-5) to avoid zero probabilities.
        """
        import pandas as pd
        import numpy as np
        import scipy.stats as stats

        # Load datasets
        df_base = pd.read_csv("data/swiggy_remaining_sample.csv")
        df_curr = pd.read_csv("data/swiggy_test_sample.csv")

        # Extract 'cost' feature and drop missing values
        base_cost = df_base['cost'].dropna()
        curr_cost = df_curr['cost'].dropna()

        # Simulate drift if parameter is set
        if self.simulate_drift:
            print("SIMULATING DRIFT: Applying 50% inflation (x1.5) to cost values of current dataset...")
            curr_cost = curr_cost * 1.5

        # 1. Kolmogorov-Smirnov Test
        self.ks_stat, self.p_value = stats.ks_2samp(base_cost, curr_cost)
        
        # 2. KL Divergence (discrete binning approximation)
        bins = np.percentile(base_cost, np.linspace(0, 100, 20))
        bins = np.unique(bins)
        
        base_hist, _ = np.histogram(base_cost, bins=bins)
        curr_hist, _ = np.histogram(curr_cost, bins=bins)
        
        # Laplace smoothing to avoid division by zero
        base_probs = (base_hist + 1e-5) / (len(base_cost) + 1e-5 * len(base_hist))
        curr_probs = (curr_hist + 1e-5) / (len(curr_cost) + 1e-5 * len(curr_hist))
        
        self.kl_div = stats.entropy(base_probs, curr_probs)

        # Drift Assessment
        self.drift_detected = self.p_value < 0.05

        print("\n================ DRIFT MONITORING REPORT ================")
        print(f"Feature Monitored : cost")
        print(f"Baseline Size    : {len(base_cost)}")
        print(f"Current Size     : {len(curr_cost)}")
        print(f"KS Statistic      : {self.ks_stat:.4f}")
        print(f"KS p-value        : {self.p_value:.4f}")
        print(f"KL Divergence     : {self.kl_div:.4f}")
        print(f"Drift Detected    : {self.drift_detected}")
        print("=========================================================")

        if self.drift_detected:
            print("[ALERT] Significant data drift detected on 'cost' column! Model predictions may be unreliable.")
        else:
            print("[OK] No significant data drift detected. The data distribution matches expectations.")

        self.next(self.end)

    @step
    def end(self):
        """
        Finish monitoring flow.
        """
        print("Swiggy monitoring flow execution complete.")

if __name__ == "__main__":
    SwiggyMonitoringFlow()
