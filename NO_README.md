To run the project locally:
For running the project python 3.12 must be used.

Then run the project:
Either using [UV](https://docs.astral.sh/uv/#highlights)
> uv run main.py

Or:
> pip install -r requirements/requirements.txt 
> python main.py

To run the project in a Docker container:
> docker build -t swiggly .
> docker run swiggly


## About the dataset
The dataset is from [Kaggle](https://www.kaggle.com/datasets/ashishjangra27/swiggy-restaurants-dataset?select=swiggy.csv). This dataset is the data of all the restaurants listed on Swiggy in India. Swiggy is a food ordering and delivery company.

The dataset contains the following columns:
id - Every restaurant is having a unique ID
name - Name of the Restaurant
city - The city where the restaurant is located
rating - Rating of the Restaurant
rating_count - Number of People given the Rating
cost - Cost of eating in that restaurant
cuisine - Cuisines that restaurant serves
lic_no - License number of that restaurant
link - Restaurant link on Swiggy website
address - Full address of the restaurant

Original dataset size: 148k, which is then reduced to 21485 rows.
1. Throwing Out Bad Data
If a restaurant has a blank rating or explicitly says it doesn't have enough reviews to get a proper score, those rows are completely removed.
2. Fixing the Prices
It then looks at the cost column. It removes the currency symbols which is a Rupee sign and any extra spaces so that only the raw numbers are left. 
3. Sampling
This is the main step that reduces the size of the data. Instead of just taking a random chunk of the whole file, it organizes the remaining restaurants by their city. Then, it randomly selects 35% of the restaurants from each city. This creates a much smaller dataset, but ensures that every city is still fairly represented just like it was in the original data.


### Selecting Representative Sample
As my representative sample I took the restaurants with the highest and lowest rating for each city with all the rating count categories. Highest and lowest rating counts are taken in order to capture the extremes of the data which are the most interesting and important for a business. Those reviews change effect the business the most because they can dramatically increase or decrease the restaurants visibility and attract or repel customers and therefore are the most important ones to keep and other data should fall between the values of those extremes.

### Unit tests for null values
Unit test for null values check if there are any null values in the cost column, with 99% threshold. That is because even in my refrence dataset there are some rows that have missing cost values, and consequently the tests are not expected to pass with 100% threshold. There could be a restourant that is actually giving out food for free.  

### Unit tests for threshold values
Unit test for threshold values check if the values in the rating column are within the range of minimum and maximum rating from the representative sample, because representative sample captures all the extremes of the data, therefore tests have to follow the ranges. There is no threshold on this test because the test should always pass otherwise the dataset does not following rating system or have corrupted data.

Unit test for rating count checks if all the rating count values in the remaining sample are within the unique set of rating count values from the representative sample, because there should be no new rating count categories in the remaining sample because the representative sample should capture all the rating count categories. There is no threshold on this test aswell because the rating count categories should be consistent throughout the dataset.



## Swiggly ML Pipeline — Metaflow Flow

The end-to-end machine learning pipeline is managed using **Metaflow** and consists of the following steps:
1. **`data_tests`**: Preprocessing (sample reduction, stratification) + Great Expectations null & threshold validations.
2. **`train_model`**: Trains a LightGBM regressor to predict restaurant ratings.
3. **`register_model`**: Versions the trained model using MLflow local tracking.
4. **`test_robustness`**: Evaluates model robustness against perturbations and extreme inputs.
5. **`end`**: Wraps up pipeline run.

To run the pipeline with the default native text model format:
```bash
uv run python flow.py --environment=pypi run --model_format text
```

To run the pipeline with ONNX model format:
```bash
uv run python flow.py --environment=pypi run --model_format onnx
```

---

## Key Design Decisions

### 1. Training Data Splitting Choice
The model is trained strictly on the **remaining sample** (`swiggy_remaining_sample.csv`) and evaluated on the held-out validation set and city-stratified test sample (`swiggy_test_sample.csv`). We explicitly avoid training on the **representative sample**, since it contains the extreme highest/lowest ratings per city (designed for threshold test baselines). Training on such a dataset would heavily bias the regressor.

### 2. Model Serialization Format
The trained model can be serialized in two formats, selectable via the `--model_format` input parameter:
* **Native Text Format (Default)**: Saved as `models/rating_predictor_lr<lr>_md<md>_est<est>.txt`. It is human-readable, language-agnostic, completely safe from arbitrary code execution exploits, and requires no Python runtime environment to load or execute predictions.
* **ONNX Format**: Converted using `onnxmltools` and saved as `models/rating_predictor_lr<lr>_md<md>_est<est>.onnx`. This maximizes platform interoperability, allowing it to run via `onnxruntime` in C++, C#, Java, or JavaScript runtimes.

### 3. Error Handling & Recovery Choice
We use Metaflow's `@catch(var="train_error")` decorator on the `train_model` step to gracefully handle unexpected system crashes or data validation issues during training (such as the training dataset size falling below 1,000 records, which raises a `ValueError`). 
If an exception is raised:
* The error is captured and stored in `self.train_error` instead of crashing the flow.
* Downstream steps (`register_model` and `test_robustness`) check for `self.train_error` and safely bypass execution. This ensures we never deploy, version, or test a corrupted model, maintaining pipeline stability.

### 4. Model Versioning & Registry Store Choice
We use **MLflow** for model registry and hyperparameter/metric tracking:
* The tracking store is configured to write to a local directory: `mlruns` (`mlflow.set_tracking_uri("mlruns")`). This makes the registry fully portable and easily browseable using the MLflow UI (by running `uv run mlflow ui`).
* If ONNX format is selected, it versions the model using `mlflow.onnx.log_model`. If native text format is selected, it uses `mlflow.lightgbm.log_model`.

### 5. Model Robustness Testing Expectations
We define model robustness as the capability to handle anomalous inputs and mild perturbations without predicting absurd values:
* **Test A (Extreme Outliers)**: Expects that even under an absurdly inflated input cost (e.g., ₹10,000,000), predictions do not extrapolate wildly and remain strictly bound to the valid restaurant rating range of `[1.0, 5.0]`.
* **Test B (Price Inflation Invariance)**: Expects that minor perturbations (e.g., a 20% uniform price inflation) shift the predicted ratings by less than 0.2 stars on average, as ratings should be more strongly driven by cuisine, city, and rating counts rather than minor price variances.












def make_features
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