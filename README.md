To run the project locally:
For running the project python 3.12 must be used.

Then run the project:
Either using [UV](https://docs.astral.sh/uv/#highlights)
> uv run main.py

Or:
> pip install -r requirements.txt 
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


# ML-Ops-Swiggly
