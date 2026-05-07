import great_expectations as gx
import pandas as pd

# Explanations of tests and the reference dataset are in README.md

def main():
    reduce_dataset()
    create_representative_sample_2()
    null_val_dict = check_null_values()
    print(null_val_dict)
    run_ge_null_tests()

    min_rating, max_rating = get_expected_thresholds_rating()

    rating_count_set = get_expected_rating_count_set()
    create_random_test_sample()
    
    run_ge_threshold_tests(min_rating, max_rating, rating_count_set)
    

def reduce_dataset():
    df = pd.read_csv('data/swiggy.csv')

    print(f"Original dataset size: {len(df)}")

    df = df.dropna(subset=['rating_count'])
    df = df[df['rating_count'] != 'Too Few Ratings']

    df['cost'] = df['cost'].str.replace('₹', '', regex=False).str.strip()
    df['cost'] = pd.to_numeric(df['cost'], errors='raise')

    df['rating_count'] = df['rating_count'].astype(str).str.strip()

    sampled_df = df.groupby('city').sample(frac=0.35)
    
    output_file = 'data/swiggy_sample_21k.csv'
    sampled_df.to_csv(output_file, index=False)

    print(f"Sampled dataset columns: {list(sampled_df.columns)}, {len(sampled_df)}")



def create_representative_sample_2():
    df = pd.read_csv('data/swiggy_sample_21k.csv')
    

    df = df.sort_values(by='rating', ascending=False)
    explainable_sample_1 = df.drop_duplicates(subset=['city', 'rating_count'], keep='first')


    df = df.sort_values(by='rating', ascending=True)
    explainable_sample_2 = df.drop_duplicates(subset=['city', 'rating_count'], keep='first')

    explainable_sample = pd.concat([explainable_sample_1, explainable_sample_2])

    df_remaining = df.drop(explainable_sample.index)

    output_repr_file = 'data/swiggy_representative_sample.csv'
    output_remaining_file = 'data/swiggy_remaining_sample.csv'

    explainable_sample.to_csv(output_repr_file, index=False)
    df_remaining.to_csv(output_remaining_file, index=False)

    print(f"Representative sample size: {explainable_sample.shape}")
    print(f"Remaining sample size: {df_remaining.shape}")
    

def check_null_values():
    df = pd.read_csv('data/swiggy_representative_sample.csv')
    null_val_dict = df.isnull().sum()/len(df)
    df_remaining = pd.read_csv('data/swiggy_remaining_sample.csv')
    
    print(df_remaining[df_remaining.rating == '--'].rating)
    return null_val_dict



def run_ge_null_tests():
    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas("swiggy_data")

    data_asset = data_source.add_csv_asset(
        "remaining_sample", 
        filepath_or_buffer="data/swiggy_remaining_sample.csv"
    )
    batch_def = data_asset.add_batch_definition_whole_dataframe("remaining_sample_batch")
    batch = batch_def.get_batch()
    
    cost_null_test = gx.expectations.ExpectColumnValuesToNotBeNull(
        column="cost",
        mostly=0.99 # there are some rating count with value in the representative sample 0.001330
    )
    
    result_dash = batch.validate(cost_null_test)
    
    if not result_dash.success:
        print(f"  -> Found {result_dash.result['unexpected_percent']:.2f}% '--' values.")
    print(f"Cost Null Test Passed: {result_dash.success}")
    if not result_dash.success:
        print(f"  -> Found {result_dash.result['unexpected_percent']:.2f}% '--' values.")


def get_expected_thresholds_rating():
    df = pd.read_csv('data/swiggy_representative_sample.csv')

    cost_min = df['rating'].min()
    cost_max = df['rating'].max()
    
    print(f"Rating min: {cost_min}, Rating max: {cost_max}")
    return (cost_min, cost_max)


def get_expected_rating_count_set():
    df = pd.read_csv('data/swiggy_representative_sample.csv')

    rating_count_unique_set = df['rating_count'].unique().tolist()

    print(f"Rating count unique set: {rating_count_unique_set}")
    return rating_count_unique_set

def create_random_test_sample():
    df_test = pd.read_csv('data/swiggy_remaining_sample.csv')

    sampled_df = df_test.groupby('city').sample(frac=0.20)
    
    # Remove the test data from the remaining sample
    remaining_df = df_test.drop(sampled_df.index)
    remaining_df.to_csv('data/swiggy_remaining_sample.csv', index=False)
    
    output_file = 'data/swiggy_test_sample.csv'
    sampled_df.to_csv(output_file, index=False)

    print(f"Sampled test dataset columns: {list(sampled_df.columns)}, {len(sampled_df)}")
    print(f"Remaining training dataset size: {len(remaining_df)}")


def run_ge_threshold_tests(min_rating, max_rating, rating_count_set):
    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas("swiggy_data")

    data_asset = data_source.add_csv_asset(
        "swiggy_test_sample", 
        filepath_or_buffer="data/swiggy_test_sample.csv"
    )
    batch_def = data_asset.add_batch_definition_whole_dataframe("remaining_sample_batch")
    batch = batch_def.get_batch()
    
    rating_range_test = gx.expectations.ExpectColumnValuesToBeBetween(
        column="rating",
        min_value=min_rating,
        max_value=max_rating,
        mostly=1.0
    )    

    rating_count_test = gx.expectations.ExpectColumnValuesToBeInSet(
        column="rating_count",
        value_set=rating_count_set,
        mostly=1.0
    )

    result_rating_count = batch.validate(rating_count_test)
    result_rating = batch.validate(rating_range_test)

    print(f"Rating Count Test Passed: {result_rating_count.success}")
    if not result_rating_count.success:
        print(f"  -> Found {result_rating_count.result['unexpected_percent']:.2f}% values out of range.")
        
    print(f"Rating Range Test Passed: {result_rating.success}")
    if not result_rating.success:
        print(f"  -> Found {result_rating.result['unexpected_percent']:.2f}% values out of range.")
    
    
if __name__ == "__main__":
    main()