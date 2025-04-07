#!/usr/bin/env python
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Read the data from the data folder
    data_path = os.path.join('data', 'AmesHousing.csv')
    housing_df = pd.read_csv(data_path)

    # Show the count of missing values (optional: you may choose to log or print these)
    missing_vals = housing_df.isnull().sum()
    missing_vals = missing_vals[missing_vals > 0].sort_values(ascending=False)
    print("Missing values per column:\n", missing_vals)

    # Drop columns with high missing data
    high_missing_cols = ['Pool QC', 'Misc Feature', 'Alley', 'Fence']
    housing_df.drop(columns=high_missing_cols, inplace=True)

    # Fill categorical columns with mode
    cat_fill_mode = ['Mas Vnr Type', 'Fireplace Qu', 'Garage Type', 'Garage Finish', 
                     'Garage Qual', 'Garage Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 
                     'BsmtFin Type 2', 'Bsmt Qual', 'Bsmt Cond', 'Electrical']
    for col in cat_fill_mode:
        if col in housing_df.columns:
            housing_df[col].fillna(housing_df[col].mode()[0], inplace=True)
        else:
            print(f"Warning: Column {col} not in dataset.")

    # Fill numerical columns with median
    num_fill_median = ['Mas Vnr Area', 'Lot Frontage', 'Garage Yr Blt', 'Bsmt Half Bath', 
                       'Bsmt Full Bath', 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 
                       'Total Bsmt SF', 'Garage Cars', 'Garage Area']
    for col in num_fill_median:
        if col in housing_df.columns:
            housing_df[col].fillna(housing_df[col].median(), inplace=True)
        else:
            print(f"Warning: Column {col} not in dataset.")

    # One-hot encode categorical variables (drop_first to avoid dummy variable trap)
    housing_df_encoded = pd.get_dummies(housing_df, drop_first=True)

    # Split the data into features and target
    if 'SalePrice' not in housing_df_encoded.columns:
        raise ValueError("Expected target column 'SalePrice' not found.")
    y = housing_df_encoded['SalePrice']
    X = housing_df_encoded.drop(columns=['SalePrice'])

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

    # Create output folder if it does not exist and save files
    out_dir = 'data_cleaned'
    os.makedirs(out_dir, exist_ok=True)
    X_train.to_parquet(os.path.join(out_dir, 'X_train.parquet'))
    X_test.to_parquet(os.path.join(out_dir, 'X_test.parquet'))
    y_train.to_frame().to_parquet(os.path.join(out_dir, 'y_train.parquet'))
    y_test.to_frame().to_parquet(os.path.join(out_dir, 'y_test.parquet'))

    print(f"Preprocessing complete. Cleaned data saved to: {out_dir}")

if __name__ == '__main__':
    main()
