"""
preprocessing.py

Preprocessing functions for the used car price prediction dataset.

Author: Quang Huynh
"""

import re
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    """
    description
    Load the dataset from a CSV file into a pandas DataFrame

    :param file_path: Path to the CSV dataset file
    :returns: DataFrame containing the raw dataset
    """
    return pd.read_csv(file_path)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    description
    Clean and standardize column names by stripping whitespace, converting
    names to lowercase, and replacing spaces with underscores

    :param df: Input DataFrame
    :returns: DataFrame with standardized column names
    """
    df = df.copy()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset

    :param df: Input DataFrame
    :returns: DataFrame with duplicate rows removed
    """
    return df.drop_duplicates().copy()


def clean_price_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the price column from strings such as '$10,300' into numeric values

    :param df: Input DataFrame
    :returns: DataFrame with a numeric price column
    """
    df = df.copy()
    if "price" in df.columns:
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def clean_milage_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the milage column from strings such as '51,000 mi.' into numeric values

    :param df: Input DataFrame
    :returns: DataFrame with a numeric milage column
    """
    df = df.copy()
    if "milage" in df.columns:
        df["milage"] = (
            df["milage"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("mi.", "", regex=False)
            .str.replace("mi", "", regex=False)
            .str.strip()
        )
        df["milage"] = pd.to_numeric(df["milage"], errors="coerce")
    return df


def extract_engine_size(engine_value: str) -> float:
    """
    Extract engine displacement size in liters from the engine text field when possible
    Ex: '3.5L V6' becomes 3.5

    :param engine_value: Raw engine string
    :returns: Extracted engine size as a float, or None if not found
    """
    if pd.isna(engine_value):
        return None
    match = re.search(r"(\d+(\.\d+)?)L", str(engine_value))
    if match:
        return float(match.group(1))
    return None


def add_engine_size_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new numeric feature called engine_size_liters by extracting
    engine displacement from the engine column

    :param df: Input DataFrame
    :returns: DataFrame with an added engine_size_liters column
    """
    df = df.copy()
    if "engine" in df.columns:
        df["engine_size_liters"] = df["engine"].apply(extract_engine_size)
    return df


def normalize_text_columns(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
    """
    Normalize string columns by stripping whitespace and converting values to lowercase

    :param df: Input DataFrame
    :param text_columns: List of text column names to normalize
    :returns: DataFrame with normalized text columns
    """
    df = df.copy()
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.lower()
    return df


def drop_rows_with_missing_target(df: pd.DataFrame, target_column: str = "price") -> pd.DataFrame:
    """
    Remove rows where the target column is missing because these rows cannot
    be used for supervised learning

    :param df: Input DataFrame
    :param target_column: Name of the target column
    :returns: DataFrame without rows missing the target value
    """
    if target_column not in df.columns:
        return df.copy()
    return df.dropna(subset=[target_column]).copy()

def remove_price_outliers(df: pd.DataFrame, column: str = "price") -> pd.DataFrame:
    """
    Remove outliers in the price column using the IQR method

    :param df: Input DataFrame
    :param column: Target column name (default = price)
    :returns: DataFrame with outliers removed
    """
    df = df.copy()

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df


def basic_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform the core cleaning steps for the used car dataset, including
    standardizing columns, removing duplicates, cleaning numeric text fields,
    normalizing text columns, engineering engine size, and dropping rows with
    missing target values

    :param df: Raw input DataFrame
    :returns: Cleaned DataFrame ready for train test split and preprocessing pipeline
    """
    df = standardize_column_names(df)
    df = remove_duplicates(df)
    df = clean_price_column(df)
    df = clean_milage_column(df)
    df = remove_price_outliers(df, "price")

    text_columns = [
        "brand",
        "model",
        "fuel_type",
        "engine",
        "transmission",
        "ext_col",
        "int_col",
        "accident",
        "clean_title",
    ]
    df = normalize_text_columns(df, text_columns)
    df = add_engine_size_feature(df)
    df = drop_rows_with_missing_target(df, "price")
    return df


def split_features_target(df: pd.DataFrame, target_column: str = "price") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the cleaned dataset into features and target

    :param df: Cleaned input DataFrame
    :param target_column: Name of the target column
    :returns: Tuple containing X features DataFrame and y target Series
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def get_feature_lists(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical feature columns for preprocessing.

    :param X: Feature DataFrame
    :returns: Tuple containing numeric column list and categorical column list
    """
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessing pipeline that imputes missing values, scales numeric
    features, and one hot encodes categorical features

    :param X: Feature DataFrame used to determine column types
    :returns: Configured ColumnTransformer preprocessing object
    """
    numeric_cols, categorical_cols = get_feature_lists(X)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor


def clean_and_prepare_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    End to end helper function that loads the dataset, performs basic cleaning,
    splits features and target, and builds the preprocessing transformer

    :param file_path: Path to the CSV dataset file
    :returns: Tuple containing X features, y target, and the preprocessing transformer
    """
    df = load_data(file_path)
    df = basic_clean_data(df)
    X, y = split_features_target(df)
    preprocessor = build_preprocessor(X)
    return X, y, preprocessor


# main guard
if __name__ == "__main__":
    """
    Simple test block for quick verification
    """
    FILE_PATH = "data/raw/Used Car Price Prediction Dataset export 2026-03-20 19-46-48.csv"

    X, y, preprocessor = clean_and_prepare_data(FILE_PATH)

    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)
    print("\nFeature columns:")
    print(X.columns.tolist())
    print("\nPreprocessor created successfully.")