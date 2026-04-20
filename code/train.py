"""
train.py

Training severl models (Linear Regression, KNN Regressor,
Decision Tree Regressor, Random Forest Regressor, Neural Network (MLPRegressor), Support Vector Regressor) 
on the used car dataset

Author: Kai Fan kf5601
Updated by Quang Huynh (qth9368) 
"""

# Import necessary libraries
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from preprocessing import clean_and_prepare_data
from evaluate import (
    evaluate_model,
    evaluate_and_store,
    build_results_dataframe,
    format_results_dataframe,
)

def linear_regression(X_train, y_train, X_val, y_val, log_transformed: bool = True):
    """
    Train and evaluate a Linear Regression model

    :param X_train: Preprocessed training features
    :param y_train: Training target values
    :param X_val: Preprocessed validation features
    :param y_val: Validation target values
    :param log_transformed: Whether the target was log transformed
    :returns: Tuple of trained model and evaluation metrics
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return model, evaluate_model(y_val, y_pred, log_transformed=log_transformed)

# Adapted based on Assignment 4, except eval metrics are now: 
# MSE, RMSE, MAE, R2 instead of accuracy and F1 score
def knn_regressor(X_train, y_train, X_val, y_val, log_transformed: bool = True):
    """
    Train and evaluate multiple KNN Regressor configurations

    :param X_train: Preprocessed training features
    :param y_train: Training target values
    :param X_val: Preprocessed validation features
    :param y_val: Validation target values
    :param log_transformed: Whether the target was log transformed
    :returns: List of result dictionaries
    """
    results = []

    for k in [3, 5, 7, 9]:
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse, rmse, mae, r2 = evaluate_model(
            y_val, y_pred, log_transformed=log_transformed
        )

        results.append({
            "Model": "KNN Regressor",
            "Configuration": f"k = {k}",
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

    return results

# adapted based on Assignment 4, except eval metrics are now:
# MSE, RMSE, MAE, R2 instead of accuracy and F1 score
def decision_tree_regressor(X_train, y_train, X_val, y_val, log_transformed: bool = True):
    """
    Train and evaluate multiple Decision Tree Regressor configurations

    :param X_train: Preprocessed training features
    :param y_train: Training target values
    :param X_val: Preprocessed validation features
    :param y_val: Validation target values
    :param log_transformed: Whether the target was log transformed
    :returns: List of result dictionaries
    """
    results = []

    for depth in [None, 3, 5, 7]:
        model = DecisionTreeRegressor(max_depth=depth, random_state=35)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse, rmse, mae, r2 = evaluate_model(
            y_val, y_pred, log_transformed=log_transformed
        )

        results.append({
            "Model": "Decision Tree Regressor",
            "Configuration": f"max_depth = {depth}",
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

    return results

# adapted based on Assignment 4, except eval metrics are now:
# MSE, RMSE, MAE, R2 instead of accuracy and F1 score
def random_forest_regressor(X_train, y_train, X_val, y_val, log_transformed: bool = True):
    """
    Train and evaluate multiple Random Forest Regressor configurations

    :param X_train: Preprocessed training features
    :param y_train: Training target values
    :param X_val: Preprocessed validation features
    :param y_val: Validation target values
    :param log_transformed: Whether the target was log transformed
    :returns: List of result dictionaries
    """
    results = []

    for n in [50, 100]:
        for depth in [None, 5, 10]:
            model = RandomForestRegressor(
                n_estimators=n,
                max_depth=depth,
                random_state=35,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mse, rmse, mae, r2 = evaluate_model(
                y_val, y_pred, log_transformed=log_transformed
            )

            results.append({
                "Model": "Random Forest Regressor",
                "Configuration": f"n_estimators = {n}, max_depth = {depth}",
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2
            })

    return results

def svr_regressor(X_train, y_train, X_val, y_val, log_transformed: bool = True):
    """
    Train and evaluate multiple SVR configurations

    :param X_train: Preprocessed training features
    :param y_train: Training target values
    :param X_val: Preprocessed validation features
    :param y_val: Validation target values
    :param log_transformed: Whether the target was log transformed
    :returns: List of result dictionaries
    """
    results = []

    for kernel_type in ["linear", "rbf"]:
        for c_value in [0.1, 1, 10]:
            model = SVR(kernel=kernel_type, C=c_value)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mse, rmse, mae, r2 = evaluate_model(
                y_val, y_pred, log_transformed=log_transformed
            )

            results.append({
                "Model": "SVR",
                "Configuration": f"kernel = {kernel_type}, C = {c_value}",
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2
            })

    return results

# adapted based on Assignment 4, except eval metrics are now:
# MSE, RMSE, MAE, R2 instead of accuracy and F1 score
def mlp_regressor(X_train, y_train, X_val, y_val, log_transformed: bool = True):
    """
    Train and evaluate an MLP Regressor model

    :param X_train: Preprocessed training features
    :param y_train: Training target values
    :param X_val: Preprocessed validation features
    :param y_val: Validation target values
    :param log_transformed: Whether the target was log transformed
    :returns: Tuple of trained model and evaluation metrics
    """
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        max_iter=1000,
        random_state=35
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return model, evaluate_model(y_val, y_pred, log_transformed=log_transformed)

def train_and_store_result(
    model,
    model_name: str,
    configuration: str,
    X_train,
    y_train,
    X_val,
    y_val,
    all_results,
    log_transformed: bool = True
):
    """
    Fit a model, generate predictions on the validation set, and store
    evaluation results in the shared results list

    :param model: Sklearn regression model instance
    :param model_name: Name of the model
    :param configuration: Description of model hyperparameters
    :param X_train: Preprocessed training features
    :param y_train: Training target values
    :param X_val: Preprocessed validation features
    :param y_val: Validation target values
    :param all_results: Shared list storing all result dictionaries
    :param log_transformed: Whether the target was log transformed
    :returns: Trained model
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    evaluate_and_store(
        all_results,
        model_name,
        configuration,
        y_val,
        y_pred,
        log_transformed=log_transformed
    )

    return model


def main():
    """
    Load data, preprocess it, train multiple regression models,
    evaluate them, and print a comparison table

    :returns: None
    """
    # Load and preprocess the data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(
        parent_dir,
        "data",
        "raw",
        "Used Car Price Prediction Dataset export 2026-03-20 19-46-48.csv"
    )

    # Quang's function used to pull preprocessed data
    X, y, preprocessor = clean_and_prepare_data(file_path)

    # Split into training and validation sets
    # Using 20% validation size, and random_state = 35 adapted from Assignment 4
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=35
    )

    # IMPORTANT: fit on train only, transform val only
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    # Store all results here
    all_results = []

    # 1. Linear Regression
    train_and_store_result(
        LinearRegression(),
        "Linear Regression",
        "default",
        X_train_processed,
        y_train,
        X_val_processed,
        y_val,
        all_results
    )

    # 2. KNN Regressor
    for k in [3, 5, 7, 9]:
        train_and_store_result(
            KNeighborsRegressor(n_neighbors=k),
            "KNN Regressor",
            f"k = {k}",
            X_train_processed,
            y_train,
            X_val_processed,
            y_val,
            all_results
        )

    # 3. Decision Tree Regressor
    for depth in [None, 3, 5, 7]:
        train_and_store_result(
            DecisionTreeRegressor(max_depth=depth, random_state=35),
            "Decision Tree Regressor",
            f"max_depth = {depth}",
            X_train_processed,
            y_train,
            X_val_processed,
            y_val,
            all_results
        )

    # 4. Random Forest Regressor
    for n in [50, 100]:
        for depth in [None, 5, 10]:
            train_and_store_result(
                RandomForestRegressor(
                    n_estimators=n,
                    max_depth=depth,
                    random_state=35,
                    n_jobs=-1
                ),
                "Random Forest Regressor",
                f"n_estimators = {n}, max_depth = {depth}",
                X_train_processed,
                y_train,
                X_val_processed,
                y_val,
                all_results
            )

    # 5. SVR
    for kernel_type in ["linear", "rbf"]:
        for c_value in [0.1, 1, 10]:
            train_and_store_result(
                SVR(kernel=kernel_type, C=c_value),
                "SVR",
                f"kernel = {kernel_type}, C = {c_value}",
                X_train_processed,
                y_train,
                X_val_processed,
                y_val,
                all_results
            )

    # 6. MLP Regressor
    train_and_store_result(
        MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=1000,
            random_state=35
        ),
        "MLP Regressor",
        "hidden_layer_sizes = (32, 16), activation = relu",
        X_train_processed,
        y_train,
        X_val_processed,
        y_val,
        all_results
    )

    # Create results table AFTER all models are added
    results_df = build_results_dataframe(all_results)

    # Make a cleaner display copy
    results_df_display = format_results_dataframe(results_df)

    # model comparison table
    print("\n=== Model Comparison (Validation Set) ===\n")
    print(results_df_display.to_string(
        index=False,
        col_space=18,
        justify="center"
    ))

    #  best model
    print("\n=== Best Model ===\nNOTE: NOT TRAINED ON FULL DATA SET YET\n")
    best_row = results_df_display.iloc[0]
    print(best_row.to_string())


    # TODO: delete after: used to just see the price spread from preprocessed data
    print("\n=== Price Distribution (Log Scale) ===\n")
    print(y.describe())

    print("\n=== Price Distribution (Original Dollar Scale) ===\n")
    print(pd.Series(np.expm1(y)).describe())

if __name__ == "__main__":
    main()