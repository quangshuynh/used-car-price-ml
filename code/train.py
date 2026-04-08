"""
train.py

Training severl models (Linear Regression, KNN Regressor,
Decision Tree Regressor, Random Forest Regressor, Neural Network (MLPRegressor), Support Vector Regressor) 
on the used car dataset

Author: Kai Fan kf5601
"""
# TODO: A lot of stuff needs to be moved to evaluate.py, use this version for check in for now

# Import necessary libraries
import os
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from preprocessing import clean_and_prepare_data

# TODO: Move to evaluate.py later, use this version for check in for now
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

# TODO: Move to evaluate.py later, use this version for check in for now
def linear_regression(X_train, y_train, X_val, y_val):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return model, evaluate_model(y_val, y_pred)

# Adapted based on Assignment 4, except eval metrics are now: 
# MSE, RMSE, MAE, R2 instead of accuracy and F1 score
def knn_regressor(X_train, y_train, X_val, y_val):
    results = []

    for k in [3, 5, 7, 9]:
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse, rmse, mae, r2 = evaluate_model(y_val, y_pred)

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
def decision_tree_regressor(X_train, y_train, X_val, y_val):
    results = []

    for depth in [None, 3, 5, 7]:
        model = DecisionTreeRegressor(max_depth=depth, random_state=35)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse, rmse, mae, r2 = evaluate_model(y_val, y_pred)

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
def random_forest_regressor(X_train, y_train, X_val, y_val):
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
            mse, rmse, mae, r2 = evaluate_model(y_val, y_pred)

            results.append({
                "Model": "Random Forest Regressor",
                "Configuration": f"n_estimators = {n}, max_depth = {depth}",
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2
            })

    return results

# adapted based on Assignment 4, except eval metrics are now:
# MSE, RMSE, MAE, R2 instead of accuracy and F1 score
def svr_regressor(X_train, y_train, X_val, y_val):
    results = []

    for kernel_type in ["linear", "rbf"]:
        for c_value in [0.1, 1, 10]:
            model = SVR(kernel=kernel_type, C=c_value)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mse, rmse, mae, r2 = evaluate_model(y_val, y_pred)

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
def mlp_regressor(X_train, y_train, X_val, y_val):
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        max_iter=1000,
        random_state=35
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return model, evaluate_model(y_val, y_pred)


def main():
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
    _, (mse, rmse, mae, r2) = linear_regression(
        X_train_processed, y_train, X_val_processed, y_val
    )
    all_results.append({
        "Model": "Linear Regression",
        "Configuration": "default",
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

    # 2. KNN Regressor
    all_results.extend(
        knn_regressor(X_train_processed, y_train, X_val_processed, y_val)
    )

    # 3. Decision Tree Regressor
    all_results.extend(
        decision_tree_regressor(X_train_processed, y_train, X_val_processed, y_val)
    )

    # 4. Random Forest Regressor
    all_results.extend(
        random_forest_regressor(X_train_processed, y_train, X_val_processed, y_val)
    )

    # 5. SVR
    all_results.extend(
        svr_regressor(X_train_processed, y_train, X_val_processed, y_val)
    )

    # 6. MLP Regressor
    _, (mse, rmse, mae, r2) = mlp_regressor(
        X_train_processed, y_train, X_val_processed, y_val
    )
    all_results.append({
        "Model": "MLP Regressor",
        "Configuration": "hidden_layer_sizes = (32, 16), activation = relu",
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

    # Create results table AFTER all models are added
    results_df = pd.DataFrame(all_results)

    # Sort by RMSE (lower is better)
    results_df = results_df.sort_values(by="RMSE", ascending=True).reset_index(drop=True)

    # Make a cleaner display copy
    results_df_display = results_df.copy()
    results_df_display["MSE"] = results_df_display["MSE"].round(2)
    results_df_display["RMSE"] = results_df_display["RMSE"].round(2)
    results_df_display["MAE"] = results_df_display["MAE"].round(2)
    results_df_display["R2"] = results_df_display["R2"].round(4)

    print("\n=== Model Comparison (Validation Set) ===\n")
    print(results_df_display.to_string(
        index=False,
        col_space=18,
        justify="center"
    ))

    print("\n=== Best Model ===\n")
    best_row = results_df_display.iloc[0]
    print(best_row.to_string())

if __name__ == "__main__":
    main()