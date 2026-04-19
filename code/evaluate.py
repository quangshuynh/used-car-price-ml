"""
evaluate.py

Evaluation helpers for used car price prediction models

Author: Quang Huynh (qth9368)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(y_true, y_pred, log_transformed: bool = True):
    """
    Evaluate regression predictions using MSE, RMSE, MAE, and R2.
    If the target was log transformed with np.log1p, convert both true
    values and predictions back to original dollar scale before scoring

    :param y_true: Ground truth target values
    :param y_pred: Predicted target values
    :param log_transformed: Whether y_true and y_pred are in log1p space
    :returns: Tuple of mse, rmse, mae, r2
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if log_transformed:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, rmse, mae, r2


def evaluate_and_store(results_list, model_name: str, configuration: str, y_true, y_pred, log_transformed: bool = True):
    """
    Evaluate a model prediction and append its metrics as a dictionary
    into a shared results list

    :param results_list: List that stores all model result dictionaries
    :param model_name: Name of the model
    :param configuration: Description of model hyperparameters
    :param y_true: Ground truth target values
    :param y_pred: Predicted target values
    :param log_transformed: Whether y_true and y_pred are in log1p space
    :returns: None
    """
    mse, rmse, mae, r2 = evaluate_model(
        y_true,
        y_pred,
        log_transformed=log_transformed
    )

    results_list.append({
        "Model": model_name,
        "Configuration": configuration,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })


def build_results_dataframe(results_list):
    """
    Convert a list of model evaluation dictionaries into a sorted DataFrame

    :param results_list: List of dictionaries containing evaluation results
    :returns: Sorted pandas DataFrame
    """
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by="RMSE", ascending=True).reset_index(drop=True)
    return results_df


def format_results_dataframe(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a rounded copy of the results DataFrame for cleaner printing

    :param results_df: Raw results DataFrame
    :returns: Rounded display DataFrame
    """
    results_df_display = results_df.copy()
    results_df_display["MSE"] = results_df_display["MSE"].round(2)
    results_df_display["RMSE"] = results_df_display["RMSE"].round(2)
    results_df_display["MAE"] = results_df_display["MAE"].round(2)
    results_df_display["R2"] = results_df_display["R2"].round(4)
    return results_df_display