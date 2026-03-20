# Car Prices Model Predictor

## Team Members
- Quang Huynh
- Kai Fan
- Dawn Grant

## Project Overview
Buying or selling a used car can be difficult because prices vary based on many factors such as brand, body type, mileage, engine size, horsepower, fuel type, and more.

This project builds a machine learning regression model that predicts the selling price of a car using historical car listing data. Our goal is to compare multiple regression models and determine which one best predicts car prices.

## Problem Statement
Used car prices vary widely depending on many vehicle characteristics. This project aims to build a machine learning model that can accurately predict a car's selling price based on its features.

## Dataset
- **Dataset Name:** Car Price DataSet
- **Source:** Kaggle
- **Link:** https://www.kaggle.com/datasets/mirzahasnine/car-price-dataset
- **File Name:** `car_prices.csv`

## Project Goals
- Clean and preprocess the dataset
- Explore patterns and relationships between car features and price
- Train multiple regression models
- Compare model performance using regression metrics
- Identify which features are most useful for predicting price

## Methodology

### 1. Data Preprocessing
We will prepare the dataset by:
- Handling missing values if present
- Removing duplicate rows
- Checking for and handling outliers
- Encoding categorical variables using one-hot encoding or label encoding
- Scaling numerical features when needed
- Separating features and target variable

### 2. Exploratory Data Analysis
We will perform EDA to better understand the data, including:
- Distribution of car prices
- Summary statistics of numerical features
- Correlation analysis
- Visualizations such as histograms, boxplots, and scatterplots
- Feature importance analysis for tree-based models

### 3. Models
We plan to test and compare the following regression models:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Possible future extension:
- Gradient Boosting Regressor
- XGBoost, if time allows

### 4. Validation Strategy
- Train test split using 80 percent training and 20 percent testing
- Cross-validation for more reliable model comparison

### 5. Evaluation Metrics
Since this is a regression problem, we will evaluate models using:
- MAE, Mean Absolute Error
- MSE, Mean Squared Error
- RMSE, Root Mean Squared Error
- R² Score

## Tools and Libraries
We expect to use:
- Python
- pandas
- numpy
- matplotlib
- scikit-learn
- jupyter notebook
