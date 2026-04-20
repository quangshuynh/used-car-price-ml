# Used Car Price Prediction

## Abstract
This project predicts used car selling prices from vehicle listing data using supervised machine learning regression. The dataset includes attributes such as brand, model year, mileage, fuel type, engine description, transmission, exterior and interior color, accident history, clean title status, and sale price. The workflow cleans and prepares the raw data, engineers useful numeric features, trains multiple traditional and neural-network-based models, and compares them using regression metrics on a held-out validation set.

## Developers
- Quang Huynh
- Kai Fan
- Dawn Grant

## Repository Structure
```text
.
|-- code/
|   |-- preprocessing.py
|   |-- train.py
|   |-- evaluate.py
|   `-- notebooks/
|       `-- cars-price-prediction.ipynb
|-- data/
|   `-- raw/
|       `-- Used Car Price Prediction Dataset export 2026-03-20 19-46-48.csv
|-- resources/
|   `-- CSCI-335 Machine Learning Presentation (Group 3).pdf
|-- README.md
`-- requirements.txt
```

## Dataset
- **Dataset:** Used Car Price Prediction Dataset
- **Source:** Gigasheet sample data based on used car listings
- **Link:** https://www.gigasheet.com/sample-data/used-car-price-prediction-dataset
- **Local file:** `data/raw/Used Car Price Prediction Dataset export 2026-03-20 19-46-48.csv`
- **Rows in raw dataset:** 4,009
- **Target variable:** `price`
- **Task type:** Regression

## Project Goals
- Clean and preprocess real used-car listing data.
- Handle missing values, duplicate rows, text-formatted numeric values, and price outliers.
- Engineer an `engine_size_liters` feature from the raw engine description.
- Apply numerical scaling and categorical one-hot encoding.
- Train and compare several regression models.
- Evaluate model performance using dollar-scale error metrics and R2.
- Identify the strongest model and explain limitations and possible improvements.

## Methodology
### Data Preprocessing
The preprocessing pipeline is implemented in `code/preprocessing.py` and performs the following steps:

- Standardizes column names.
- Removes duplicate rows.
- Converts `price` values such as `$10,300` into numeric values.
- Converts `milage` values such as `51,000 mi.` into numeric values.
- Drops rows with missing target prices.
- Removes price outliers using the IQR method.
- Applies `log1p` transformation to the price target to reduce skew.
- Normalizes text columns by trimming whitespace and converting values to lowercase.
- Extracts `engine_size_liters` from the engine description when available.
- Splits features and target.
- Uses median imputation and standard scaling for numeric features.
- Uses most-frequent imputation and one-hot encoding for categorical features.

### Models
The training script in `code/train.py` compares the following models:

- Linear Regression
- K-Nearest Neighbors Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor
- Multi-Layer Perceptron Regressor

Several models are evaluated with different hyperparameter settings, including KNN neighbor count, decision tree depth, random forest tree count and depth, and SVR kernel and `C` value.

### Evaluation
The project uses an 80/20 train-validation split with `random_state=35`. Model predictions are converted back from log scale to original dollar scale before evaluation.

Metrics:

- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error
- R2: Coefficient of determination

Current best validation result from `python code/train.py`:

| Model | Configuration | RMSE | MAE | R2 |
| --- | --- | ---: | ---: | ---: |
| SVR | `kernel = linear, C = 1` | 8577.33 | 5539.65 | 0.8336 |

This result is based on the current validation split and has not been retrained on the full dataset.

## How To Run
### 1. Clone or open the repository
```bash
cd path/to/ml
```

### 2. Create a virtual environment
Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run model training and evaluation
```bash
python code/train.py
```

The script loads the CSV from `data/raw/`, preprocesses the data, trains all configured models, prints a model comparison table, and reports the best validation model.


## Computational Resources
The project is designed to run on a standard laptop CPU. Random Forest uses all available CPU cores through `n_jobs=-1`, so runtime may vary by machine. If runtime becomes an issue, reduce the number of random forest estimators, test fewer hyperparameter combinations, or run the notebook and scripts in Google Colab.

## Result Analysis
The current validation results show that the linear-kernel Support Vector Regressor performs best among the tested configurations. This suggests that the transformed and encoded feature space contains strong linear structure after preprocessing. Linear Regression also performs reasonably well, while deeper tree-based models show weaker validation results, which may indicate overfitting or limited benefit from tree splits after one-hot encoding. The MLP Regressor is included as a neural-network baseline but does not currently outperform the best traditional model.

Limitations:
- The dataset contains listing data rather than confirmed final transaction prices.
- Some useful details may be hidden inside text fields such as `engine` and `transmission`.
- The current split uses one validation set rather than repeated cross-validation.
- The best model has not yet been retrained on the full cleaned dataset.
- External market factors such as location, seasonality, and demand are not included.

Potential improvements:
- Add cross-validation for more stable model comparison.
- Tune SVR, Random Forest, and MLP hyperparameters more systematically.
- Engineer additional features from engine, transmission, brand/model combinations, and vehicle age.
- Save the best trained model and preprocessing pipeline for reuse.
- Add charts for residual analysis, prediction error distribution, and feature importance where applicable.
