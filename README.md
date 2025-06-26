# Task3-Linear-Regression
# Task 3 – Linear Regression: Housing Price Prediction

## Objective

The objective of this task is to implement and evaluate a Linear Regression model to predict housing prices using a real-world dataset. This includes both simple and multiple linear regression techniques. Key steps include data preprocessing, model training, evaluation using regression metrics, and interpretation of model coefficients.

## Dataset

- **Name**: Housing Price Prediction
- **Source**: [Kaggle - Housing Price Prediction](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)
- **File Used**: `Housing.csv`

Each record in the dataset represents a house and includes features such as:
- `area` (square feet)
- `bedrooms`
- `bathrooms`
- `stories`
- `furnishingstatus`
- `mainroad`
- `basement`
- `airconditioning`
- and more.

The target variable is `price`.

## Tools and Libraries

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Implementation Overview

### 1. Data Preprocessing
- Loaded dataset using Pandas.
- Categorical variables were encoded using `pandas.factorize()` to convert string values to numerical codes.
- No missing values were found, so no imputation was required.

### 2. Feature and Target Selection
- `X`: All features except `price`
- `y`: Target column `price`

### 3. Train-Test Split
- Used `train_test_split()` from `scikit-learn` to split data into 80% training and 20% testing sets.

### 4. Model Training
- Used `LinearRegression()` from `scikit-learn` to fit the model on training data.

### 5. Evaluation Metrics
Model performance was evaluated using:
- **MAE (Mean Absolute Error)**: Average absolute difference between actual and predicted values.
- **MSE (Mean Squared Error)**: Average of the squares of the errors.
- **R² Score**: Indicates how much variance in the target is explained by the model.

### 6. Regression Line Plotting (Simple Linear Regression)
- A regression line was plotted using only the `area` feature to show how predicted prices change with square footage.
- Blue points represent actual prices; red line represents the model's prediction.

### 7. Coefficient Interpretation
- **Intercept**: The base price when all features are zero.
- **Slope**: For each additional unit increase in the feature (e.g., 1 sqft of area), how much the predicted price changes.

## Key Results

| Metric        | Value (example)        |
|---------------|------------------------|
| Intercept     | 19815.80               |
| Coefficient (Area) | 122.50            |
| MAE           | ~300000.00             |
| MSE           | ~180000000000.00       |
| R² Score      | ~0.68                  |

These values will vary depending on the dataset and preprocessing.

