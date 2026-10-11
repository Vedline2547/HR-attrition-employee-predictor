# HR Employee Attrition Prediction using XGBoost

## Project Overview

This project uses **XGBoost (Extreme Gradient Boosting)** to predict employee attrition using the IBM HR Analytics Employee Attrition dataset.

Employee attrition occurs when employees leave an organization. Understanding the factors associated with employee turnover can help organizations identify patterns, improve employee retention strategies, and make data-driven human resource decisions.

The project applies data preprocessing, categorical feature encoding, model training, and performance evaluation to build a binary classification model that predicts whether an employee is likely to leave the organization.

## Objectives

- Analyze employee data to understand factors associated with attrition.
- Preprocess numerical and categorical features for machine learning.
- Apply One-Hot Encoding to categorical variables.
- Train an XGBoost classification model.
- Evaluate model performance using accuracy, precision, recall, and F1-score.
- Explore how machine learning can support employee retention decisions.

## Technologies Used

- **Python** – Programming language
- **Pandas** – Data manipulation and analysis
- **Scikit-learn** – Data preprocessing, pipeline construction, dataset splitting, and evaluation
- **XGBoost** – Gradient boosting classification algorithm

## Dataset

The project uses the IBM HR Analytics Employee Attrition dataset, which contains employee information such as:

- Age and gender
- Department and job role
- Monthly income
- Job satisfaction and environment satisfaction
- Years at the company
- Work-life balance
- Overtime status
- Attrition status

The target variable is `Attrition`, which indicates whether an employee left the organization.

- `No` → 0
- `Yes` → 1

## Project Workflow

1. **Data Loading:** Import the dataset using Pandas.
2. **Data Cleaning:** Remove columns considered unnecessary for this prediction task, including `EmployeeNumber`, `Over18`, and `StandardHours`.
3. **Feature and Target Separation:** Separate employee attributes from the `Attrition` target variable.
4. **Categorical Encoding:** Apply `OneHotEncoder` to convert categorical features into numerical representations.
5. **Train-Test Split:** Split the data into 80% training and 20% testing sets, using stratification to preserve the class distribution.
6. **Model Training:** Train an XGBoost classifier using the training data.
7. **Model Evaluation:** Evaluate predictions using accuracy and a classification report.

## Model Configuration

The XGBoost classifier uses the following parameters:

| Parameter | Value |
|---|---:|
| `n_estimators` | 200 |
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `objective` | `binary:logistic` |
| `eval_metric` | `logloss` |
| `random_state` | 42 |

These settings provide a starting point for binary classification and can be optimized through cross-validation and hyperparameter tuning.

## Evaluation Metrics

The model is evaluated using:

- **Accuracy:** The proportion of correct predictions.
- **Precision:** How many employees predicted to leave actually left.
- **Recall:** How many employees who left were correctly identified.
- **F1-score:** The harmonic mean of precision and recall.

Because employee attrition datasets are typically imbalanced, recall and F1-score for employees who leave are particularly important.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Vedline2547/HR-attrition-employee-predictor.git
cd HR-Employee-Attrition-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install pandas scikit-learn xgboost
```

### 4. Add the dataset

Place `WA_Fn-UseC_-HR-Employee-Attrition.csv` in the `Attrition` folder, or update the dataset path in your Python script to match your local directory structure.

### 5. Run the project

```bash
python main.py
```

Replace `main.py` with the actual name of your Python script if it differs.

## Results

The model outputs an accuracy score and a classification report containing precision, recall, and F1-score for both attrition classes.

Actual performance results should be recorded after running the model. No specific accuracy or performance score is claimed in this README.

## Potential Improvements

- Perform exploratory data analysis and visualize attrition patterns.
- Apply hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`.
- Compare XGBoost with Random Forest, Logistic Regression, and LightGBM.
- Investigate class imbalance and tune the classification threshold.
- Analyze feature importance to identify influential predictors.
- Deploy the trained model using Flask or Streamlit.

## Disclaimer

This project is intended for educational and research purposes. Predictions should not be used as the sole basis for employment decisions. Employee data must be handled responsibly, with attention to privacy, fairness, and potential model bias.

## Author

**Vedline Ochieng**

Civil Engineering Student | Machine Learning Enthusiast | Python Developer

GitHub: [@Vedline2547](https://github.com/Vedline2547)

LinkedIn: [Vedline Ochieng](https://www.linkedin.com/in/vedline-ochieng-025146290/)

---

*Exploring machine learning applications through practical, data-driven projects.*
