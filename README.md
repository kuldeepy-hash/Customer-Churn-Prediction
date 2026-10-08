# Customer Churn Prediction Using Machine Learning

## Project Overview

This project analyzes customer churn using Python and Machine Learning to identify patterns associated with customers leaving a telecommunications service provider.

The project includes data cleaning, exploratory data analysis (EDA), data preprocessing, classification model development, performance evaluation, cross-validation, and feature coefficient analysis.

## Dataset Information

**Dataset:** IBM Telco Customer Churn

| Description | Value |
|---|---|
| Original Records | 7,043 |
| Cleaned Records | 7,032 |
| Original Columns | 21 |
| Input Features | 19 |
| Target Variable | Churn |
| Customers Who Stayed | 5,163 |
| Customers Who Churned | 1,869 |
| Overall Churn Rate | 26.58% |

### Data Cleaning

- Converted `TotalCharges` from object to numeric datatype.
- Identified 11 missing values in `TotalCharges`.
- Removed 11 records containing missing `TotalCharges`.
- Verified zero duplicate records and duplicate customer IDs.
- Confirmed zero remaining missing values after cleaning.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook
- Visual Studio Code

## Exploratory Data Analysis

### 1. Customer Churn Distribution

- Customers who stayed: 5,163 (73.42%)
- Customers who churned: 1,869 (26.58%)

### 2. Contract Type vs Churn

| Contract Type | Churn Rate |
|---|---:|
| Month-to-month | 42.71% |
| One year | 11.28% |
| Two year | 2.85% |

Month-to-month customers showed the highest observed churn rate.

### 3. Monthly Charges vs Churn

| Customer Group | Average Monthly Charges |
|---|---:|
| Stayed | 61.31 |
| Churned | 74.44 |

Churned customers had higher average monthly charges in the dataset.

### 4. Customer Tenure vs Churn

| Customer Group | Average Tenure |
|---|---:|
| Stayed | 37.65 months |
| Churned | 17.98 months |

Customers who churned had shorter average tenure than customers who stayed.

## Machine Learning Methodology

1. Selected 19 input features and the binary churn target.
2. Removed the customer identifier from model inputs.
3. Applied an 80:20 stratified train-test split.
4. Used One-Hot Encoding for categorical variables.
5. Applied StandardScaler to numerical variables.
6. Implemented preprocessing within Scikit-learn pipelines.
7. Trained Logistic Regression and Random Forest classifiers.
8. Evaluated classification performance using multiple metrics.
9. Performed five-fold stratified cross-validation on training data.
10. Analyzed Logistic Regression coefficients.

## Machine Learning Results

### Five-Fold Cross-Validation

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 80.25% | 79.63% |
| Precision | 65.41% | 65.44% |
| Recall | 54.65% | 49.50% |
| F1 Score | 59.53% | 56.34% |
| ROC-AUC | 84.61% | 82.51% |

**Selected Model: Logistic Regression**

Logistic Regression achieved the highest cross-validation F1 Score and ROC-AUC among the evaluated models.

### Logistic Regression Test Performance

| Metric | Score |
|---|---:|
| Accuracy | 80.38% |
| Precision | 64.85% |
| Recall | 57.22% |
| F1 Score | 60.80% |
| ROC-AUC | 83.59% |

### Confusion Matrix

| Result | Customers |
|---|---:|
| True Negatives | 917 |
| False Positives | 116 |
| False Negatives | 160 |
| True Positives | 214 |

The model correctly identified 214 of the 374 customers who churned in the test dataset.

## Feature Coefficient Analysis

The Logistic Regression model identified associations involving:

- Contract type
- Customer tenure
- Internet service type
- Total charges
- Monthly charges

These coefficients describe model associations and should not be interpreted as proof of causation.

## Project Files

- `Customer_Churn_Prediction.ipynb` — Complete analysis and machine learning notebook
- `Telco-Customer-Churn.csv` — Original dataset
- `customer_churn_model.pkl` — Saved Logistic Regression pipeline
- `model_comparison_results.csv` — Cross-validation results
- `requirements.txt` — Python dependencies
- `README.md` — Project documentation

## How to Run

1. Clone or download this repository.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Open `Customer_Churn_Prediction.ipynb` in VS Code or Jupyter Notebook.
4. Run the notebook cells in order.

## Limitations

- The dataset is a sample telecommunications churn dataset.
- Class imbalance can affect classification performance.
- Model coefficients indicate associations, not causal relationships.
- The model requires further validation before real-world deployment.

## Conclusion

This project demonstrates an end-to-end customer churn analysis and prediction workflow using Python and Machine Learning.

Logistic Regression achieved the strongest overall performance among the two evaluated models, with 80.38% test accuracy and 83.59% test ROC-AUC.

The analysis highlights customer contract type, tenure, and monthly charges as useful factors to investigate when developing customer retention strategies.