# `src/` Folder

The `src` directory contains three main modules used for cleaning, analyzing, and visualizing insurance data.

---

## 1. `eda_preprocessing.py`

### Purpose
- Handles the full data cleaning pipeline.

### Features
- Load raw TXT/CSV files
- Missing value detection
- Missing value visualization
- Date conversions
- Drop columns with >90% missing
- Save cleaned output

### Main Class
- `EDADataPreprocessor`

---

## 2. `eda.py`

### Purpose
- Provides text-based exploratory data analysis (no visual plots).

### Features
- Numeric & categorical summaries
- Loss ratio calculations
- Outlier detection (IQR)
- Monthly trends
- Correlations
- Geographic summaries
- Vehicle claims insights
- Full EDA pipeline via `run()`

### Main Class
- `InsuranceEDA`

---

## 3. `eda_visualizations.py`

### Purpose
- Extends the EDA engine with visualizations.

### Features
- Histograms (numeric)
- Bar plots (categorical)
- Scatter + correlation heatmap
- Loss ratio plots
- Temporal trends
- Outlier boxplots
- Top/bottom vehicle claim plots

### Main Class
- `InsuranceEDAVisualizer`

## 4. `hypothesis_tests.py`

### Purpose
- Provides statistical hypothesis testing helpers.

### Features
- Chi-squared test for categorical variables

- Two-sample t-test for numerical variables across two groups

- One-way ANOVA for numerical variables across multiple groups

### Main Classes
- `HypothesisTester`
- `HypothesisTester.TestResult` – container for test results with statistic, p-value, and null hypothesis decision


## 5. `prep_modeling.py`

### Purpose
- Data preparation for predictive modeling..

### Features
- Feature selection

- Handling missing values

- Categorical encoding

- Feature engineering (vehicle_age, premium_per_sum_insured)

- Train-test splitting for regression and classification

### Main Classes
- `DataPrepHelper`


## 6. `modeling.py`

### Purpose
- Training, evaluating, and analyzing predictive models for regression and classification..

### Features
- Linear Regression, Logistic Regression

- Random Forest Regressor & Classifier

- Optional XGBoost Regressor & Classifier

- Model evaluation metrics:

    - Regression: RMSE, R²

    - Classification: Accuracy, Precision, Recall, F1

    - Feature importance extraction for tree-based models

### Main Classes
- `ModelHelper`