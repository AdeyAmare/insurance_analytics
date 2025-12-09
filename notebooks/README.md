# `notebooks/` Folder

The `notebooks` directory contains Jupyter notebooks for interactive data exploration and analysis.

---

## `eda_preprocessing.ipynb`

### Purpose
- Perform initial data loading, cleaning, and assessment using `EDADataPreprocessor` from `src/eda_preprocessing.py`.

### Features
- Load raw insurance data.
- Inspect missing values and basic statistics.
- Run the cleaning pipeline interactively.

---

## `eda_pipeline.ipynb`

### Purpose
- Perform exploratory data analysis (EDA) on the cleaned dataset using `InsuranceEDA` and `InsuranceEDAVisualizer`.

### Features
- Text-based numeric and categorical summaries (`eda.py`).
- Visualizations: histograms, bar plots, correlations, outliers (`eda_visualizations.py`).
- Run full EDA pipeline interactively.


## `hypothesis_testing.ipynb`

### Purpose
- Perform statistical hypothesis testing using the `HypothesisTester` class to validate relationships between variables and support analytical conclusions.

### Features

- Chi-Squared Tests to evaluate the independence of categorical variables.

- Two-Sample T-Tests for comparing numerical values across two groups (e.g., vehicles types, fuel types).

- ANOVA Tests for comparing numerical distributions across multiple categories.

- Clear interpretation of p-values and whether to reject the null hypothesis.

## `modeling.ipynb`

### Purpose
- Build, train, and evaluate predictive models for claim severity (regression) and claim occurrence (classification) using `ModelHelper` and `DataPrepHelper`.

### Features
- Data Preparation: feature selection, missing value handling, outlier capping, categorical encoding, engineered features, and train/test splitting.

- Regression Models: Linear Regression, Random Forest Regressor, optional XGBoost Regressor.

- Classification Models: Logistic Regression, Random Forest Classifier, optional XGBoost Classifier.

- Model Evaluation: RMSE, R², accuracy, precision, recall, and F1-score.

- Feature Importance: extraction and visualization for tree-based models.

- Optional SHAP analysis for model explainability.