# End-to-End Insurance Risk Analytics & Predictive Modeling


Dive into real insurance data to uncover low-risk segments and build smart models that optimize premiums.

---

## Project Overview

1. **Data Loading:** Import raw insurance datasets (TXT/CSV) into memory.

2. **Data Cleaning:** Detect and handle missing values, convert date columns, drop columns with high missing rates, and save cleaned output..  

3. **Exploratory Data Analysis (EDA):**  
   - Generate text-based summaries for numeric and categorical features.
   - Detect outliers, correlations, temporal trends, geographic patterns, and vehicle claim insights.  

4. **Visualizations:** Produce histograms, bar plots, scatter plots, correlation heatmaps, trend plots, and outlier detection plots.

4. **Hypothesis Testing:** tatistically validate business hypotheses using chi-squared, t-tests, and ANOVA.

4. **Predictive Modeling:** Build regression and classification models to predict claim severity, claim occurrence, and optimize premiums. Feature importance and interpretability analyses are included using SHAP.
---

## Folder Structure

- `data/`:  Raw and cleaned insurance datasets (DVC-tracked).  
- `notebooks/`: Jupyter notebooks for interactive analysis
- `scripts/`:Utility scripts for running pipelines
- `src/`: Core Python modules.  
- `dvc.yaml`: DVC pipeline definition.  
- `dvc.lock`: Locked pipeline versions.

---

## Source Modules (src/)

The `src/` directory contains all core modules that power the data cleaning, analysis, hypothesis testing, and modeling workflows of the project. 

The `eda_preprocessing.py` module implements the full data cleaning pipeline through its main class, `EDADataPreprocessor`, which loads TXT/CSV files, detects and visualizes missing values, converts date fields, drops columns with excessive missingness, and saves cleaned outputs.

The `eda.py` module provides text-based exploratory data analysis via the `InsuranceEDA` class. It generates numeric and categorical summaries, computes loss ratios, detects outliers, analyzes monthly and geographic trends, explores correlations, and provides insights into vehicle-related claims. To complement this, `eda_visualizations`.py extends the EDA engine using the `InsuranceEDAVisualizer` class, which produces histograms, bar charts, scatter plots, correlation heatmaps, loss ratio plots, temporal trend visualizations, boxplots for outlier detection, and top/bottom vehicle claim charts.

For statistical analysis, the `hypothesis_tests.py` module introduces the `HypothesisTester` class and its `TestResult` container to run chi-squared tests for categorical independence, two-sample t-tests for group comparisons, and one-way ANOVA for multi-group numerical comparisons. These tools help validate hypotheses and support data-driven conclusions.

The modeling workflow begins with `prep_modeling.py`, where the `DataPrepHelper` class handles modeling-focused data preparation, including feature selection, missing value handling, categorical encoding, and engineering features such as vehicle age and premium-per-sum-insured. 

It also performs train–test splitting for both regression and classification tasks. The `modeling.py` module then provides the `ModelHelper` class, which trains and evaluates predictive models—such as Linear Regression, Logistic Regression, Random Forests, and optionally XGBoost—using performance metrics like RMSE, R², accuracy, precision, recall, and F1. It also supports extracting and analyzing feature importance.

---

## Notebooks 

The `notebooks/` directory contains interactive Jupyter notebooks that guide users through the workflow. 

The `eda_preprocessing.ipynb` notebook demonstrates data loading and cleaning with `EDADataPreprocessor`, while `eda_pipeline.ipynb` walks through the full EDA process using `InsuranceEDA` and `InsuranceEDAVisualizer`. 

The `hypothesis_testing.ipynb` notebook applies the `HypothesisTester` to real business questions, and `modeling.ipynb` covers end-to-end model building, evaluation, feature importance exploration, and optional SHAP-based explainability.

---


## How to Use 

1. Clone the repo

```bash
git clone https://github.com/AdeyAmare/insurance_analytics.git
cd insurance_analytics
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run cleaning & EDA:
```bash
python src/eda_preprocessing.py
python src/eda.py
python src/eda_visualizations.py
```

4. Perform hypothesis tests:
```bash
python src/hypothesis_tests.py
```
5. Prepare data and run predictive models::
```bash
python src/prep_modeling.py
python src/modeling.py
```
Or use notebooks for interactive exploration.