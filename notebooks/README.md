# `notebooks/` Folder

The `notebooks` directory contains Jupyter notebooks for interactive data exploration and analysis.

---

## `data_assessment.ipynb`

### Purpose
- Perform initial data loading, cleaning, and assessment using `DataCleaner` from `src/cleaner.py`.

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
