# `src/` Folder

The `src` directory contains three main modules used for cleaning, analyzing, and visualizing insurance data.

---

## 1. `cleaner.py`

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
- `DataCleaner`

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
