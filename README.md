# Insurance Data Cleaning & EDA Pipeline

A reproducible data pipeline to clean and analyze insurance datasets.

---

## Project Pipeline Stages

1. **Data Loading:** Load raw insurance datasets (`TXT`/`CSV`) into memory.  

2. **Data Cleaning:** Handle missing values, convert date columns, and drop columns with high missing rates.  

3. **Exploratory Data Analysis (EDA):**  
   - Text-based summaries for numeric and categorical features.  
   - Detect outliers, correlations, trends, and vehicle claim patterns.  

4. **Visualizations:** Generate histograms, bar plots, scatter plots, heatmaps, and trend plots to complement EDA.

---

## Folder Structure

- `data/`: Raw and cleaned insurance data (DVC-tracked).  
- `notebooks/`: Interactive Jupyter notebooks for data assessment and EDA.  
- `scripts/`: Utility scripts to run the cleaning and analysis pipelines.  
- `src/`: Core modules for cleaning (`cleaner.py`), EDA (`eda.py`), and visualizations (`eda_visualizations.py`).  
- `dvc.yaml`: DVC pipeline definition.  
- `dvc.lock`: Locked pipeline versions.

---

## Scripts Overview

1. `scripts/run_cleaner.py`  
   - **Purpose:** Run the full data cleaning pipeline.  
   - **Effect:** Load raw data, clean it using `DataCleaner`, and save cleaned output.



