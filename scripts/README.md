# `scripts/` Folder

The `scripts` directory contains utility scripts for running the data cleaning and analysis pipelines on the insurance dataset.

---

## `run_cleaner.py`

### Purpose
- Run the `DataCleaner` pipeline to clean raw insurance data and save the cleaned output.

### Features
- Adds the project root to `sys.path` to allow importing `src` modules.
- Loads raw data from `data/MachineLearningRating_v3.txt`.
- Runs the full cleaning pipeline.
- Saves cleaned data to `data/MachineLearningRating_v3_cleaned.txt`.

### Usage
```bash
python scripts/run_cleaner.py
