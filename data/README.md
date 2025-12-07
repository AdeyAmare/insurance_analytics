# DVC Workflow for Data Cleaning

The data cleaning pipeline is managed using **DVC** (Data Version Control), ensuring reproducibility and versioning of processed datasets.

---

## `dvc.yaml`

### Purpose
- Defines pipeline stages and their dependencies/outputs.
- Tracks scripts, libraries, and raw data required for cleaning.

---

## `dvc.lock`

### Purpose
- Automatically generated to record exact versions and hashes of dependencies and outputs.
- Ensures reproducibility of the cleaned data.
- Updated whenever the pipeline is reproduced or outputs change.
