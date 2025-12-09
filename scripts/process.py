import sys
import os
import pandas as pd

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.eda_preprocessing import EDADataPreprocessor
from src.prep_modeling import DataPrepHelper

def main(input_path, output_path, current_year=2015):
    """
    Full data preprocessing pipeline:
    1. EDA cleaning
    2. Feature selection
    3. Missing value handling
    4. Outlier handling
    5. Categorical encoding
    6. Feature engineering
    """
    
    # --------------------------------------
    # Step 1: EDA cleaning
    # --------------------------------------
    cleaner = EDADataPreprocessor(input_path)
    cleaner.process()

    # FIX: call the method, not reference it
    df = cleaner.load_data()

    # --------------------------------------
    # Step 2: Modeling prep
    # --------------------------------------
    prep = DataPrepHelper(current_year=current_year)

    df = prep.select_features(df)
    df = prep.handle_missing_values(df)
    df, _ = prep.encode_categoricals(df)
    df = prep.create_features(df)

    # --------------------------------------
    # Step 3: Save output
    # --------------------------------------
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Final preprocessed dataset saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Full preprocessing pipeline")
    parser.add_argument("--input", type=str, required=True, help="Raw data path")
    parser.add_argument("--output", type=str, required=True, help="Processed data path")
    parser.add_argument("--current_year", type=int, default=2015)

    args = parser.parse_args()
    main(args.input, args.output, args.current_year)
