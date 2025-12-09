import sys
import os

# Add project root to path so we can import src modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.eda_preprocessing import EDADataPreprocessor

DATA_FILE_PATH = os.path.join(project_root, "data", "MachineLearningRating_v3.txt")
OUTPUT_PATH = os.path.join(project_root, "data", "MachineLearningRating_v3_cleaned.txt")

if __name__ == "__main__":
    cleaner = EDADataPreprocessor(DATA_FILE_PATH)
    cleaner.process()
    cleaner.save_cleaned(OUTPUT_PATH)
