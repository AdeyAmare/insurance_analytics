import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataCleaner:
    """
    Handles data loading, assessment, controlled cleaning,
    and safe type conversions for insurance EDA.
    """

    def __init__(self, file_path, delimiter=","):
        self.file_path = file_path
        self.delimiter = delimiter
        self.df = None

    def load_data(self):
        """Load CSV or TXT file into a DataFrame."""
        print(f"Loading data from: {self.file_path}")
        try:
            self.df = pd.read_csv(self.file_path, delimiter=self.delimiter, low_memory=False)
            print("Data loaded successfully.\n")
            
            print("--- Sample Data ---")
            print(self.df.head())
            print(f"\nData Shape: {self.df.shape}")
            
            return self.df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    def initial_assessment(self):
        """Print structure, missing values, and duplicates before cleaning."""
        if self.df is None:
            print("Load data first.")
            return
        print("\n--- INITIAL DATA ASSESSMENT ---")
        print(f"\nRows: {len(self.df)}, Columns: {len(self.df.columns)}")
        dup = self.df.duplicated().sum()
        print(f"Duplicate Rows: {dup}")
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df) * 100).round(2)
        info = pd.DataFrame({
            "Missing Count": missing,
            "Missing %": missing_percent,
            "Dtype": self.df.dtypes.astype(str)
        })
        print("\nColumns with Missing Values:")
        print(info[info["Missing Count"] > 0].sort_values("Missing %", ascending=False))

    def clean_data(self):
        """Clean duplicates, high-missing-value columns, poor rows, and text."""
        if self.df is None:
            print("Load data first.")
            return
        print("\n--- CLEANING PROCESS STARTED ---")

        # 1. Remove Duplicate Rows
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"- Removed {before - len(self.df)} duplicate rows.")

        # 2. Drop Columns With >50% Missing Values
        missing_pct = self.df.isnull().mean() * 100
        high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
        print(f"- Columns >50% missing (dropping): {high_missing_cols}")
        self.df = self.df.drop(columns=high_missing_cols, errors="ignore")

        # 3. Drop Rows Missing Too Many Values (require at least 40% non-null)
        min_required = int(0.4 * len(self.df.columns))
        before_rows = len(self.df)
        self.df = self.df.dropna(thresh=min_required)
        print(f"- Dropped {before_rows - len(self.df)} rows with <40% usable data.")

        print("--- CLEANING COMPLETE ---")
        return self.df

    def data_conversion(self):
        """Convert dates, numerics, and mixed-type columns safely."""
        if self.df is None:
            print("Load data first.")
            return
        print("\n--- DATA TYPE CONVERSION ---")

        # 1. Date Conversions
        date_cols = {
            "TransactionMonth": None,
            "VehicleIntroDate": "%m/%Y"
        }
        for col, fmt in date_cols.items():
            if col in self.df.columns:
                if fmt:
                    self.df[col] = pd.to_datetime(self.df[col], errors="coerce", format=fmt).dt.to_period("M")
                else:
                    self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

        # 2. Numeric Conversions
        numeric_candidate_cols = [
            "RegistrationYear",
            "CapitalOutstanding",
            "TotalPremium",
            "TotalClaims",
            "SumInsured",
            "Cubiccapacity",
            "Kilowatts",
            "NumberOfDoors"
        ]
        for col in numeric_candidate_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        if "RegistrationYear" in self.df.columns:
            self.df["RegistrationYear"] = self.df["RegistrationYear"].astype("Int64")

        print(self.df.dtypes)

        print("- Completed numeric + date conversions.")
        return self.df

    def assess_quality(self):
        """Show final dtypes and remaining missing values."""
        if self.df is None:
            print("Load data first.")
            return
        print("\n--- FINAL POST-CLEANING ASSESSMENT ---")
        print("\nColumn Data Types:")
        print(self.df.dtypes)
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        miss = pd.DataFrame({
            "Missing Count": missing,
            "Missing %": missing_pct
        })
        print("\nRemaining Missing Values:")
        print(miss[miss["Missing Count"] > 0].sort_values("Missing %", ascending=False))
        return self.df
    
    def visualize_data_quality(self):
        """Visualizes data quality: missing values, duplicates, and data types."""
        if self.df is None:
            print("Load data first.")
            return
        
        # --- 1. Missing Values Percentage Bar Plot ---
        missing_pct = (self.df.isnull().mean() * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]  # Only columns with missing
        if not missing_pct.empty:
            plt.figure(figsize=(10,6))
            missing_pct.plot(kind='barh', color='pink')
            plt.xlabel("Missing %")
            plt.ylabel("Columns")
            plt.title("Percentage of Missing Values by Column")
            plt.show()
        else:
            print("No missing values to visualize.")

    def process(self):
        """Executes the complete data cleaning pipeline."""
        print("\n================= PIPELINE STARTED =================")
        self.load_data()
        self.initial_assessment()
        self.visualize_data_quality()
        self.data_conversion()
        self.clean_data()
        self.assess_quality()
        print("\n================= PIPELINE COMPLETED ================")
        return self.df
