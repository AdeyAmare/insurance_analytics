import pandas as pd
import matplotlib.pyplot as plt

class DataCleaner:
    """Basic data cleaning pipeline for insurance data with informative prints."""

    def __init__(self, file_path, delimiter="|"):
        """Initialize the cleaner with file path and delimiter."""
        self.file_path = file_path
        self.delimiter = delimiter
        self.df = None

    def load_data(self):
        print(f"Loading data from '{self.file_path}'...")
        self.df = pd.read_csv(self.file_path, delimiter=self.delimiter, low_memory=False)
        print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns.")
        return self.df

    def check_missing(self):
        if self.df is None:
            print("No data loaded yet.")
            return
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({"Missing": missing, "Missing %": missing_pct}).sort_values("Missing %", ascending=False)
        print(f"Columns with missing data: {missing_df[missing_df['Missing']>0].shape[0]}")
        return missing_df

    def data_conversion(self):
        if self.df is None:
            print("No data loaded yet.")
            return
        date_cols = {
            "TransactionMonth": None,
            "VehicleIntroDate": "%m/%Y"
        }
        for col, fmt in date_cols.items():
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce", format=fmt)
                print(f"Converted '{col}' to datetime.")
            else:
                print(f"Column '{col}' not found, skipped conversion.")
        return self.df

    def drop_useless_columns(self):
        if self.df is None:
            print("No data loaded yet.")
            return
        threshold = 0.9
        high_missing = self.df.columns[self.df.isna().mean() > threshold]
        if len(high_missing) > 0:
            self.df.drop(columns=high_missing, inplace=True)
            print(f"Dropped {len(high_missing)} columns with >{threshold*100}% missing values: {list(high_missing)}")
        else:
            print("No columns exceeded missing threshold.")
        return self.df

    def visualize_missing(self):
        if self.df is None:
            print("No data loaded yet.")
            return
        missing_pct = (self.df.isnull().mean() * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]
        if not missing_pct.empty:
            print("Visualizing missing values...")
            missing_pct.plot(kind="barh", color="pink")
            plt.xlabel("Missing %")
            plt.show()
        else:
            print("No missing values to visualize.")

    def assess_quality(self):
        if self.df is None:
            print("No data loaded yet.")
            return
        print("Final Data Assessment:")
        print(self.df.dtypes)
        return self.check_missing()
    
    def save_cleaned_txt(self, output_path, delimiter="|"):
        """Save the cleaned DataFrame to a TXT file with a delimiter."""
        if self.df is None:
            print("No data to save.")
            return
        self.df.to_csv(output_path, sep=delimiter, index=False)
        print(f"Cleaned data saved to '{output_path}'.")


    def process(self):
        """Run the full cleaning pipeline in order."""
        self.load_data()
        print("\n--- Initial Missing Values ---")
        print(self.check_missing())

        self.visualize_missing()

        self.data_conversion()
        self.drop_useless_columns()

        print("\n--- Final Data Assessment ---")
        print(self.assess_quality())

        
        print("Data cleaning completed.")
        return self.df
