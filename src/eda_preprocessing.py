import pandas as pd
import matplotlib.pyplot as plt

class EDADataPreprocessor:
    """
    Basic data cleaning pipeline for insurance datasets, designed to prepare the
    data for exploratory analysis (EDA). Includes type fixing, removal of bad rows,
    imputation of missing values, dropping high-missing columns, duplicate removal,
    and simple quality assessment with missing value visualization.
    """

    def __init__(self, file_path: str, delimiter: str = "|"):
        """
        Initialize the EDADataPreprocessor.

        Parameters
        ----------
        file_path : str
            Path to the raw insurance dataset.
        delimiter : str, default '|'
            Delimiter used in the input file.
        """
        self.file_path = file_path
        self.delimiter = delimiter
        self.df: pd.DataFrame | None = None

    # -----------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset into memory.

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame.
        """
        self.df = pd.read_csv(self.file_path, delimiter=self.delimiter, low_memory=False)
        print(f"Loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df

    # -----------------------------------------------------
    def convert_types(self) -> pd.DataFrame:
        """
        Convert specific columns to appropriate data types (datetime).

        Columns handled:
        - TransactionMonth
        - VehicleIntroDate

        Returns
        -------
        pd.DataFrame
            DataFrame with converted types.
        """
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

    # -----------------------------------------------------
    def drop_high_missing(self, threshold: float = 0.9) -> pd.DataFrame:
        """
        Drop columns with missing values above a specified threshold.

        Parameters
        ----------
        threshold : float, default 0.9
            Fraction of allowed missing values. Columns exceeding this are dropped.

        Returns
        -------
        pd.DataFrame
            DataFrame with high-missing columns removed.
        """
        missing_fraction = self.df.isna().mean()
        to_drop = missing_fraction[missing_fraction > threshold].index

        if len(to_drop) > 0:
            self.df = self.df.drop(columns=to_drop)
            print(f"Dropped {len(to_drop)} high-missing columns (> {threshold * 100}%).")

        return self.df

    # -----------------------------------------------------
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Fill missing values for domain-specific categorical flags.

        Flags handled: 'WrittenOff', 'Rebuilt', 'Converted'
        Missing values are filled with 'No'.

        Returns
        -------
        pd.DataFrame
            DataFrame with missing values handled.
        """
        if self.df is None:
            print("No data loaded yet.")
            return

        risk_flags = ['WrittenOff', 'Rebuilt', 'Converted']
        for col in risk_flags:
            if col in self.df:
                self.df[col] = self.df[col].fillna("No")

        return self.df

    # -----------------------------------------------------
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.

        Returns
        -------
        pd.DataFrame
            DataFrame with duplicates removed.
        """
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = before - len(self.df)

        if removed > 0:
            print(f"Removed {removed} duplicates.")

        return self.df

    # -----------------------------------------------------
    def visualize_missing(self, title: str = "Missing Values"):
        """
        Visualize missing values as a horizontal bar chart.

        Parameters
        ----------
        title : str, default 'Missing Values'
            Title for the plot.
        """
        missing_pct = (self.df.isna().mean() * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]

        if missing_pct.empty:
            print("No missing values to visualize.")
            return

        plt.figure(figsize=(10, max(4, len(missing_pct) * 0.3)))
        missing_pct.plot(kind="barh", color="salmon")
        plt.title(title)
        plt.xlabel("Missing %")
        plt.gca().invert_yaxis()
        plt.show()

    # -----------------------------------------------------
    def assess_quality(self) -> pd.DataFrame:
        """
        Assess the dataset quality by printing dtypes and returning a sorted
        missing values summary.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'Missing' and 'Missing %', sorted descending by 'Missing %'.
        """
        print("\nFinal Data Assessment:")
        print(self.df.dtypes)

        missing = self.df.isna().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        qa = pd.DataFrame({"Missing": missing, "Missing %": missing_pct})
        qa = qa.sort_values(by="Missing %", ascending=False)
        return qa

    # -----------------------------------------------------
    
    def save_cleaned(self, output_path: str, delimiter="|"):
        """
        Save the cleaned dataset to a CSV file.

        Parameters
        ----------
        output_path : str
            File path to save the cleaned dataset.
        """
        if self.df is None:
            print("No data to save.")
            return
        self.df.to_csv(output_path, sep=delimiter, index=False)
        print(f"Cleaned data saved to '{output_path}'.")

    # -----------------------------------------------------
    def process(self) -> pd.DataFrame:
        """
        Execute the full cleaning pipeline:
        1. Load data
        2. Visualize initial missing values
        3. Convert types
        4. Drop high-missing columns
        5. Handle domain-specific missing values
        6. Remove duplicates
        7. Assess final quality

        Returns
        -------
        pd.DataFrame
            Fully cleaned dataset ready for EDA.
        """
        self.load_data()

        print("\n--- Initial Missing Values ---")
        print(self.df.isnull().sum())
        self.visualize_missing("Missing Values Before Cleaning")

        # Cleaning steps
        self.convert_types()
        self.drop_high_missing()
        self.handle_missing_values()
        self.remove_duplicates()

        print("\n--- Final Quality Assessment ---")
        qa = self.assess_quality()
        print(qa)

        print("\nData cleaning completed.")
        return self.df
