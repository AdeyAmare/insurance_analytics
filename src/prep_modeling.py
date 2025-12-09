import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataPrepHelper:
    """
    Helper class for data preparation tasks for Task 4 modeling, including:
    - Feature selection
    - Handling missing values
    - Categorical encoding
    - Feature engineering
    - Train-test splitting for regression and classification tasks
    """

    def __init__(self, current_year=2015):
        """
        Initialize the DataPrepHelper.

        Parameters
        ----------
        current_year : int, default=2015
            The current year used for calculating vehicle age.
        """
        self.current_year = current_year
        self.encoders = {}

    # ----------------------------
    # Feature selection
    # ----------------------------

    @staticmethod
    def select_features(df):
        """
        Drop unnecessary columns like IDs, targets, or high-missing columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.

        Returns
        -------
        pd.DataFrame
            DataFrame with selected features.
        """
        drop_cols = [
            "UnderwrittenCoverID",
            "PolicyID",
            "TransactionMonth",
            "NumberOfVehiclesInFleet",
            "CrossBorder",
            "CustomValueEstimate",
            "WrittenOff",
            "Rebuilt",
            "Converted",
        ]
        return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # ----------------------------
    # Missing value handling
    # ----------------------------

    @staticmethod
    def handle_missing_values(df):
        """
        Fill missing values: numeric → median, categorical → mode/Unknown.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.

        Returns
        -------
        pd.DataFrame
            DataFrame with missing values handled.
        """
        df = df.copy()
        for col in df.columns:
            if df[col].isna().sum() == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode()
                df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "Unknown")
        return df

    def handle_outliers_iqr(df):
        """
        Handle outliers using the IQR method.
        Numeric columns are winsorized to the IQR boundaries.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.

        Returns
        -------
        pd.DataFrame
            DataFrame with outliers capped.
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:
                continue  # Skip constant columns

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            df[col] = np.where(df[col] < lower, lower,
                            np.where(df[col] > upper, upper, df[col]))

        return df
    # ----------------------------
    # Categorical encoding
    # ----------------------------

    def encode_categoricals(self, df, exclude=None):
        """
        Label-encode categorical columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        exclude : list of str, optional
            Columns to exclude from encoding.

        Returns
        -------
        df_encoded : pd.DataFrame
            DataFrame with categorical columns encoded.
        encoders : dict
            Dictionary of LabelEncoders for each encoded column.
        """
        df = df.copy()
        exclude = exclude or []
        self.encoders = {}

        for col in df.select_dtypes(include=["object", "bool"]).columns:
            if col in exclude:
                continue
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le

        return df, self.encoders

    # ----------------------------
    # Feature engineering
    # ----------------------------

    def create_features(self, df):
        """
        Add engineered features:
        - vehicle_age
        - premium_per_sum_insured

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.

        Returns
        -------
        pd.DataFrame
            DataFrame with new features added.
        """
        df = df.copy()

        if "RegistrationYear" in df.columns:
            df["vehicle_age"] = self.current_year - df["RegistrationYear"]

        if "TotalPremium" in df.columns and "SumInsured" in df.columns:
            df["premium_per_sum_insured"] = np.where(
                df["SumInsured"] > 0,
                df["TotalPremium"] / df["SumInsured"],
                0,
            )

        return df

    # ----------------------------
    # Train-test split
    # ----------------------------

    @staticmethod
    def prepare_severity_data(df, test_size=0.2, random_state=42):
        """
        Prepare data for severity (regression) model: use only rows with claims > 0.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        test_size : float
            Fraction of data to use for testing.
        random_state : int
            Random seed.

        Returns
        -------
        X_train, X_test, y_train, y_test : tuple
            Train-test split of features and target.
        """
        df = df[df["TotalClaims"] > 0].copy()
        target = "TotalClaims"
        features = [c for c in df.columns if c not in [target, "TotalPremium", "has_claim", "margin"]]
        X = df[features]
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    @staticmethod
    def prepare_classification_data(df, test_size=0.2, random_state=42):
        """
        Prepare data for classification model: target = has_claim.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        test_size : float
            Fraction of data to use for testing.
        random_state : int
            Random seed.

        Returns
        -------
        X_train, X_test, y_train, y_test : tuple
            Train-test split of features and target.
        """
        df = df.copy()
        if "has_claim" not in df.columns:
            df["has_claim"] = (df["TotalClaims"] > 0).astype(int)

        features = [c for c in df.columns if c not in ["has_claim", "TotalClaims", "TotalPremium", "margin"]]
        X = df[features]
        y = df["has_claim"]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
