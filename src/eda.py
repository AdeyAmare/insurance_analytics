import pandas as pd
import numpy as np

class EDAProcessor:
    """
    Performs comprehensive Exploratory Data Analysis (EDA) for an insurance portfolio.

    Attributes:
        df (pd.DataFrame): The cleaned DataFrame to analyze.
        financial_cols (list): List of numeric financial columns.
        categorical_cols (list): List of categorical columns (dtype object).
        stats (pd.DataFrame): Descriptive statistics of numeric columns.
        categorical_freqs (dict): Frequency tables of categorical columns.
        loss_ratios_data (dict): Overall and grouped loss ratios.
        temporal_trends_data (pd.DataFrame): Monthly aggregated trends.
        vehicle_claims_data (pd.Series): Average claims per vehicle make.
        correlations_data (pd.DataFrame): Correlation matrix of numeric financial columns.
        outliers_data (dict): Boolean masks for outlier detection.
        bivariate_summaries_data (dict): Bivariate summaries for selected groups.
        geographic_trends_data (dict): Aggregated metrics by geography.
    """

    def __init__(self, df):
        """Initializes the EDAProcessor with a DataFrame.

        Args:
            df (pd.DataFrame): Cleaned insurance portfolio data.
        """
        self.df = df.copy()

        # Remove ONLY "Unknown" and "unknown"
        unknowns = ["Unknown", "unknown"]
        self.categorical_cols = df.select_dtypes(include='object').columns.tolist()

        for col in self.categorical_cols:
            self.df = self.df[~self.df[col].isin(unknowns)]

        all_financial = [
            'TotalPremium', 'TotalClaims', 'CustomValueEstimate', 
            'SumInsured', 'CalculatedPremiumPerTerm'
        ]
        self.financial_cols = [col for col in all_financial if col in df.columns]

        self.stats = None
        self.categorical_freqs = None
        self.loss_ratios_data = {}
        self.temporal_trends_data = None
        self.vehicle_claims_data = None
        self.correlations_data = None
        self.outliers_data = None
        self.bivariate_summaries_data = {}
        self.geographic_trends_data = {}

    def descriptive_statistics(self):
        """Computes descriptive statistics for numeric financial columns.

        Returns:
            pd.DataFrame: Table with mean, std, min, max, quartiles, and CV.
        """
        stats = self.df[self.financial_cols].describe().T
        stats['CV (StdDev/Mean)'] = stats['std'] / stats['mean']
        print("\n--- Descriptive Statistics (Numeric Features) ---")
        print(stats)
        self.stats = stats
        return stats

    def categorical_frequencies(self, top_n=10):
        """
        Prints top N values for all categorical columns with counts and percentages.

        Args:
            top_n (int): Number of top values to display for each categorical column.

        Returns:
            dict: Dictionary mapping column name to value counts (pd.Series).
        """
        freq_dict = {}

        for col in self.categorical_cols:
            if col not in self.df.columns:
                continue
            counts = self.df[col].value_counts(dropna=False)
            freq_dict[col] = counts

            total = counts.sum()
            top_counts = counts.head(top_n)
            print(f"\n--- Top {top_n} for {col} ---")
            for value, count in top_counts.items():
                percent = count / total * 100
                print(value, count, f"{percent:.2f}%")

        self.categorical_freqs = freq_dict
        return freq_dict


    def calculate_loss_ratio(self, group_col=None):
        """Calculates loss ratio as TotalClaims / TotalPremium.

        Args:
            group_col (str, optional): Column to group by. Defaults to None.

        Returns:
            pd.DataFrame or float: Overall loss ratio or grouped DataFrame with LossRatio.
        """
        if group_col:
            if group_col not in self.df.columns:
                print(f"Warning: {group_col} not in dataframe → skipping loss ratio by this group.")
                return pd.DataFrame()
            grouped = self.df.groupby(group_col)[['TotalClaims', 'TotalPremium']].sum()
            grouped['LossRatio'] = grouped['TotalClaims'] / grouped['TotalPremium']
            grouped.sort_values('LossRatio', ascending=False, inplace=True)
            print(f"\n--- Loss Ratio by {group_col} ---")
            print(grouped)
            self.loss_ratios_data[group_col] = grouped
            return grouped
        else:
            overall_lr = self.df['TotalClaims'].sum() / self.df['TotalPremium'].sum()
            print(f"\n--- Overall Portfolio Loss Ratio: {overall_lr:.4f} ---")
            self.loss_ratios_data['overall'] = overall_lr
            return overall_lr

    def analyze_temporal_trends(self):
        """Analyzes monthly trends of premiums, claims, claim frequency, and severity.

        Returns:
            pd.DataFrame: Aggregated monthly trends.
        """
        if 'TransactionMonth' not in self.df.columns or not np.issubdtype(self.df['TransactionMonth'].dtype, np.datetime64):
            print("Error: 'TransactionMonth' must be a datetime column.")
            return pd.DataFrame()

        monthly = (self.df
                   .set_index('TransactionMonth')
                   .resample('M')
                   .agg({'TotalPremium':'sum', 'TotalClaims':'sum', 'PolicyID':'count'}))
        monthly.rename(columns={'PolicyID':'PolicyCount'}, inplace=True)
        monthly['ClaimFrequency'] = monthly['TotalClaims'] / monthly['PolicyCount']
        monthly['ClaimSeverity'] = monthly['TotalClaims'] / monthly['TotalPremium']
        print("\n--- Monthly Temporal Trends (First 5 Months) ---")
        print(monthly.head())
        self.temporal_trends_data = monthly
        return monthly

    def analyze_vehicle_claims(self):
        """Analyzes average TotalClaims per vehicle make.

        Returns:
            pd.Series: Average TotalClaims per Make.
        """
        if 'Make' not in self.df.columns and 'make' in self.df.columns:
            self.df['Make'] = self.df['make']

        if 'Make' not in self.df.columns:
            print("Warning: 'Make' column not found → skipping vehicle claims analysis.")
            return pd.Series()

        avg_claims = self.df.groupby('Make')['TotalClaims'].mean().sort_values(ascending=False).dropna()
        print("\n--- Top 5 Vehicle Makes by Average Claims ---")
        print(avg_claims.head(5))
        print("\n--- Bottom 5 Vehicle Makes by Average Claims ---")
        print(avg_claims.tail(5))
        self.vehicle_claims_data = avg_claims
        return avg_claims

    def calculate_correlations(self):
        """Calculates correlation matrix of numeric financial columns.

        Returns:
            pd.DataFrame: Correlation matrix.
        """
        if len(self.financial_cols) < 2:
            print("Warning: Not enough financial columns to calculate correlations.")
            return pd.DataFrame()

        corr_matrix = self.df[self.financial_cols].corr()
        print("\n--- Correlation Matrix (Financial Features) ---")
        print(corr_matrix)
        self.correlations_data = corr_matrix
        return corr_matrix

    def detect_outliers(self, threshold=3):
        """Detects outliers in numeric columns using z-score.

        Args:
            threshold (float, optional): Z-score threshold to classify as outlier. Defaults to 3.

        Returns:
            dict: Boolean masks where True indicates an outlier.
        """
        outliers = {}
        for col in ['TotalClaims', 'CustomValueEstimate']:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                mean = self.df[col].mean()
                std = self.df[col].std()
                z_scores = (self.df[col] - mean) / std
                mask = z_scores.abs() > threshold
                outliers[col] = mask
                print(f"\n--- Outliers in {col} (z-score > {threshold}) ---")
                print(self.df.loc[mask, [col]].head(10))
        self.outliers_data = outliers
        return outliers

    def bivariate_group_summary(self, group_col, x_col, y_col):
        """Aggregates x_col and y_col statistics by a grouping column.

        Args:
            group_col (str): Column to group by.
            x_col (str): Independent variable.
            y_col (str): Dependent variable.

        Returns:
            pd.DataFrame: Aggregated statistics.
        """
        if group_col not in self.df.columns:
            print(f"Warning: {group_col} not found → skipping bivariate summary.")
            return pd.DataFrame()

        summary = self.df.groupby(group_col).agg({x_col:'mean', y_col:['mean','sum','count']})
        print(f"\n--- Bivariate Summary: {y_col} vs {x_col} by {group_col} ---")
        print(summary.head(10))
        self.bivariate_summaries_data[(group_col, x_col, y_col)] = summary
        return summary

    def geographic_trends(self, group_col, metric_col):
        """Aggregates metric_col by geographic group.

        Args:
            group_col (str): Geographic column (e.g., Province, PostalCode).
            metric_col (str): Metric column to aggregate.

        Returns:
            pd.Series: Aggregated metric by geography.
        """
        if group_col not in self.df.columns:
            print(f"Warning: {group_col} not found → skipping geographic trend.")
            return pd.Series()

        summary = self.df.groupby(group_col)[metric_col].mean().sort_values(ascending=False)
        print(f"\n--- Geographic Trend: {metric_col} by {group_col} ---")
        print(summary.head(10))
        self.geographic_trends_data[(group_col, metric_col)] = summary
        return summary

    def run_eda(self):
        """Runs all EDA steps sequentially."""
        print("\n===== RUNNING FULL EDA PIPELINE =====")
        self.descriptive_statistics()
        self.categorical_frequencies()
        self.calculate_loss_ratio()
        for col in ['Province', 'VehicleType', 'Gender']:
            if col in self.df.columns:
                self.calculate_loss_ratio(group_col=col)
        self.analyze_temporal_trends()
        self.analyze_vehicle_claims()
        self.calculate_correlations()
        self.detect_outliers()
        for group_col in ['Province', 'PostalCode']:
            if group_col in self.df.columns:
                self.bivariate_group_summary(group_col, 'TotalPremium', 'TotalClaims')
                self.geographic_trends(group_col, 'TotalPremium')
        print("\n===== FULL EDA COMPLETE =====")
