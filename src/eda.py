import pandas as pd

class InsuranceEDA:
    """Text-based, comprehensive EDA for insurance datasets without visualizations."""

    def __init__(self, df: pd.DataFrame):
        """Initialize the EDA processor."""
        self.df = df.copy()
        self.categorical_cols = self.df.select_dtypes(include="object").columns.tolist()
        self.numeric_cols = self.df.select_dtypes(include="number").columns.tolist()

        # Remove 'Unknown' in categorical columns
        for col in self.categorical_cols:
            self.df = self.df[~self.df[col].isin(["Unknown", "unknown"])]

    # ----------------------------
    # Descriptive Statistics
    # ----------------------------
    def numeric_summary(self) -> pd.DataFrame:
        """Summary statistics for numeric columns with coefficient of variation (CV)."""
        stats = self.df[self.numeric_cols].describe().T
        stats["CV"] = stats["std"] / stats["mean"]
        return stats

    def categorical_summary(self, top_n: int = 10) -> dict:
        """Top value counts for categorical columns."""
        freq_dict = {col: self.df[col].value_counts().head(top_n) for col in self.categorical_cols}
        return freq_dict

    # ----------------------------
    # Loss Ratios
    # ----------------------------
    def overall_loss_ratio(self) -> float:
        """Overall portfolio loss ratio."""
        if "TotalClaims" in self.df.columns and "TotalPremium" in self.df.columns:
            return self.df["TotalClaims"].sum() / self.df["TotalPremium"].sum()
        return None

    def grouped_loss_ratio(self, group_col: str) -> pd.DataFrame:
        """Loss ratio by a categorical column."""
        if group_col in self.df.columns and "TotalClaims" in self.df.columns and "TotalPremium" in self.df.columns:
            grouped = self.df.groupby(group_col)[["TotalClaims", "TotalPremium"]].sum()
            grouped["LossRatio"] = grouped["TotalClaims"] / grouped["TotalPremium"]
            return grouped.sort_values("LossRatio", ascending=False)
        return pd.DataFrame()

    # ----------------------------
    # Outlier Detection
    # ----------------------------
    def detect_outliers(self, columns=None) -> dict:
        """Detect outliers in numeric columns using IQR method."""
        if columns is None:
            columns = ["TotalClaims", "CustomValueEstimate"]
        outliers = {}
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                mask = (self.df[col] < Q1 - 1.5 * IQR) | (self.df[col] > Q3 + 1.5 * IQR)
                outliers[col] = self.df[col][mask]
        return outliers

    # ----------------------------
    # Temporal Trends
    # ----------------------------
    def monthly_trends(self) -> pd.DataFrame:
        """Monthly aggregation of TotalPremium, TotalClaims, PolicyCount, ClaimFrequency, ClaimSeverity."""
        if "TransactionMonth" in self.df.columns:
            monthly = (
                self.df.set_index("TransactionMonth")
                .resample("M")
                .agg({"TotalPremium": "sum", "TotalClaims": "sum", "PolicyID": "count"})
            )
            monthly.rename(columns={"PolicyID": "PolicyCount"}, inplace=True)
            monthly["ClaimFrequency"] = monthly["TotalClaims"] / monthly["PolicyCount"]
            monthly["ClaimSeverity"] = monthly["TotalClaims"] / monthly["TotalPremium"]
            return monthly
        return pd.DataFrame()

    # ----------------------------
    # Correlations
    # ----------------------------
    def correlations(self) -> pd.DataFrame:
        """Correlation matrix for numeric columns."""
        return self.df[self.numeric_cols].corr()

    def zip_code_correlations(self, zip_col="PostalCode") -> pd.DataFrame:
        """Aggregate TotalPremium and TotalClaims by ZipCode."""
        if zip_col in self.df.columns and "TotalPremium" in self.df.columns and "TotalClaims" in self.df.columns:
            return self.df.groupby(zip_col)[["TotalPremium", "TotalClaims"]].sum()
        return pd.DataFrame()

    # ----------------------------
    # Geographic Trends
    # ----------------------------
    def geographic_trends(self, group_cols) -> dict:
        """Aggregate TotalPremium, TotalClaims, and PolicyCount by geographic columns."""
        summaries = {}
        for col in group_cols:
            if col in self.df.columns:
                summary = self.df.groupby(col).agg({
                    "TotalPremium": "sum",
                    "TotalClaims": "sum",
                    "PolicyID": "count"
                }).rename(columns={"PolicyID": "PolicyCount"})
                summaries[col] = summary
        return summaries

    # ----------------------------
    # Top/Bottom Vehicle Claims
    # ----------------------------
    def vehicle_claims_summary(self, top_n=5) -> dict:
        """Identify vehicle makes/models with highest and lowest average TotalClaims."""
        results = {}
        for col in ["make", "Model"]:
            if col in self.df.columns:
                avg_claims = self.df.groupby(col)["TotalClaims"].mean().sort_values()
                results[col] = {
                    "lowest": avg_claims.head(top_n),
                    "highest": avg_claims.tail(top_n).sort_values(ascending=False)
                }
        return results

    # ----------------------------
    # Full EDA Pipeline
    # ----------------------------
    def run(self) -> dict:
        """Run full EDA and return results as a dictionary."""
        results = {
            "numeric_summary": self.numeric_summary(),
            "categorical_summary": self.categorical_summary(),
            "overall_loss_ratio": self.overall_loss_ratio(),
            "grouped_loss_ratio_province": self.grouped_loss_ratio("Province"),
            "grouped_loss_ratio_vehicle": self.grouped_loss_ratio("VehicleType"),
            "grouped_loss_ratio_gender": self.grouped_loss_ratio("Gender"),
            "outliers": self.detect_outliers(),
            "monthly_trends": self.monthly_trends(),
            "correlations": self.correlations(),
            "zip_code_aggregates": self.zip_code_correlations(),
            "geographic_trends": self.geographic_trends(["Province", "CoverType", "make"]),
            "vehicle_claims_summary": self.vehicle_claims_summary()
        }
        return results
