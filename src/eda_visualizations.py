import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.eda import InsuranceEDA

class InsuranceEDAVisualizer(InsuranceEDA):
    """
    Enhanced visualizations for insurance portfolio data.

    Inherits from InsuranceEDA. All parent methods use self.df.
    Visualizations can safely use self.df too.
    """

    def __init__(self, df: pd.DataFrame,
                 numeric_cols=None,
                 categorical_cols=None,
                 top_n=5,
                 hist_bins=30):
        """
        Initialize the visualizer.

        Parameters
        ----------
        df : pd.DataFrame
            The insurance dataset.
        numeric_cols : list, optional
            Numeric columns to visualize.
        categorical_cols : list, optional
            Categorical columns to visualize.
        top_n : int, optional
            Top N categories to display for categorical plots.
        hist_bins : int, optional
            Number of bins for histograms.
        """
        # Initialize parent, sets self.df
        super().__init__(df)
        
        # Make self.data an alias to self.df — now both point to the same DataFrame
        self.data = self.df

        # Visualization configs
        self.numeric_cols = numeric_cols or ["TotalPremium", "TotalClaims", "CustomValueEstimate"]
        self.categorical_cols = categorical_cols or ["Province", "VehicleType", "Gender", "make", "Model", "CoverType"]
        self.top_n = top_n
        self.hist_bins = hist_bins
        sns.set_theme(style="whitegrid")

    # ----------------------------
    # UNIVARIATE ANALYSIS
    # ----------------------------
    def univariate_analysis(self):
        """
        Plot univariate distributions for numeric and categorical columns.

        Numeric: boxplots + log-scaled histograms.
        Categorical: top N value counts.
        """
        # Numeric Boxplots
        plt.figure(figsize=(5 * len(self.numeric_cols), 4))
        for i, col in enumerate(self.numeric_cols):
            if col in self.data.columns:
                plt.subplot(1, len(self.numeric_cols), i + 1)
                sns.boxplot(y=self.data[col], color='skyblue')
                plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.show()

        # Log-scaled histograms
        plt.figure(figsize=(5 * len(self.numeric_cols), 4))
        for i, col in enumerate(self.numeric_cols):
            if col in self.data.columns:
                plt.subplot(1, len(self.numeric_cols), i + 1)
                sns.histplot(np.log1p(self.data[self.data[col] > 0][col]),
                             bins=self.hist_bins, kde=True, color='teal')
                plt.title(f"Log(1+{col}) Histogram")
        plt.tight_layout()
        plt.show()

        # Categorical distributions
        plt.figure(figsize=(5 * len(self.categorical_cols), 4))
        for i, col in enumerate(self.categorical_cols):
            if col in self.data.columns:
                plt.subplot(1, len(self.categorical_cols), i + 1)
                counts = self.data[col].value_counts().head(self.top_n)
                sns.barplot(x=counts.index, y=counts.values, palette='pastel')
                plt.xticks(rotation=45)
                plt.title(f"Top {self.top_n} {col}")
        plt.tight_layout()
        plt.show()

    # ----------------------------
    # BIVARIATE ANALYSIS
    # ----------------------------
    def plot_bivariate(self):
        """
        Plot bivariate relationships for numeric columns.

        Includes scatter plot of TotalPremium vs TotalClaims and correlation heatmap.
        """
        if {"TotalPremium", "TotalClaims"}.issubset(self.data.columns):
            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                data=self.data,
                x="TotalPremium",
                y="TotalClaims",
                alpha=0.4,
                s=20,
                color="#F8766D",
                edgecolor=None
            )
            plt.title("Total Premium vs Total Claims", fontsize=14)
            plt.xlabel("Total Premium")
            plt.ylabel("Total Claims")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        numeric = [c for c in self.numeric_cols if c in self.data.columns]
        if numeric:
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                self.data[numeric].corr(),
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                linewidths=0.8,
                cbar_kws={"shrink": 0.8}
            )
            plt.title("Correlation Heatmap", fontsize=14)
            plt.tight_layout()
            plt.show()

    # ----------------------------
    # LOSS RATIO
    # ----------------------------
    def plot_loss_ratio(self, group_col):
        """
        Plot loss ratio for a categorical column.

        Parameters
        ----------
        group_col : str
            Column name to group by (e.g., Province, VehicleType, Gender).
        """
        data = self.grouped_loss_ratio(group_col)
        if not data.empty:
            data = data.sort_values("LossRatio", ascending=False)
            plt.figure(figsize=(10, 5))
            sns.barplot(x=data["LossRatio"], y=data.index, palette="flare")
            plt.axvline(1, color="black", linestyle="--", linewidth=1)
            plt.title(f"Loss Ratio by {group_col}")
            plt.xlabel("Loss Ratio")
            plt.tight_layout()
            plt.show()

    # ----------------------------
    # TEMPORAL TRENDS
    # ----------------------------
    def plot_temporal_trends(self):
        """
        Plot monthly trends for TotalPremium and TotalClaims.
        """
        monthly = self.monthly_trends()
        if not monthly.empty:
            plt.figure(figsize=(12, 5))
            sns.lineplot(data=monthly[["TotalPremium", "TotalClaims"]])
            plt.title("Total Premium vs Total Claims Over Time")
            plt.xlabel("Month")
            plt.ylabel("Amount")
            plt.tight_layout()
            plt.show()

    # ----------------------------
    # VEHICLE CLAIMS
    # ----------------------------
    def plot_vehicle_claims(self, top_n=5):
        """
        Plot highest and lowest average claims for vehicle make and model.

        Parameters
        ----------
        top_n : int
            Number of top/bottom entries to display.
        """
        summary = self.vehicle_claims_summary(top_n=top_n)
        for col in ["make", "Model"]:
            if col not in summary:
                continue
            highest = summary[col]["highest"]
            lowest = summary[col]["lowest"]

            plt.figure(figsize=(8, 4))
            sns.barplot(x=highest.values, y=highest.index, palette="rocket")
            plt.title(f"Top {top_n} {col} by Avg Claims")
            plt.xlabel("Average Claims")
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(8, 4))
            sns.barplot(x=lowest.values, y=lowest.index, palette="light:#5A9367")
            plt.title(f"Lowest {top_n} {col} by Avg Claims")
            plt.xlabel("Average Claims")
            plt.tight_layout()
            plt.show()

    # ----------------------------
    # OUTLIERS
    # ----------------------------
    def plot_outliers(self, columns=None):
        """
        Plot horizontal boxplots to detect outliers in numeric columns.

        Parameters
        ----------
        columns : list, optional
            List of numeric columns to check for outliers. Defaults to ["TotalClaims", "CustomValueEstimate"].
        """
        if columns is None:
            columns = ["TotalClaims", "CustomValueEstimate"]
        columns = [c for c in columns if c in self.data.columns]
        if not columns:
            return

        plt.figure(figsize=(8, len(columns)*2.5))
        for i, col in enumerate(columns, 1):
            plt.subplot(len(columns), 1, i)
            sns.boxplot(
                x=self.data[col],
                color="#F6AE2D",
                fliersize=5,
                linewidth=1.2
            )
            plt.title(f"Outliers: {col}", fontsize=12)
            plt.xlabel("")
            plt.tight_layout()
        plt.show()

    # ----------------------------
    # ZIP CODE ANALYSIS
    # ----------------------------
    def plot_zipcode_correlation(self):
        """
        Plot Spearman-like correlation (via ranks) between TotalPremium and TotalClaims
        for top N ZIP codes.
        """
        if "PostalCode" not in self.data.columns:
            print("PostalCode column not found.")
            return

        top_zips = self.data['PostalCode'].value_counts().nlargest(self.top_n).index
        correlations = {}
        for z in top_zips:
            zip_data = self.data[self.data['PostalCode'] == z].groupby('TransactionMonth').agg(
                TotalPremium=('TotalPremium', 'sum'),
                TotalClaims=('TotalClaims', 'sum')
            )
            if len(zip_data) > 1:
                ranked_premium = zip_data['TotalPremium'].rank()
                ranked_claims = zip_data['TotalClaims'].rank()
                correlations[z] = ranked_premium.corr(ranked_claims)

        corr_series = pd.Series(correlations).sort_values(ascending=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=corr_series.index, y=corr_series.values, palette='coolwarm')
        plt.ylabel("Spearman Correlation (via ranks)")
        plt.title(f"Top {self.top_n} ZIP Codes Correlation")
        plt.ylim(-1, 1)
        plt.tight_layout()
        plt.show()

    def plot_zipcode_scatter(self):
        """
        Plot monthly scatter and regression for TotalPremium vs TotalClaims
        by top N ZIP codes.
        """
        if "PostalCode" not in self.data.columns:
            print("PostalCode column not found.")
            return
        top_zips = self.data['PostalCode'].value_counts().nlargest(self.top_n).index
        plt.figure(figsize=(12, 6))
        for z in top_zips:
            zip_data = self.data[self.data['PostalCode']==z].groupby('TransactionMonth').agg(
                TotalPremium=('TotalPremium', 'sum'),
                TotalClaims=('TotalClaims', 'sum')
            )
            sns.scatterplot(x=zip_data['TotalPremium'], y=zip_data['TotalClaims'], label=f"ZIP {z}", s=70)
            sns.regplot(x=zip_data['TotalPremium'], y=zip_data['TotalClaims'], scatter=False, ci=None)
        plt.xlabel("Monthly Total Premium")
        plt.ylabel("Monthly Total Claims")
        plt.title("Monthly Premium vs Claims by ZIP Code")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ----------------------------
    # GEOGRAPHIC / CATEGORY TRENDS
    # ----------------------------
    def plot_geographic_trends(self, group_col: str):
        """
        Plot TotalPremium, TotalClaims, and PolicyCount for a geographic or categorical column.

        Parameters
        ----------
        group_col : str
            Column name to group by (e.g., CoverType, make, Province).
        """
        if group_col not in self.data.columns:
            print(f"{group_col} column not found.")
            return

        summary = self.data.groupby(group_col).agg(
            TotalPremium=('TotalPremium', 'sum'),
            TotalClaims=('TotalClaims', 'sum'),
            PolicyCount=('PolicyID', 'count')
        ).sort_values('TotalPremium', ascending=False)

        # Total Premium
        plt.figure(figsize=(10, 4))
        sns.barplot(x=summary.index, y=summary['TotalPremium'], palette='Blues_r')
        plt.xticks(rotation=45)
        plt.title(f"Total Premium by {group_col}")
        plt.ylabel("Total Premium")
        plt.tight_layout()
        plt.show()

        # Total Claims
        plt.figure(figsize=(10, 4))
        sns.barplot(x=summary.index, y=summary['TotalClaims'], palette='Reds_r')
        plt.xticks(rotation=45)
        plt.title(f"Total Claims by {group_col}")
        plt.ylabel("Total Claims")
        plt.tight_layout()
        plt.show()

        # Policy Count
        plt.figure(figsize=(10, 4))
        sns.barplot(x=summary.index, y=summary['PolicyCount'], palette='Greens_r')
        plt.xticks(rotation=45)
        plt.title(f"Policy Count by {group_col}")
        plt.ylabel("Policy Count")
        plt.tight_layout()
        plt.show()
