import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.eda import InsuranceEDA

class InsuranceEDAVisualizer(InsuranceEDA):
    """Simplified beginner-friendly EDA visualizations inheriting InsuranceEDA."""

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        sns.set(style="whitegrid")
        # Essential columns only
        self.essential_numeric = ["TotalPremium", "TotalClaims", "CustomValueEstimate"]
        self.essential_categorical = ["Province", "VehicleType", "Gender", "Make", "Model", "CoverType"]

    # ----------------------------
    # UNIVARIATE
    # ----------------------------
    def plot_univariate(self):
        """Combined histograms for numeric and bar charts for categorical columns."""

        # ----------------------------
        # Numeric
        # ----------------------------
        numeric_cols = [col for col in self.essential_numeric if col in self.df.columns]
        if numeric_cols:
            self.df[numeric_cols].hist(bins=20, figsize=(12, 4), layout=(1, len(numeric_cols)), color="skyblue", edgecolor="black")
            plt.suptitle("Distribution of Numeric Columns")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

        # ----------------------------
        # Categorical
        # ----------------------------
        cat_cols = [col for col in self.essential_categorical if col in self.df.columns]
        if cat_cols:
            fig, axes = plt.subplots(1, len(cat_cols), figsize=(12, 4))
            if len(cat_cols) == 1:
                axes = [axes]  # Ensure axes is iterable
            for ax, col in zip(axes, cat_cols):
                self.df[col].value_counts().head(10).plot(kind="bar", ax=ax, color="lightgreen")
                ax.set_title(col)
                ax.set_ylabel("Count")
            plt.suptitle("Top Categories in Categorical Columns")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

    # ----------------------------
    # BIVARIATE
    # ----------------------------
    def plot_bivariate(self):
        """Simple scatter plot for TotalPremium vs TotalClaims and correlation heatmap."""
        if {"TotalPremium", "TotalClaims"}.issubset(self.df.columns):
            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=self.df, x="TotalPremium", y="TotalClaims", alpha=0.5)
            plt.title("Premium vs Claims")
            plt.tight_layout()
            plt.show()

        # Correlation matrix of numeric columns
        plt.figure(figsize=(6, 5))
        sns.heatmap(self.df[self.essential_numeric].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()

    # ----------------------------
    # LOSS RATIO
    # ----------------------------
    def plot_loss_ratio(self, group_col):
        """Bar plot of loss ratio by Province, VehicleType, or Gender."""
        df = self.grouped_loss_ratio(group_col)
        if not df.empty:
            plt.figure(figsize=(8, 4))
            df["LossRatio"].sort_values().plot(kind="bar", color="skyblue")
            plt.axhline(1, color="red", linestyle="--")
            plt.title(f"Loss Ratio by {group_col}")
            plt.ylabel("Loss Ratio")
            plt.tight_layout()
            plt.show()

    # ----------------------------
    # TEMPORAL TRENDS
    # ----------------------------
    def plot_temporal_trends(self):
        """Line plot for TotalPremium and TotalClaims over time."""
        monthly_df = self.monthly_trends()
        if not monthly_df.empty:
            plt.figure(figsize=(10, 4))
            plt.plot(monthly_df.index, monthly_df["TotalPremium"], label="Total Premium")
            plt.plot(monthly_df.index, monthly_df["TotalClaims"], label="Total Claims")
            plt.title("Premium vs Claims Over Time")
            plt.xlabel("Month")
            plt.ylabel("Amount")
            plt.legend()
            plt.tight_layout()
            plt.show()

    # ----------------------------
    # OUTLIERS
    # ----------------------------
    def plot_outliers(self):
        """Boxplots for outlier detection in TotalClaims and CustomValueEstimate."""
        for col in ["TotalClaims", "CustomValueEstimate"]:
            if col in self.df.columns:
                plt.figure(figsize=(5, 4))
                sns.boxplot(x=self.df[col])
                plt.title(f"Boxplot of {col}")
                plt.tight_layout()
                plt.show()

    # ----------------------------
    # VEHICLE CLAIMS
    # ----------------------------
    def plot_vehicle_claims(self, top_n=5):
        """Bar plots for top and bottom vehicle makes/models by average claims."""
        vehicle_summary = self.vehicle_claims_summary(top_n=top_n)
        for col, summary in vehicle_summary.items():
            if col in summary:
                # Top
                plt.figure(figsize=(8, 4))
                summary["highest"].plot(kind="bar", color="salmon")
                plt.title(f"Top {top_n} {col} by Avg Claims")
                plt.ylabel("Average Claims")
                plt.tight_layout()
                plt.show()
                # Bottom
                plt.figure(figsize=(8, 4))
                summary["lowest"].plot(kind="bar", color="lightgreen")
                plt.title(f"Bottom {top_n} {col} by Avg Claims")
                plt.ylabel("Average Claims")
                plt.tight_layout()
                plt.show()

    # ----------------------------
    # TOP INSIGHTS
    # ----------------------------
    def plot_top_insights(self):
        """Produce 3 key visualizations for reporting."""
        self.plot_loss_ratio("Province")
        self.plot_temporal_trends()
        self.plot_vehicle_claims(top_n=5)
