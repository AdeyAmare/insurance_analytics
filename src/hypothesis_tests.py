"""Statistical hypothesis testing helpers for Task 3."""

import pandas as pd
from scipy import stats

class HypothesisTester:
    """
    Wrapper for statistical hypothesis tests.

    Methods include:
    - chi_squared_test: for categorical variables
    - ttest_two_groups: for comparing two numerical groups
    - anova_test: for comparing multiple numerical groups
    """

    class TestResult:
        """Container for hypothesis test results."""
        def __init__(self, statistic, p_value, reject_null):
            self.statistic = statistic
            self.p_value = p_value
            self.reject_null = reject_null

        def __repr__(self):
            return (
                f"TestResult(statistic={self.statistic:.4f}, "
                f"p_value={self.p_value:.6f}, "
                f"reject_null={self.reject_null})"
            )

    def __init__(self, df: pd.DataFrame):
        """Initialize with the DataFrame to test."""
        self.df = df.copy()

    def chi_squared_test(self, col1, col2, alpha=0.05):
        """Chi-squared test for independence between two categorical variables."""
        contingency = pd.crosstab(self.df[col1], self.df[col2])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        return self.TestResult(
            statistic=float(chi2),
            p_value=float(p_value),
            reject_null=(p_value < alpha),
        )

    def ttest_two_groups(self, group_col, value_col, group_a, group_b, alpha=0.05):
        """Two-sample independent t-test for numerical variable across two groups."""
        a = self.df.loc[self.df[group_col] == group_a, value_col].dropna()
        b = self.df.loc[self.df[group_col] == group_b, value_col].dropna()
        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
        return self.TestResult(
            statistic=float(t_stat),
            p_value=float(p_value),
            reject_null=(p_value < alpha),
        )

    def anova_test(self, group_col, value_col, alpha=0.05):
        """One-way ANOVA for numerical variable across multiple groups."""
        groups = [
            group[value_col].dropna().values
            for _, group in self.df.groupby(group_col)
        ]
        f_stat, p_value = stats.f_oneway(*groups)
        return self.TestResult(
            statistic=float(f_stat),
            p_value=float(p_value),
            reject_null=(p_value < alpha),
        )
