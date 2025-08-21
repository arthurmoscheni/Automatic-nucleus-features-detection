from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    chi2, shapiro, anderson, kstest, normaltest, levene, bartlett
)
from viz.normality import create_comprehensive_plots


class NormalityTester:
    """A class for performing comprehensive normality and variance equality tests."""
    
    def __init__(self, df1: pd.DataFrame = None, df2: pd.DataFrame = None, 
                 group1: str = "Group1", group2: str = "Group2", alpha: float = 0.05, plot: bool = False):
        """
        Initialize the NormalityTester.
        
        Args:
            df1: First group DataFrame
            df2: Second group DataFrame
            group1: Name for the first group
            group2: Name for the second group
            alpha: Significance level for tests (default: 0.05)
        """
        self.df1 = df1
        self.df2 = df2
        self.group1 = group1
        self.group2 = group2
        self.alpha = alpha
        self.plot = plot

    
    def univariate_normality_tests(self, data: pd.DataFrame, group_name: str = "") -> dict:
        """
        Perform comprehensive univariate normality tests.
        
        Args:
            data: DataFrame containing the data to test
            group_name: Optional name for the group being tested
            
        Returns:
            Dictionary containing test results for each feature
        """
        results: dict = {}

        for feature in data.columns:
            feature_data = data[feature].dropna()

            if len(feature_data) < 3:
                results[feature] = {"error": "Insufficient data points"}
                continue

            feature_results: dict = {}

            # Shapiro–Wilk (≤ 5000)
            if len(feature_data) <= 5000:
                try:
                    stat, p_val = shapiro(feature_data)
                    feature_results["shapiro_wilk"] = {
                        "statistic": float(stat),
                        "p_value": float(p_val),
                        "normal": bool(p_val > self.alpha),
                    }
                except Exception:
                    feature_results["shapiro_wilk"] = {"error": "Test failed"}

            # Anderson–Darling
            try:
                result = anderson(feature_data, dist="norm")
                feature_results["anderson_darling"] = {
                    "statistic": float(result.statistic),
                    "critical_values": result.critical_values,
                    "significance_levels": result.significance_levels,
                    "normal": bool(result.statistic <= result.critical_values[2]),  # 5%
                }
            except Exception:
                feature_results["anderson_darling"] = {"error": "Test failed"}

            # Kolmogorov–Smirnov (vs fitted normal)
            try:
                stat, p_val = kstest(
                    feature_data, "norm", args=(np.mean(feature_data), np.std(feature_data))
                )
                feature_results["kolmogorov_smirnov"] = {
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "normal": bool(p_val > self.alpha),
                }
            except Exception:
                feature_results["kolmogorov_smirnov"] = {"error": "Test failed"}

            # D'Agostino–Pearson (n ≥ 8)
            if len(feature_data) >= 8:
                try:
                    stat, p_val = normaltest(feature_data)
                    feature_results["dagostino_pearson"] = {
                        "statistic": float(stat),
                        "p_value": float(p_val),
                        "normal": bool(p_val > self.alpha),
                    }
                except Exception:
                    feature_results["dagostino_pearson"] = {"error": "Test failed"}

            # Descriptives
            feature_results["descriptive"] = {
                "mean": float(np.mean(feature_data)),
                "median": float(np.median(feature_data)),
                "std": float(np.std(feature_data)),
                "skewness": float(stats.skew(feature_data)),
                "kurtosis": float(stats.kurtosis(feature_data)),
                "n_samples": int(len(feature_data)),
            }

            results[feature] = feature_results

        return results

    def variance_equality_tests(self, df1: pd.DataFrame, df2: pd.DataFrame, feature_names: list[str]) -> dict:
        """
        Test equality of variances between two groups (Levene, Bartlett, F).
        
        Args:
            df1: First group DataFrame
            df2: Second group DataFrame
            feature_names: List of feature names to test
            
        Returns:
            Dictionary containing variance equality test results
        """
        results: dict = {}

        for feature in feature_names:
            data1 = df1[feature].dropna()
            data2 = df2[feature].dropna()

            if len(data1) < 2 or len(data2) < 2:
                results[feature] = {"error": "Insufficient data for variance tests"}
                continue

            feature_results: dict = {}

            # Levene
            try:
                stat, p_val = levene(data1, data2)
                feature_results["levene"] = {
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "equal_variances": bool(p_val > self.alpha),
                }
            except Exception:
                feature_results["levene"] = {"error": "Test failed"}

            # Bartlett
            try:
                stat, p_val = bartlett(data1, data2)
                feature_results["bartlett"] = {
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "equal_variances": bool(p_val > self.alpha),
                }
            except Exception:
                feature_results["bartlett"] = {"error": "Test failed"}

            # F-test
            try:
                var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
                f_stat = var1 / var2 if var1 > var2 else var2 / var1
                df_1, df_2 = len(data1) - 1, len(data2) - 1
                p_val = 2 * (1 - stats.f.cdf(f_stat, df_1, df_2))
                feature_results["f_test"] = {
                    "statistic": float(f_stat),
                    "p_value": float(p_val),
                    "equal_variances": bool(p_val > self.alpha),
                    "variance_ratio": float(var1 / var2),
                }
            except Exception:
                feature_results["f_test"] = {"error": "Test failed"}

            results[feature] = feature_results

        return results

    def mardia_test(self, data: pd.DataFrame, group_name: str = "") -> dict:
        """
        Mardia's test for multivariate normality.
        
        Args:
            data: DataFrame containing the data to test
            group_name: Optional name for the group being tested
            
        Returns:
            Dictionary containing Mardia's test results
        """
        if data.empty:
            return {"error": "Empty dataset"}

        n, p = data.shape
        if n < p + 1:
            return {"error": f"Insufficient samples (n={n}) for {p} variables. Need at least {p+1} samples."}
        if n < 20:
            return {"warning": f"Small sample size (n={n}). Results may be unreliable."}

        try:
            data_centered = data - data.mean()
            S = np.cov(data_centered.T)

            eigenvals = np.linalg.eigvals(S)
            if np.min(eigenvals) < 1e-10:
                S += np.eye(p) * 1e-8

            S_inv = np.linalg.inv(S)

            mahal_dist = []
            for i in range(n):
                d = data_centered.iloc[i].values
                mahal_dist.append(float(d.T @ S_inv @ d))
            mahal_dist = np.array(mahal_dist)

            b1p = float(np.sum(mahal_dist ** 3) / n)
            skew_stat = float(n * b1p / 6)
            skew_df = float(p * (p + 1) * (p + 2) / 6)
            skew_p_value = float(1 - chi2.cdf(skew_stat, skew_df))

            b2p = float(np.sum(mahal_dist ** 2) / n)
            expected_b2p = float(p * (p + 2))
            kurt_stat = float((b2p - expected_b2p) / np.sqrt(8 * p * (p + 2) / n))
            kurt_p_value = float(2 * (1 - stats.norm.cdf(abs(kurt_stat))))

            combined_stat = float(skew_stat + kurt_stat ** 2)
            combined_df = float(skew_df + 1)
            combined_p_value = float(1 - chi2.cdf(combined_stat, combined_df))

            return {
                "n_samples": int(n),
                "n_variables": int(p),
                "skewness_stat": skew_stat,
                "skewness_df": skew_df,
                "skewness_p_value": skew_p_value,
                "skewness_normal": bool(skew_p_value > self.alpha),
                "kurtosis_stat": kurt_stat,
                "kurtosis_p_value": kurt_p_value,
                "kurtosis_normal": bool(kurt_p_value > self.alpha),
                "combined_stat": combined_stat,
                "combined_df": combined_df,
                "combined_p_value": combined_p_value,
                "multivariate_normal": bool(combined_p_value > self.alpha),
                "condition_number": float(np.linalg.cond(S)),
            }

        except np.linalg.LinAlgError as e:
            return {"error": f"Linear algebra error: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def box_m_test(self, df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
        """
        Box's M test for equality of covariance matrices.
        
        Args:
            df1: First group DataFrame
            df2: Second group DataFrame
            
        Returns:
            Dictionary containing Box's M test results
        """
        if df1.empty or df2.empty:
            return {"error": "One or both datasets are empty"}

        n1, p = df1.shape
        n2, _ = df2.shape
        if n1 <= p or n2 <= p:
            return {"error": f"Insufficient samples for Box's M test. Need > {p} samples per group, have {n1} and {n2}"}

        try:
            S1 = np.cov(df1.T)
            S2 = np.cov(df2.T)

            cond1 = float(np.linalg.cond(S1))
            cond2 = float(np.linalg.cond(S2))
            if cond1 > 1e12 or cond2 > 1e12:
                return {"error": "Covariance matrices are nearly singular (high condition number)"}

            S_pooled = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)

            det_S1 = float(np.linalg.det(S1))
            det_S2 = float(np.linalg.det(S2))
            det_Sp = float(np.linalg.det(S_pooled))
            if det_S1 <= 0 or det_S2 <= 0 or det_Sp <= 0:
                return {"error": "Non-positive determinant encountered"}

            M = float((n1 - 1) * np.log(det_S1) + (n2 - 1) * np.log(det_S2) - (n1 + n2 - 2) * np.log(det_Sp))
            c = float((2 * p ** 2 + 3 * p - 1) / (6 * (p + 1)) * (1 / (n1 - 1) + 1 / (n2 - 1) - 1 / (n1 + n2 - 2)))

            chi2_stat = float(-M * (1 - c))
            df = float(p * (p + 1) / 2)
            p_value = float(1 - chi2.cdf(chi2_stat, df))

            return {
                "M_statistic": M,
                "chi2_statistic": chi2_stat,
                "degrees_of_freedom": df,
                "p_value": p_value,
                "equal_covariances": bool(p_value > self.alpha),
                "condition_numbers": {"group1": cond1, "group2": cond2},
                "sample_sizes": {"group1": int(n1), "group2": int(n2)},
            }

        except np.linalg.LinAlgError as e:
            return {"error": f"Linear algebra error: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    def plot_results(self, df1: pd.DataFrame, df2: pd.DataFrame, group1: str, group2: str) -> None:
        create_comprehensive_plots(df1, df2, group1, group2)


    def run_all_tests(self) -> dict:
        """
        Run all normality and variance equality tests for both groups.
        
        Returns:
            Dictionary containing all test results
        """
        results = {}
        
        if self.df1 is not None:
            results[f"{self.group1}_normality"] = self.univariate_normality_tests(self.df1, self.group1)
            results[f"{self.group1}_multivariate"] = self.mardia_test(self.df1, self.group1)
        
        if self.df2 is not None:
            results[f"{self.group2}_normality"] = self.univariate_normality_tests(self.df2, self.group2)
            results[f"{self.group2}_multivariate"] = self.mardia_test(self.df2, self.group2)
        
        if self.df1 is not None and self.df2 is not None:
            common_features = list(set(self.df1.columns) & set(self.df2.columns))
            results["variance_equality"] = self.variance_equality_tests(self.df1, self.df2, common_features)
            results["covariance_equality"] = self.box_m_test(self.df1, self.df2)
        if self.plot:
            self.plot_results(self.df1, self.df2, self.group1, self.group2)
        return results