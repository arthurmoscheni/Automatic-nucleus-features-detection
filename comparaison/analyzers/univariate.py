from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp
from statsmodels.stats.multitest import multipletests

# all plotting lives in viz; methods below just delegate to these helpers
from viz.univariate import (
    plot_pvalue_histogram as _plot_pvalue_histogram_viz,
    plot_violinplots as _plot_violinplots_viz,
)


class UnivariateComparison:
    def __init__(self, df1: pd.DataFrame, df2: pd.DataFrame,
                 group1: Optional[str] = None, group2: Optional[str] = None,
                 features: Optional[List[str]] = None):
        """
        Initialize with two groups for comparison.

        Parameters
        ----------
        df1, df2 : DataFrame
        group1, group2 : str
        features : list[str] | None
        """
        self.df1 = df1.copy()
        self.df2 = df2.copy()
        self.features = features if features is not None else self.df1.columns.tolist()
        self.group1_name = group1 if group1 is not None else "Pop1"
        self.group2_name = group2 if group2 is not None else "Pop2"

        # keep your original normalization of 'ratio' -> 'wrinkle_ratio'
        self._rename_ratio_column()

    # ---------- internals (unchanged logic) ----------
    def _rename_ratio_column(self) -> None:
        if "ratio" in self.df1.columns:
            self.df1 = self.df1.rename(columns={"ratio": "wrinkle_ratio"})
        if "ratio" in self.df2.columns:
            self.df2 = self.df2.rename(columns={"ratio": "wrinkle_ratio"})
        if "ratio" in self.features:
            self.features = [f if f != "ratio" else "wrinkle_ratio" for f in self.features]

    def permutation_test(self, x1: np.ndarray, x2: np.ndarray, n_permutations: int = 10000) -> Tuple[float, float]:
        observed_diff = float(np.mean(x1) - np.mean(x2))
        combined = np.concatenate([x1, x2])
        n1 = len(x1)
        perm_diffs = np.empty(n_permutations, dtype=float)
        for i in range(n_permutations):
            np.random.shuffle(combined)
            perm_diffs[i] = float(np.mean(combined[:n1]) - np.mean(combined[n1:]))
        p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(observed_diff)))
        return observed_diff, p_value

    def _calculate_rank_biserial_correlation(self, x1: np.ndarray, x2: np.ndarray, mwu_stat: float) -> float:
        n1, n2 = len(x1), len(x2)
        return float((2 * mwu_stat) / (n1 * n2) - 1)

    def _get_effect_size_label(self, correlation: float) -> str:
        a = abs(correlation)
        if a >= 0.5:
            return "Large"
        elif a >= 0.3:
            return "Medium"
        return "Small"

    # ---------- main compute (unchanged outputs) ----------
    def comprehensive_tests(self, alpha: float = 0.05) -> pd.DataFrame:
        results = []
        p_values_mwu = []
        p_values_ks = []

        for feat in self.features:
            x1 = self.df1[feat].dropna().values
            x2 = self.df2[feat].dropna().values

            mwu_stat, mwu_p = mannwhitneyu(x1, x2, alternative="two-sided")
            ks_stat, ks_p = ks_2samp(x1, x2)
            perm_diff, perm_p = self.permutation_test(x1, x2)

            rb_corr = self._calculate_rank_biserial_correlation(x1, x2, mwu_stat)

            print(f"Feature: {feat}, Permutation diff: {perm_diff:.4f}, Permutation p-value: {perm_p:.2e}")

            results.append({
                "feature": feat,
                "mwu_statistic": float(mwu_stat),
                "perm_pvalue": float(perm_p),
                "mwu_pvalue": float(mwu_p),
                "rank_biserial_correlation": float(rb_corr),
                "rb_effect_size": self._get_effect_size_label(rb_corr),
            })
            p_values_mwu.append(mwu_p)
            p_values_ks.append(ks_p)

        # FDR correction (kept)
        _, mwu_fdr, _, _ = multipletests(p_values_mwu, alpha=alpha, method="fdr_bh")
        _, ks_fdr, _, _ = multipletests(p_values_ks, alpha=alpha, method="fdr_bh")

        for i, row in enumerate(results):
            row["mwu_pvalue_fdr"] = float(mwu_fdr[i])
            row["ks_pvalue_fdr"] = float(ks_fdr[i])
            row["mwu_significant_fdr"] = bool(mwu_fdr[i] < alpha)
            row["ks_significant_fdr"] = bool(ks_fdr[i] < alpha)

        return pd.DataFrame(results)

    # ---------- plotting (delegates to viz helpers) ----------
    def plot_pvalue_histogram(self, results_df: pd.DataFrame, test_type: str = "mwu",
                              corrected: bool = True, bins: int = 20, figsize=(10, 6)) -> None:
        _plot_pvalue_histogram_viz(results_df, test_type=test_type, corrected=corrected,
                                   bins=bins, figsize=figsize)

    def plot_violinplots(self, results_df: pd.DataFrame, alpha: float = 0.05,
                         max_plots_per_row: int = 4, figsize_per_plot=(4, 3)) -> None:
        _plot_violinplots_viz(
            df1=self.df1,
            df2=self.df2,
            results_df=results_df,
            features=self.features,
            group1_name=self.group1_name,
            group2_name=self.group2_name,
            alpha=alpha,
            max_plots_per_row=max_plots_per_row,
            figsize_per_plot=figsize_per_plot,
        )

    # ---------- one-call runner (kept semantics) ----------
    def run_all(self, alpha: float = 0.05, visualization: bool = True) -> pd.DataFrame:
        results_df = self.comprehensive_tests(alpha=alpha)
        if visualization:
            self.plot_pvalue_histogram(results_df, test_type="mwu", corrected=True)
            self.plot_violinplots(results_df, alpha=alpha)
        return results_df
