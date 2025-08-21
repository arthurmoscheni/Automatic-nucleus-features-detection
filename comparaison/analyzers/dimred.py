from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind, mannwhitneyu, ks_2samp
import umap  # umap-learn

# plotting-only deps live in viz to keep this class testable
from viz.dimred import (
    plot_correlation_matrices,
    plot_variance_explained,
    plot_pca_components,
    plot_significant_pcs,
    plot_pc_statistics,
    plot_umap_results,
    plot_umap_statistics,
    plot_umap_feature_importance,
)


class PCA_UMAP:
    """Class wrapper that runs correlation, PCA and UMAP and (optionally) plots via viz helpers."""

    def __init__(self, df1: pd.DataFrame, df2: pd.DataFrame,
                 group1: str = "Group 1", group2: str = "Group 2",
                 features: Optional[List[str]] = None):
        self.df1 = df1.dropna()
        self.df2 = df2.dropna()
        self.label1 = group1
        self.label2 = group2
        self.features = features if features is not None else self.df1.columns.tolist()

    # ---------- Correlations ----------
    def correlation_matrices(self, plot: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        corr1 = self.df1[self.features].corr()
        corr2 = self.df2[self.features].corr()
        corr_diff = corr1 - corr2
        if plot:
            plot_correlation_matrices(corr1, corr2, corr_diff, self.label1, self.label2)
        return corr1, corr2, corr_diff

    # ---------- PCA ----------
    def pca_analysis(self, plot_variance: bool = True):
        combined, pca, _ = self._prepare_pca_data()

        pca_df = self._create_pca_dataframe(combined["transformed"], combined["labels"])
        if plot_variance:
            plot_variance_explained(pca)
            plot_pca_components(pca_df, pca.explained_variance_ratio_, self.label1, self.label2)

        pc_test_results = self._test_pca_components(pca_df, pca.explained_variance_ratio_)
        # keep your original behavior of attaching results to the DataFrame
        pca_df.pc_test_results = pd.DataFrame(pc_test_results)

        # same downstream plots as your original
        plot_significant_pcs(pca, pc_test_results, self.features)
        plot_pc_statistics(pca_df, pc_test_results, pca.explained_variance_ratio_, self.label1, self.label2)
        return pca_df, pca

    def _prepare_pca_data(self):
        d1 = self.df1[self.features].copy()
        d2 = self.df2[self.features].copy()
        d1["population"] = self.label1
        d2["population"] = self.label2

        combined = pd.concat([d1, d2], ignore_index=True)
        X = combined[self.features].values
        y = combined["population"].values

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        pca = PCA()
        Xp = pca.fit_transform(Xs)

        return {"transformed": Xp, "labels": y}, pca, scaler

    def _create_pca_dataframe(self, transformed: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(transformed, columns=[f"PC{i+1}" for i in range(transformed.shape[1])])
        df["population"] = labels
        return df

    def _test_pca_components(self, pca_df: pd.DataFrame, variance_ratios: np.ndarray) -> List[Dict]:
        print("\n--- Statistical Tests on Principal Components ---")
        results: List[Dict] = []
        for i in range(min(4, len(variance_ratios))):
            pc_col = f"PC{i+1}"
            v1 = pca_df[pca_df["population"] == self.label1][pc_col].values
            v2 = pca_df[pca_df["population"] == self.label2][pc_col].values

            t_stat, t_p = ttest_ind(v1, v2)
            _, mwu_p = mannwhitneyu(v1, v2, alternative="two-sided")
            _, ks_p = ks_2samp(v1, v2)

            result = {
                "PC": pc_col,
                "variance_explained": variance_ratios[i],
                "mean_pop1": float(v1.mean()),
                "mean_pop2": float(v2.mean()),
                "t_statistic": float(t_stat),
                "t_pvalue": float(t_p),
                "mwu_pvalue": float(mwu_p),
                "ks_pvalue": float(ks_p),
            }
            results.append(result)
            print(f"{pc_col} ({variance_ratios[i]:.1%} variance): "
                  f"t-test p={t_p:.2e}, MWU p={mwu_p:.2e}, KS p={ks_p:.2e}")
        return results

    # ---------- UMAP ----------
    def umap_analysis(self, n_neighbors: int = 10, min_dist: float = 0.05,
                      n_components: int = 2, random_state: int = 42, plot: bool = True):
        combined, umap_model = self._prepare_umap_data(n_neighbors, min_dist, n_components, random_state)
        umap_df = self._create_umap_dataframe(combined["transformed"], combined["labels"])

        if plot and n_components >= 2:
            plot_umap_results(umap_df)

        umap_test_results = self._test_umap_components(umap_df, n_components)
        umap_df.umap_test_results = pd.DataFrame(umap_test_results)

        plot_umap_statistics(umap_df, umap_test_results, n_components, self.label1, self.label2)
        plot_umap_feature_importance(umap_df, combined["standardized"], self.features, n_components)
        return umap_df, umap_model

    def _prepare_umap_data(self, n_neighbors: int, min_dist: float,
                           n_components: int, random_state: int):
        d1 = self.df1[self.features].copy()
        d2 = self.df2[self.features].copy()
        d1["population"] = self.label1
        d2["population"] = self.label2
        combined = pd.concat([d1, d2], ignore_index=True)

        X = combined[self.features].values
        y = combined["population"].values

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=random_state
        )
        X_umap = umap_model.fit_transform(Xs)
        return {"transformed": X_umap, "labels": y, "standardized": Xs}, umap_model

    def _create_umap_dataframe(self, transformed: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
        cols = [f"UMAP{i+1}" for i in range(transformed.shape[1])]
        df = pd.DataFrame(transformed, columns=cols)
        df["population"] = labels
        return df

    def _test_umap_components(self, umap_df: pd.DataFrame, n_components: int) -> List[Dict]:
        print("\n--- Statistical Tests on UMAP Components ---")
        results: List[Dict] = []
        for i in range(min(n_components, 4)):
            col = f"UMAP{i+1}"
            v1 = umap_df[umap_df["population"] == self.label1][col].values
            v2 = umap_df[umap_df["population"] == self.label2][col].values

            t_stat, t_p = ttest_ind(v1, v2)
            _, mwu_p = mannwhitneyu(v1, v2, alternative="two-sided")
            _, ks_p = ks_2samp(v1, v2)

            result = {
                "Component": col,
                "mean_pop1": float(v1.mean()),
                "mean_pop2": float(v2.mean()),
                "t_statistic": float(t_stat),
                "t_pvalue": float(t_p),
                "mwu_pvalue": float(mwu_p),
                "ks_pvalue": float(ks_p),
            }
            results.append(result)
            print(f"{col}: t-test p={t_p:.2e}, MWU p={mwu_p:.2e}, KS p={ks_p:.2e}")
        return results

    # ---------- Orchestration ----------
    def run_all(self, n_neighbors: int = 15, min_dist: float = 0.1,
                n_components: int = 2, random_state: int = 42, plot: bool = True):
        self.correlation_matrices(plot=plot)
        pca_results = self.pca_analysis(plot_variance=plot)
        umap_results = self.umap_analysis(
            n_neighbors=n_neighbors, min_dist=min_dist,
            n_components=n_components, random_state=random_state, plot=plot
        )
        return pca_results, umap_results
