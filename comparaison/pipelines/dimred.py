from __future__ import annotations
from typing import Optional, List, Tuple
import pandas as pd

from analyzers.dimred import PCA_UMAP

def run_all_dimred(df1: pd.DataFrame, df2: pd.DataFrame,
                   group1: str = "Group 1", group2: str = "Group 2",
                   features: Optional[List[str]] = None,
                   n_neighbors: int = 15, min_dist: float = 0.1,
                   n_components: int = 2, random_state: int = 42, plot: bool = True):
    analyzer = PCA_UMAP(df1, df2, group1, group2, features)
    return analyzer.run_all(n_neighbors=n_neighbors, min_dist=min_dist,
                            n_components=n_components, random_state=random_state, plot=plot)

def run_pca(df1: pd.DataFrame, df2: pd.DataFrame,
            group1: str = "Group 1", group2: str = "Group 2",
            features: Optional[List[str]] = None, plot: bool = True):
    analyzer = PCA_UMAP(df1, df2, group1, group2, features)
    return analyzer.pca_analysis(plot_variance=plot)

def run_umap(df1: pd.DataFrame, df2: pd.DataFrame,
             group1: str = "Group 1", group2: str = "Group 2",
             features: Optional[List[str]] = None,
             n_neighbors: int = 10, min_dist: float = 0.05,
             n_components: int = 2, random_state: int = 42, plot: bool = True):
    analyzer = PCA_UMAP(df1, df2, group1, group2, features)
    return analyzer.umap_analysis(n_neighbors=n_neighbors, min_dist=min_dist,
                                  n_components=n_components, random_state=random_state, plot=plot)
