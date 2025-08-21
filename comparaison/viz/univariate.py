from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_pvalue_histogram(results_df: pd.DataFrame, test_type: str = "mwu",
                          corrected: bool = True, bins: int = 20, figsize=(10, 6)) -> None:
    p_col = f"{test_type}_pvalue_fdr" if corrected else f"{test_type}_pvalue"
    title_suffix = "(FDR Corrected)" if corrected else "(Uncorrected)"
    p_values = results_df[p_col].values
    alpha_threshold = 0.05

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    n, edges, patches = ax.hist(p_values, bins=bins, alpha=0.7, color="skyblue", edgecolor="black")

    # color bars that fall entirely below alpha
    for patch, start, end in zip(patches, edges[:-1], edges[1:]):
        if end <= alpha_threshold:
            patch.set_facecolor("lightcoral")

    test_names = {"mwu": "Mann-Whitney U", "ks": "Kolmogorov-Smirnov"}
    ax.axvline(alpha_threshold, color="red", linestyle="--", label=f"α = {alpha_threshold}")
    ax.set_xlabel("P-values"); ax.set_ylabel("Frequency")
    ax.set_title(f"{test_names.get(test_type, test_type)} P-value Distribution {title_suffix}")
    ax.legend(); ax.grid(True, alpha=0.3)

    n_sig = int(np.sum(p_values < alpha_threshold))
    n_total = len(p_values)
    txt = f"Significant: {n_sig}/{n_total} ({(n_sig/n_total*100 if n_total else 0):.1f}%)"
    ax.text(0.95, 0.95, txt, transform=ax.transAxes, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout(); plt.show()


def plot_violinplots(df1: pd.DataFrame, df2: pd.DataFrame, results_df: pd.DataFrame,
                     features: List[str], group1_name: str, group2_name: str,
                     alpha: float = 0.05, max_plots_per_row: int = 4,
                     figsize_per_plot=(4, 3)) -> None:
    n_features = len(features)
    n_rows = (n_features + max_plots_per_row - 1) // max_plots_per_row

    fig, axes = plt.subplots(
        n_rows, max_plots_per_row,
        figsize=(figsize_per_plot[0] * max_plots_per_row, figsize_per_plot[1] * n_rows)
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes_flat[i]
        data1 = df1[feat].dropna().values
        data2 = df2[feat].dropna().values

        parts = ax.violinplot([data1, data2], positions=[1, 2], showmeans=True, showmedians=False)
        # colors
        for body, color in zip(parts["bodies"], ["lightblue", "lightcoral"]):
            body.set_facecolor(color); body.set_alpha(0.7)

        ax.set_xticks([1, 2]); ax.set_xticklabels([group1_name, group2_name])

        row = results_df[results_df["feature"] == feat].iloc[0]
        mwu_p = float(row["mwu_pvalue_fdr"])
        rb_corr = float(row["rank_biserial_correlation"])
        rb_eff = str(row["rb_effect_size"])
        stars = "***" if mwu_p < 0.001 else "**" if mwu_p < 0.01 else "*" if mwu_p < 0.05 else ""
        ax.set_title(f"{feat}\nMWU p-val: {mwu_p:.3e}{stars}\nrb: {rb_corr:.3f} ({rb_eff})", fontsize=10)
        ax.grid(True, alpha=0.3)

    # hide unused
    for i in range(n_features, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout(); plt.show()
