from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")
sns.set_palette("husl")

# ---- correlations ----
def plot_correlation_matrices(corr1, corr2, corr_diff, label1: str, label2: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(corr1, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1, fmt=".2f",
                cbar_kws={"shrink": 0.8}, ax=axes[0])
    axes[0].set_title(f"{label1} Correlations")
    sns.heatmap(corr2, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1, fmt=".2f",
                cbar_kws={"shrink": 0.8}, ax=axes[1])
    axes[1].set_title(f"{label2} Correlations")
    plt.tight_layout(); plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(corr_diff, annot=True, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5, fmt=".2f",
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(f"Correlation Difference ({label1} - {label2})")
    plt.tight_layout(); plt.show()

# ---- PCA plots ----
def plot_variance_explained(pca) -> None:
    ratios = pca.explained_variance_ratio_
    cum = np.cumsum(ratios)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(range(1, len(ratios) + 1), ratios, "bo-")
    ax[0].set_xlabel("Principal Component"); ax[0].set_ylabel("Explained Variance Ratio")
    ax[0].set_title("PCA Explained Variance"); ax[0].grid(alpha=0.3)
    ax[1].plot(range(1, len(cum) + 1), cum, "ro-")
    ax[1].set_xlabel("Number of Components"); ax[1].set_ylabel("Cumulative Variance")
    ax[1].set_title("Cumulative Explained Variance")
    ax[1].axhline(0.8, linestyle="--", alpha=0.7); ax[1].axhline(0.95, linestyle=":", alpha=0.7)
    ax[1].grid(alpha=0.3); plt.tight_layout(); plt.show()

def plot_pca_components(pca_df: pd.DataFrame, variance_ratios: np.ndarray, label1: str, label2: str) -> None:
    if pca_df.shape[1] < 6:  # need PC1..PC4 + population
        return
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    mask1 = pca_df["population"] == label1
    mask2 = pca_df["population"] == label2
    pairs = [("PC1","PC2",0,1), ("PC1","PC3",0,2), ("PC1","PC4",0,3),
             ("PC2","PC3",1,2), ("PC2","PC4",1,3), ("PC3","PC4",2,3)]
    for i,(pc1,pc2,i1,i2) in enumerate(pairs):
        r,c = divmod(i,3)
        axes[r,c].scatter(pca_df.loc[mask1,pc1], pca_df.loc[mask1,pc2], alpha=0.6, label=label1, s=30)
        axes[r,c].scatter(pca_df.loc[mask2,pc1], pca_df.loc[mask2,pc2], alpha=0.6, label=label2, s=30)
        axes[r,c].set_xlabel(f"{pc1} ({variance_ratios[i1]:.1%} var)")
        axes[r,c].set_ylabel(f"{pc2} ({variance_ratios[i2]:.1%} var)")
        axes[r,c].set_title(f"{pc1} vs {pc2}"); axes[r,c].legend(); axes[r,c].grid(alpha=0.3)
    plt.tight_layout(); plt.show()

def plot_significant_pcs(pca, pc_test_results: List[dict], features: List[str]) -> None:
    sig = [i for i,r in enumerate(pc_test_results) if r["ks_pvalue"] < 0.05]
    if not sig:
        print("No significant PCs found (p < 0.05)"); return
    loadings = pca.components_
    fig, axes = plt.subplots(1, len(sig), figsize=(8*len(sig), 6))
    if len(sig) == 1: axes = [axes]
    for j, pc_idx in enumerate(sig):
        df = pd.DataFrame({"feature":features, "loading":loadings[pc_idx],
                           "abs_loading":np.abs(loadings[pc_idx])}).sort_values("abs_loading")
        colors = ["red" if x<0 else "blue" for x in df["loading"]]
        axes[j].barh(range(len(df)), df["loading"], color=colors)
        axes[j].set_yticks(range(len(df))); axes[j].set_yticklabels(df["feature"])
        axes[j].set_xlabel("Loading Value")
        axes[j].set_title(f"PC{pc_idx+1} Feature Loadings\n(p-value: {pc_test_results[pc_idx]['t_pvalue']:.2e})")
        axes[j].axvline(0, color="black", alpha=0.3); axes[j].grid(alpha=0.3, axis="x")
        thr = 0.1
        axes[j].axvline(thr, color="gray", linestyle="--", alpha=0.5)
        axes[j].axvline(-thr, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.show()

    print("\n--- Feature Contributions to Significant PCs ---")
    for pc_idx in sig:
        df = pd.DataFrame({"feature":features,"loading":loadings[pc_idx],
                           "abs_loading":np.abs(loadings[pc_idx])}).sort_values("abs_loading", ascending=False)
        print(f"\nPC{pc_idx+1} (p-value: {pc_test_results[pc_idx]['t_pvalue']:.2e}):")
        print("Top 5 contributing features:")
        for k in range(min(5, len(df))):
            row = df.iloc[k]; print(f"  {row['feature']}: {row['loading']:.3f}")

def plot_pc_statistics(pca_df: pd.DataFrame, pc_test_results: List[dict],
                       variance_ratios: np.ndarray, label1: str, label2: str) -> None:
    if not pc_test_results: return
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    names = [r["PC"] for r in pc_test_results]
    t_p = [r["t_pvalue"] for r in pc_test_results]
    mwu_p = [r["mwu_pvalue"] for r in pc_test_results]
    ks_p = [r["ks_pvalue"] for r in pc_test_results]
    x = np.arange(len(names)); w = 0.25
    axes[0,0].bar(x-w, t_p, w, label="t-test", alpha=0.7)
    axes[0,0].bar(x, mwu_p, w, label="Mann-Whitney U", alpha=0.7)
    axes[0,0].bar(x+w, ks_p, w, label="Kolmogorov-Smirnov", alpha=0.7)
    axes[0,0].axhline(0.05, color="red", linestyle="--", alpha=0.7, label="α=0.05")
    axes[0,0].set_xlabel("Principal Components"); axes[0,0].set_ylabel("P-value")
    axes[0,0].set_title("Statistical Test P-values for PCs")
    axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(names); axes[0,0].legend()
    axes[0,0].grid(alpha=0.3); axes[0,0].set_yscale("log")

    diffs = [abs(r["mean_pop1"] - r["mean_pop2"]) for r in pc_test_results]
    axes[0,1].bar(names, diffs, alpha=0.7, color="orange")
    axes[0,1].set_xlabel("Principal Components"); axes[0,1].set_ylabel("|Mean Difference|")
    axes[0,1].set_title("Absolute Mean Differences between Populations"); axes[0,1].grid(alpha=0.3)

    for idx, pc_num in enumerate([1,2]):
        if pc_num <= len(pc_test_results):
            col = f"PC{pc_num}"
            v1 = pca_df[pca_df["population"] == label1][col].values
            v2 = pca_df[pca_df["population"] == label2][col].values
            ax = axes[1, idx]
            bp = ax.boxplot([v1, v2], labels=[label1, label2], patch_artist=True)
            bp["boxes"][0].set_facecolor("lightblue"); bp["boxes"][1].set_facecolor("lightcoral")
            ax.set_ylabel(f"{col} Score")
            ax.set_title(f"{col} Distribution by Population\n(p-value: {pc_test_results[pc_num-1]['t_pvalue']:.2e})")
            ax.grid(alpha=0.3)

    plt.tight_layout(); plt.show()

    # violin summary
    n = len(pc_test_results)
    if n == 0: return
    fig, axes = plt.subplots(1, min(4, n), figsize=(4 * min(4, n), 6))
    if n == 1: axes = [axes]
    for i in range(min(4, n)):
        col = f"PC{i+1}"
        v1 = pca_df[pca_df["population"] == label1][col].values
        v2 = pca_df[pca_df["population"] == label2][col].values
        parts = axes[i].violinplot([v1, v2], positions=[1, 2], showmeans=True)
        parts["bodies"][0].set_facecolor("lightblue"); parts["bodies"][1].set_facecolor("lightcoral")
        axes[i].set_xticks([1,2]); axes[i].set_xticklabels([label1, label2])
        axes[i].set_ylabel(f"{col} Score")
        p_val = pc_test_results[i]["t_pvalue"]
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        axes[i].set_title(f"{col}\n({variance_ratios[i]:.1%} var, p={p_val:.2e} {sig})")
        axes[i].grid(alpha=0.3)
    plt.tight_layout(); plt.show()

# ---- UMAP plots ----
def plot_umap_results(umap_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    sns.scatterplot(data=umap_df, x="UMAP1", y="UMAP2", hue="population", alpha=0.6, ax=ax)
    ax.set_title("UMAP Projection"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

def plot_umap_statistics(umap_df: pd.DataFrame, umap_test_results: List[dict],
                         n_components: int, label1: str, label2: str) -> None:
    if not umap_test_results: return
    n = min(n_components, 4)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1: axes = [axes]
    for i in range(n):
        col = f"UMAP{i+1}"
        v1 = umap_df[umap_df["population"] == label1][col].values
        v2 = umap_df[umap_df["population"] == label2][col].values
        parts = axes[i].violinplot([v1, v2], positions=[1,2], showmeans=True)
        parts["bodies"][0].set_facecolor("lightblue"); parts["bodies"][1].set_facecolor("lightcoral")
        axes[i].set_xticks([1,2]); axes[i].set_xticklabels([label1, label2])
        axes[i].set_ylabel(col)
        p_val = umap_test_results[i]["mwu_pvalue"]
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        axes[i].set_title(f"{col}\n(p={p_val:.2e} {sig})"); axes[i].grid(alpha=0.3)
    plt.tight_layout(); plt.show()

def plot_umap_feature_importance(umap_df: pd.DataFrame, standardized_data: np.ndarray,
                                 features: List[str], n_components: int) -> None:
    print("\n--- UMAP Feature Analysis ---")
    results = []
    for i in range(min(n_components, 2)):
        col = f"UMAP{i+1}"
        comp_vals = umap_df[col].values
        rows = []
        for j, feat in enumerate(features):
            fv = standardized_data[:, j]
            corr = np.corrcoef(comp_vals, fv)[0,1]
            rows.append({"feature": feat, "correlation": float(corr), "abs_correlation": float(abs(corr))})
        rows.sort(key=lambda x: x["abs_correlation"], reverse=True)
        results.append({"component": col, "correlations": rows})
        print(f"\n{col} - Top 5 correlated features:")
        for k in range(min(5, len(rows))):
            r = rows[k]; print(f"  {r['feature']}: {r['correlation']:.3f}")

    if results:
        n = min(n_components, 2)
        fig, axes = plt.subplots(1, n, figsize=(8*n, 6))
        if n == 1: axes = [axes]
        for idx, res in enumerate(results[:n]):
            corr_df = pd.DataFrame(res["correlations"]).sort_values("correlation")
            colors = ["red" if x < 0 else "blue" for x in corr_df["correlation"]]
            axes[idx].barh(range(len(corr_df)), corr_df["correlation"], color=colors)
            axes[idx].set_yticks(range(len(corr_df))); axes[idx].set_yticklabels(corr_df["feature"])
            axes[idx].set_xlabel("Correlation with UMAP Component")
            axes[idx].set_title(f"{res['component']} Feature Correlations")
            axes[idx].axvline(0, color="black", alpha=0.3); axes[idx].grid(alpha=0.3, axis="x")
            thr = 0.3
            axes[idx].axvline(thr, color="gray", linestyle="--", alpha=0.5)
            axes[idx].axvline(-thr, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout(); plt.show()
