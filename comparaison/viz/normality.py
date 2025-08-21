from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# styling equivalent to your script
plt.style.use("default")
sns.set_palette("husl")


def create_comprehensive_plots(df1: pd.DataFrame, df2: pd.DataFrame, group1: str, group2: str) -> None:
    """
    Create Q–Q plots and overlaid histograms for each feature (same logic).
    """
    feature_names = df1.columns.tolist()
    n_features = len(feature_names)

    # ---- Q-Q plots
    fig, axes = plt.subplots(2, min(4, n_features), figsize=(16, 8))
    if n_features == 1:
        axes = axes.reshape(2, 1)

    for i, feature in enumerate(feature_names[: min(4, n_features)]):
        # group 1
        stats.probplot(df1[feature].dropna(), dist="norm", plot=axes[0, i])
        axes[0, i].set_title(f"{group1} - {feature}")
        axes[0, i].grid(True, alpha=0.3)
        # group 2
        stats.probplot(df2[feature].dropna(), dist="norm", plot=axes[1, i])
        axes[1, i].set_title(f"{group2} - {feature}")
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ---- overlaid histograms
    nrows = (n_features + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 5 * nrows))
    if n_features <= 3:
        axes = axes.reshape(1, -1) if n_features > 1 else [axes]

    for i, feature in enumerate(feature_names):
        row, col = i // 3, i % 3
        ax = axes[row, col] if n_features > 3 else axes[i] if n_features > 1 else axes

        ax.hist(df1[feature].dropna(), alpha=0.6, bins=30, density=True, color="skyblue", label=group1)
        ax.hist(df2[feature].dropna(), alpha=0.6, bins=30, density=True, color="lightcoral", label=group2)
        ax.set_title(f"{feature} Distribution")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # hide empties
    if n_features > 1:
        flat = axes.flat if n_features > 3 else axes
        total_axes = len(list(flat))
        for j in range(n_features, total_axes):
            (axes.flat[j] if n_features > 3 else axes[j]).set_visible(False)

    plt.tight_layout()
    plt.show()
