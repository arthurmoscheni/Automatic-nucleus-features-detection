from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import morphology

from analyzers.dna import DNADistributionAnalyzer


def visualize_clusters_overlay(
    analyzer: DNADistributionAnalyzer,
    threshold_method: str = 'original',
    show_image: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize all cells: Hoechst (gray), boundaries (cyan), clusters (orange).
    (Logic mirrors the original method; just moved out of the class.)
    """
    hoechst = analyzer.hoechst
    mask = analyzer.mask

    unique_cells = np.unique(mask)
    unique_cells = unique_cells[unique_cells > 0]
    if len(unique_cells) == 0:
        print("No cells found in the mask")
        return

    all_clusters = np.zeros_like(hoechst)
    all_cell_boundaries = np.zeros_like(hoechst, dtype=bool)
    total_clusters = 0

    print(f"Processing {len(unique_cells)} cells...")
    for cid in unique_cells:
        cell_mask = (mask == cid).astype(int)
        if np.sum(cell_mask) == 0:
            continue

        hoechst_binary = analyzer.preprocess_hoechst_adaptive(cell_mask, method=threshold_method)
        labeled_clusters, cluster_props = analyzer.detect_dna_clusters(hoechst_binary, use_watershed=True)

        if labeled_clusters is not None and np.max(labeled_clusters) > 0:
            all_clusters[labeled_clusters > 0] = 1
            total_clusters += len(cluster_props)

        cell_boundary = cell_mask - morphology.binary_erosion(cell_mask, morphology.disk(1))
        all_cell_boundaries |= cell_boundary.astype(bool)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(hoechst, cmap='gray'); axes[0].set_title('Original Hoechst Image'); axes[0].axis('off')

    axes[1].imshow(hoechst, cmap='gray')
    axes[1].contour(all_cell_boundaries.astype(int), colors='cyan', linewidths=1)
    axes[1].set_title(f'All Cell Boundaries\n({len(unique_cells)} cells)'); axes[1].axis('off')

    axes[2].imshow(hoechst, cmap='gray')
    if np.max(all_clusters) > 0:
        overlay = np.zeros((*hoechst.shape, 4))
        overlay[all_clusters > 0] = [1, 0.5, 0, 0.7]  # orange RGBA
        axes[2].imshow(overlay)
        axes[2].set_title(f'All DNA Clusters Highlighted\n({total_clusters} clusters total)')
    else:
        axes[2].set_title('No Clusters Detected')
    axes[2].contour(all_cell_boundaries.astype(int), colors='cyan', linewidths=1, alpha=0.8)
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_image:
        plt.show()
    else:
        plt.close()

    print("\nWhole Image DNA Cluster Analysis:")
    print(f"  - Total cells processed: {len(unique_cells)}")
    print(f"  - Total clusters detected: {total_clusters}")
    print(f"  - Average clusters per cell: {total_clusters / len(unique_cells):.2f}")


def quick_dna_insights(results_df: pd.DataFrame, save_path: Optional[str] = None, show: bool = True) -> None:
    """1×3 panel: cluster count, total DNA area, localization."""
    if results_df.empty:
        print("No data to visualize")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(results_df['num_clusters'], bins=15, alpha=0.7, color='skyblue')
    axes[0].set_title(f'DNA Clusters per Cell\n(Mean: {results_df["num_clusters"].mean():.1f})')
    axes[0].set_xlabel('Number of Clusters')

    axes[1].scatter(results_df['num_clusters'], results_df['total_dna_area_um2'], alpha=0.6, color='coral')
    axes[1].set_xlabel('Number of Clusters'); axes[1].set_ylabel('Total DNA Area (μm²)')
    axes[1].set_title('Clusters vs DNA Area')

    axes[2].hist(results_df['mean_distance_from_center'], bins=15, alpha=0.7, color='lightgreen')
    axes[2].set_title('DNA Localization\n(0=center, 1=periphery)')
    axes[2].set_xlabel('Distance from Center')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
