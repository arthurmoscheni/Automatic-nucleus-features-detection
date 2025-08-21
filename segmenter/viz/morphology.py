from __future__ import annotations
from typing import List, Optional, Tuple, Dict

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from analyzers.morphology import ImageAnalyzer


def annotate_cells_with_features(
    original_img: np.ndarray,
    label_mask: np.ndarray,
    df_features: pd.DataFrame,
    contour_color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    linewidth: int = 1,
    show_image: bool = False,
    save_path: Optional[str] = None
) -> np.ndarray:
    norm = ((original_img - np.min(original_img)) /
            (np.max(original_img) - np.min(original_img)) * 255).astype(np.uint8)
    rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

    for _, row in df_features.iterrows():
        lid = int(row['label'])
        mask = (label_mask == lid).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, cnts, -1, contour_color, thickness=linewidth)
        M = cv2.moments(mask)
        if M['m00'] <= 0:
            continue
        cx = int(M['m10'] / M['m00']); cy = int(M['m01'] / M['m00'])
        txt = f"A:{int(row.get('area', 0))} P:{int(row.get('perimeter', 0))}"
        if 'circularity' in row and not pd.isna(row['circularity']):
            txt += f" C:{row['circularity']:.2f}"
        if 'eccentricity' in row and not pd.isna(row['eccentricity']):
            txt += f" E:{row['eccentricity']:.2f}"
        cv2.putText(rgb, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1, cv2.LINE_AA)

    if show_image:
        plt.figure(figsize=(10, 8)); plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        plt.title('Annotated Cells with Features'); plt.axis('off'); plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, rgb)
        print(f"Annotated image saved to: {save_path}")

    return rgb


def visualize_arcs_on_image(
    analyzer: ImageAnalyzer,
    original_img: np.ndarray,
    label_mask: np.ndarray,
    min_arc_length: int = 40,
    arc_colors: Optional[List[Tuple[int, int, int]]] = None,
    linewidth: int = 2,
    show_image: bool = True,
    save_path: Optional[str] = None
) -> np.ndarray:
    norm = ((original_img - np.min(original_img)) /
            (np.max(original_img) - np.min(original_img)) * 255).astype(np.uint8)
    rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

    if arc_colors is None:
        arc_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 165, 0), (128, 0, 128),
            (255, 192, 203), (0, 128, 0),
        ]

    contours = analyzer.get_contours(label_mask)
    arc_count = 0

    for lid, contour in contours.items():
        flags = analyzer.touches_other(label_mask, contour, lid)
        arcs = analyzer.split_arcs(contour, flags)
        for arc in arcs:
            if len(arc) >= min_arc_length:
                color = arc_colors[arc_count % len(arc_colors)]
                for i in range(len(arc) - 1):
                    pt1 = tuple(map(int, arc[i])); pt2 = tuple(map(int, arc[i+1]))
                    cv2.line(rgb, pt1, pt2, color, linewidth)
                arc_count += 1
                aci = analyzer.arc_circularity_index(arc)
                mid = tuple(map(int, arc[len(arc)//2]))
                cv2.putText(rgb, f"ACI:{aci:.2f}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
                arc_count += 1

    if show_image:
        plt.figure(figsize=(12, 10)); plt.imshow(rgb)
        plt.title(f'Arcs Visualization (min_length={min_arc_length}, {arc_count} arcs shown)')
        plt.axis('off'); plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"Arc visualization saved to: {save_path}")

    print(f"Visualized {arc_count} arcs with length >= {min_arc_length} pixels")
    return rgb


def plot_features(df_features: pd.DataFrame, save_path: Optional[str] = None) -> None:
    if df_features.empty:
        print("No features to plot. Process images first."); return
    cols = ['area', 'perimeter', 'circularity', 'eccentricity', 'solidity', 'aspect_ratio']
    cols = [c for c in cols if c in df_features.columns]
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(1, len(cols), figsize=(15, 4))
    if len(cols) == 1: axs = [axs]
    for i, feat in enumerate(cols):
        sns.histplot(df_features[feat], bins=25, ax=axs[i], kde=True, color='skyblue')
        axs[i].set_title(f"Distribution of {feat}")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

    plt.figure(figsize=(12, 4))
    for i, feat in enumerate(cols):
        plt.subplot(1, len(cols), i + 1)
        sns.boxplot(data=df_features, y=feat, color="lightblue")
        plt.title(f"{feat.capitalize()} Boxplot")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_boxplots.png'))
    plt.show()


def plot_aci_analysis(aci_results: pd.DataFrame, save_path: Optional[str] = None) -> Dict[str, float]:
    if aci_results.empty:
        print("No ACI results to plot. Process images first."); return {}
    vals = aci_results['ACI'].values
    mean = float(np.mean(vals)); std = float(np.std(vals))
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(vals)), vals, color='mediumseagreen', edgecolor='black', label='ACI')
    plt.axhline(mean, color='orange', linestyle='--', label=f'Mean = {mean:.3f}')
    plt.axhline(mean + std, color='red', linestyle=':', label=f'Mean + 1 SD = {mean + std:.3f}')
    plt.axhline(mean - std, color='blue', linestyle=':', label=f'Mean - 1 SD = {mean - std:.3f}')
    plt.xlabel('Arc Index'); plt.ylabel('Arc Circularity Index (ACI)'); plt.title('Scatter Plot of Arc Circularity Index (ACI)')
    plt.legend(); plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    print(f"ACI Mean: {mean:.4f}\nACI Std: {std:.4f}\nACI Min: {vals.min():.4f}\nACI Max: {vals.max():.4f}")
    return {'mean': mean, 'std': std, 'min': float(vals.min()), 'max': float(vals.max())}
