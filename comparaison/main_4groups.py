"""
Four-group comparison entrypoint.

Usage (from repo root):
    python -m comparaison.main4groups
"""

import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog

# utils
from utils.data_prep import (
    prepare_morpho_data,
    prepare_dna_data,
    remove_outliers,  # optional
)

from pipelines.multigroup_pipeline import MultiGroupPipeline


# ---------------------------
# File selection conveniences
# ---------------------------

def select_file(kind: str, group_num: int) -> str:
    """Open a file dialog to pick a CSV for a specific group."""
    title = f"Select {kind} CSV for Group {group_num}"
    print(title + "...")
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not path:
        raise FileNotFoundError(f"Missing selection for {kind} / Group {group_num}")
    return path


def parent_folder_name(path: str) -> str:
    """Return the folder name one level above the file."""
    return os.path.basename(os.path.dirname(path)) or ""


# ---------------------------
# Loading & preparation
# ---------------------------

def load_csvs(paths: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Load all CSVs into dataframes."""
    dfs = {}
    for key, p in paths.items():
        df = pd.read_csv(p)
        dfs[key] = df
        print(f"Loaded {key}: {df.shape}")
    return dfs


def prepare_morpho_for_4_groups(morpho_dfs: dict[str, pd.DataFrame]):
    """
    Prepare morpho features using the feature set derived from the first two groups.
    Other groups are aligned to that feature list.
    """
    group_keys = list(morpho_dfs.keys())
    m1, m2 = morpho_dfs[group_keys[0]], morpho_dfs[group_keys[1]]
    m3, m4 = morpho_dfs[group_keys[2]], morpho_dfs[group_keys[3]]

    m1_p, m2_p, features = prepare_morpho_data(m1, m2, features=None)
    # Align other groups to the same feature list
    m3_p = m3.dropna()[features]
    m4_p = m4.dropna()[features]

    print("\nMorpho features selected:", len(features))
    
    # Return as dictionary with group keys
    groups_dict = {
        group_keys[0]: m1_p,
        group_keys[1]: m2_p,
        group_keys[2]: m3_p,
        group_keys[3]: m4_p
    }
    
    return groups_dict, features


def prepare_dna_for_4_groups(dna_dfs: dict[str, pd.DataFrame]):
    """
    Prepare DNA features using the feature set derived from the first two groups.
    Other groups are aligned to that feature list.
    """
    group_keys = list(dna_dfs.keys())
    d1, d2 = dna_dfs[group_keys[0]], dna_dfs[group_keys[1]]
    d3, d4 = dna_dfs[group_keys[2]], dna_dfs[group_keys[3]]

    d1_p, d2_p, features = prepare_dna_data(d1, d2, features=None)
    # Align other groups to the same feature list
    d3_p = d3.dropna()[features]
    d4_p = d4.dropna()[features]

    print("\nDNA features selected:", len(features))
    
    # Return as dictionary with group keys
    groups_dict = {
        group_keys[0]: d1_p,
        group_keys[1]: d2_p,
        group_keys[2]: d3_p,
        group_keys[3]: d4_p
    }
    
    return groups_dict, features


# ---------------------------
# Main runner
# ---------------------------

def main():
    # Hidden Tk root
    root = tk.Tk()
    root.withdraw()

    # ---- Pick files for 4 groups ----
    print("=== File Selection ===")
    morpho_paths = {}
    dna_paths = {}
    group_labels = []
    
    for i in range(4):
        group_num = i + 1
        
        # Select morpho file
        morpho_path = select_file("morpho", group_num)
        group_label = parent_folder_name(morpho_path)
        group_labels.append(group_label)
        
        # Select DNA file
        dna_path = select_file("DNA", group_num)
        
        # Store with group label as key
        morpho_paths[group_label] = morpho_path
        dna_paths[group_label] = dna_path

    # Display selected groups
    print("\n=== Selected Groups ===")
    for i, label in enumerate(group_labels, 1):
        print(f"Group {i}: {label}")

    # ---- Load ----
    print("\n=== Loading CSVs ===")
    morpho_dfs = load_csvs(morpho_paths)
    dna_dfs = load_csvs(dna_paths)

    # ---- Prepare (feature selection on first two groups; align others) ----
    print("\n=== Preparing Morphological Data ===")
    morpho_groups, morpho_features = prepare_morpho_for_4_groups(morpho_dfs)

    print("\n=== Preparing DNA Data ===")
    dna_groups, dna_features = prepare_dna_for_4_groups(dna_dfs)

    # Optional: outlier removal (example, disabled by default)
    # for k, df in morpho_groups.items():
    #     print(f"\nOutlier removal on morpho {k} (IQR, 1.5)...")
    #     morpho_groups[k] = remove_outliers(df, method="iqr", threshold=1.5)
    # for k, df in dna_groups.items():
    #     print(f"\nOutlier removal on DNA {k} (IQR, 1.5)...")
    #     dna_groups[k] = remove_outliers(df, method="iqr", threshold=1.5)

    # ---- Run analyses (morpho) ----
    print("\n=== Running Morphological Multi-Group Analysis ===")
    plot = True
    morpho_analysis = MultiGroupPipeline(
        groups=morpho_groups,
        features=morpho_features
    )
    morpho_results = morpho_analysis.run(alpha=0.05, plot=plot)
    
    # ---- Run analyses (DNA) ----
    print("\n=== Running DNA Multi-Group Analysis ===")
    dna_analysis = MultiGroupPipeline(
        groups=dna_groups,
        features=dna_features
    )
    dna_results = dna_analysis.run(alpha=0.05, plot=plot)
    
    print("\n=== Multi-Group Analysis Complete ===")
    return morpho_results, dna_results


if __name__ == "__main__":
    main()
