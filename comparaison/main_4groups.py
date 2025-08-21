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

GROUP_LABELS = [
    ("Young", "WT"),
    ("Young", "APP"),
    ("Aged",  "WT"),
    ("Aged",  "APP"),
]

def select_file(kind: str, age: str, genotype: str) -> str:
    """Open a file dialog to pick a CSV for a specific (age, genotype) group."""
    title = f"Select {kind} CSV for {age} {genotype}"
    print(title + "...")
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not path:
        raise FileNotFoundError(f"Missing selection for {kind} / {age} {genotype}")
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
    Prepare morpho features using the feature set derived from (Young WT vs Young APP).
    Other groups are aligned to that feature list.
    """
    m1, m2 = morpho_dfs["morpho_young_wt"], morpho_dfs["morpho_young_app"]
    m3, m4 = morpho_dfs["morpho_aged_wt"], morpho_dfs["morpho_aged_app"]

    m1_p, m2_p, features = prepare_morpho_data(m1, m2, features=None)
    # Align aged groups to the same feature list
    m3_p = m3.dropna()[features]
    m4_p = m4.dropna()[features]

    print("\nMorpho features selected:", len(features))
    return m1_p, m2_p, m3_p, m4_p, features


def prepare_dna_for_4_groups(dna_dfs: dict[str, pd.DataFrame]):
    """
    Prepare DNA features using the feature set derived from (Young WT vs Young APP).
    Other groups are aligned to that feature list.
    """
    d1, d2 = dna_dfs["dna_young_wt"], dna_dfs["dna_young_app"]
    d3, d4 = dna_dfs["dna_aged_wt"], dna_dfs["dna_aged_app"]

    d1_p, d2_p, features = prepare_dna_data(d1, d2, features=None)
    # Align aged groups to the same feature list
    d3_p = d3.dropna()[features]
    d4_p = d4.dropna()[features]

    print("\nDNA features selected:", len(features))
    return d1_p, d2_p, d3_p, d4_p, features


# ---------------------------
# Main runner
# ---------------------------

def main():
    # Hidden Tk root
    root = tk.Tk()
    root.withdraw()

    # ---- Pick files (explicit prompts map cleanly to the analyzer's semantics) ----
    print("=== File Selection ===")
    morpho_paths = {}
    dna_paths = {}
    for age, geno in GROUP_LABELS:
        key_m = f"morpho_{age.lower()}_{geno.lower()}"
        key_d = f"dna_{age.lower()}_{geno.lower()}"
        morpho_paths[key_m] = select_file("morpho", age, geno)
        dna_paths[key_d] = select_file("DNA", age, geno)

    # Group names derived from parent folders (purely informative)
    print("\n=== Group Folders (by selected files) ===")
    for key, p in {**morpho_paths, **dna_paths}.items():
        print(f"{key}: {parent_folder_name(p)}")

    # ---- Load ----
    print("\n=== Loading CSVs ===")
    morpho_dfs = load_csvs(morpho_paths)
    dna_dfs = load_csvs(dna_paths)

    # ---- Prepare (feature selection on Young WT vs Young APP; align others) ----
    print("\n=== Preparing Morphological Data ===")
    m1, m2, m3, m4, morpho_features = prepare_morpho_for_4_groups(morpho_dfs)

    print("\n=== Preparing DNA Data ===")
    d1, d2, d3, d4, dna_features = prepare_dna_for_4_groups(dna_dfs)

    # Optional: outlier removal (example, disabled by default)
    # for k, df in {"m1": m1, "m2": m2, "m3": m3, "m4": m4}.items():
    #     print(f"\nOutlier removal on morpho {k} (IQR, 1.5)...")
    #     locals()[k] = remove_outliers(df, method="iqr", threshold=1.5)
    # for k, df in {"d1": d1, "d2": d2, "d3": d3, "d4": d4}.items():
    #     print(f"\nOutlier removal on DNA {k} (IQR, 1.5)...")
    #     locals()[k] = remove_outliers(df, method="iqr", threshold=1.5)

    # ---- Run analyses (morpho) ----
    print("\n=== Running Morphological Multi-Group Analysis ===")
    plot = True
    morpho_analysis = MultiGroupPipeline(
        young_wt=m1,
        young_app=m2,
        aged_wt=m3,
        aged_app=m4,
        features=morpho_features,
        
    )
    morpho_results = morpho_analysis.run(alpha=0.05, plot=plot)
    # ---- Run analyses (DNA) ----
    print("\n=== Running DNA Multi-Group Analysis ===")
    dna_analysis = MultiGroupPipeline(
        young_wt=d1,
        young_app=d2,
        aged_wt=d3,
        aged_app=d4,
        features=dna_features,
    )
    dna_results = dna_analysis.run(alpha=0.05, plot=plot)
    print("\n=== Multi-Group Analysis Complete ===")
    return morpho_results, dna_results


if __name__ == "__main__":
    main()
