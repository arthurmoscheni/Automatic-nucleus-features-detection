"""
Two-group comparison entrypoint.

Usage (recommended):
    python -m comparaison.main
"""

import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog

from pipelines.normality import run_normality_tests
from pipelines.dimred import run_all_dimred
from pipelines.univariate import run_univariate
from pipelines.multivariate import run_multivariate

# --- utils ---
from utils.data_prep import (
    prepare_morpho_data,
    prepare_dna_data,
    remove_outliers,
)


# ======================
# File selection helpers
# ======================

def select_files() -> dict:
    """Open file dialogs to choose the four CSVs."""
    root = tk.Tk()
    root.withdraw()

    files = {}

    print("Select the FIRST morpho CSV (morpho1):")
    files["morpho1"] = filedialog.askopenfilename(
        title="Select FIRST morpho CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    print("Select the FIRST DNA CSV (dna1):")
    files["dna1"] = filedialog.askopenfilename(
        title="Select FIRST DNA CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    print("Select the SECOND morpho CSV (morpho2):")
    files["morpho2"] = filedialog.askopenfilename(
        title="Select SECOND morpho CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    print("Select the SECOND DNA CSV (dna2):")
    files["dna2"] = filedialog.askopenfilename(
        title="Select SECOND DNA CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    # Optional: basic sanity check
    for k, v in files.items():
        if not v:
            raise FileNotFoundError(f"Missing selection for {k}.")

    return files


def infer_group_names(file_paths: dict) -> tuple[str, str]:
    """Derive group names from the parent folder of morpho CSVs."""
    g1 = os.path.basename(os.path.dirname(file_paths["morpho1"])) or "Group1"
    g2 = os.path.basename(os.path.dirname(file_paths["morpho2"])) or "Group2"
    print(f"\nDetected groups →  Group 1: {g1} | Group 2: {g2}")
    return g1, g2


# ======================
# Data loading & prep
# ======================

def load_and_prepare_data(file_paths: dict,
                          min_aci: float = 0.6,
                          outlier_method: str | None = None,
                          outlier_threshold: float = 1.5):
    """Load CSVs, prepare features, filter by ACI, and (optionally) remove outliers."""
    # Load
    morpho1_raw = pd.read_csv(file_paths["morpho1"])
    morpho2_raw = pd.read_csv(file_paths["morpho2"])
    dna1_raw = pd.read_csv(file_paths["dna1"])
    dna2_raw = pd.read_csv(file_paths["dna2"])

    # Prepare feature sets
    morpho1, morpho2, morpho_features = prepare_morpho_data(morpho1_raw, morpho2_raw, features=None)
    dna1, dna2, dna_features = prepare_dna_data(dna1_raw, dna2_raw, features=None)

    # Optional: ACI filter for morpho (if present)
    if "ACI" in morpho1.columns and "ACI" in morpho2.columns and min_aci is not None:
        morpho1 = morpho1[morpho1["ACI"] >= min_aci]
        morpho2 = morpho2[morpho2["ACI"] >= min_aci]

    # Optional: Outlier removal (applied per group, per dataset)
    if outlier_method:
        print(f"\nApplying outlier removal ({outlier_method}, threshold={outlier_threshold}) on MORPHO…")
        morpho1 = remove_outliers(morpho1, method=outlier_method, threshold=outlier_threshold)
        morpho2 = remove_outliers(morpho2, method=outlier_method, threshold=outlier_threshold)

        print(f"\nApplying outlier removal ({outlier_method}, threshold={outlier_threshold}) on DNA…")
        dna1 = remove_outliers(dna1, method=outlier_method, threshold=outlier_threshold)
        dna2 = remove_outliers(dna2, method=outlier_method, threshold=outlier_threshold)

    print(f"\nShapes after prep/filtering:")
    print(f"  morpho1: {morpho1.shape} | morpho2: {morpho2.shape}")
    print(f"  dna1:    {dna1.shape}    | dna2:    {dna2.shape}")

    return (morpho1, morpho2, morpho_features), (dna1, dna2, dna_features)


# ======================
# Analysis runner
# ======================

def run_two_group_analysis(df1: pd.DataFrame,
                           df2: pd.DataFrame,
                           features: list[str],
                           group1_name: str,
                           group2_name: str,
                           dataset_name: str) -> dict:
    """
    Execute the full two-group workflow on one dataset (morpho or DNA).
    Returns a dict of results objects (you can serialize as needed).
    """
    print(f"\n=== {dataset_name.upper()} ANALYSIS ({group1_name} vs {group2_name}) ===")

    # Some analyses exclude mean_intensity (if present)
    features_filtered = [f for f in features if f != "mean_intensity"]

    # 1) Normality tests
    normality = run_normality_tests(df1, df2, group1_name, group2_name, plot=True)

    # 2) PCA + UMAP
    pca_results, umap_results = run_all_dimred(df1, df2, group1_name, group2_name, features_filtered)

    # 3) Univariate tests
    univariate_results = run_univariate(df1, df2, group1_name, group2_name, features)

    # 4) Multivariate tests
    multivariate_results = run_multivariate(df1, df2, group1_name, group2_name, features_filtered)

    return {
        "normality": normality,
        "pca_results": pca_results,
        "umap_results": umap_results,
        "univariate": univariate_results,
        "multivariate": multivariate_results,
        "features_used": features,
        "features_filtered": features_filtered,
    }


# ======================
# Main
# ======================

def main():
    # ---- 1) Select files & group names ----
    file_paths = select_files()
    group1_name, group2_name = infer_group_names(file_paths)

    # ---- 2) Load & prepare ----
    # Toggle outlier removal by setting outlier_method to 'iqr' or 'zscore' (or None to disable)
    (morpho1, morpho2, morpho_features), (dna1, dna2, dna_features) = load_and_prepare_data(
        file_paths,
        min_aci=0.6,
        outlier_method=None,      # 'iqr' | 'zscore' | None
        outlier_threshold=1.5,    # 1.5 for IQR; ~3.0 for z-score
    )

    # ---- 3) Run analyses ----
    morpho_results = run_two_group_analysis(
        morpho1, morpho2, morpho_features, group1_name, group2_name, dataset_name="morpho"
    )

    dna_results = run_two_group_analysis(
        dna1, dna2, dna_features, group1_name, group2_name, dataset_name="DNA"
    )

    print("\nAll analyses complete.")
    return {"morpho": morpho_results, "dna": dna_results}


if __name__ == "__main__":
    main()
