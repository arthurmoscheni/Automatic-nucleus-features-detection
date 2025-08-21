from __future__ import annotations
import os
import pandas as pd



def save_morpho_combined_results(morpho_results: dict, output_dir: str) -> pd.DataFrame:
    """
    Writes the same CSVs your `save_analysis_results` function produced.
    """
    combined_features = morpho_results['combined_features_df']
    all_aci_results = morpho_results['combined_aci_df']
    wrinkles_results = morpho_results['combined_wrinkles_df']
    signal_intensities = morpho_results['combined_signal_intensities_df']

    print("\nSaving analysis results...")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Combined features saved to: {os.path.join(output_dir, 'combined_features.csv')}")
    print(f"ACI results saved to: {os.path.join(output_dir, 'all_aci_results.csv')}")
    print(f"Wrinkles results saved to: {os.path.join(output_dir, 'combined_wrinkles.csv')}")
    print(f"Signal intensities results saved to: {os.path.join(output_dir, 'combined_signal_intensities.csv')}")

    combined_path = os.path.join(output_dir, "combined_df.csv")
    combined_df = pd.merge(combined_features, all_aci_results, on=['image_id', 'label'], how='left')
    combined_df = pd.merge(combined_df, wrinkles_results, on=['image_id', 'label'], how='left')
    combined_df = pd.merge(combined_df, signal_intensities, on=['image_id', 'label'], how='left')

    combined_df.to_csv(combined_path, index=False)
    print(f"Combined features, ACI & wrinkles results saved to: {combined_path}")
    return combined_df


def save_dna_results(dna_df: pd.DataFrame, output_dir: str, filename: str = "dna_features.csv") -> str:
    """Write combined DNA features to CSV."""
    print("\nSaving analysis results...")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    dna_df.to_csv(path, index=False)
    print(f"DNA features saved to: {path}")
    return path
