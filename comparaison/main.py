import pandas as pd
from tkinter import filedialog
import tkinter as tk
import os
from normality import NormalityTester
from univariate import UnivariateComparison
from multivariate import GlobalMultivariateTests
from UMAP import PCA_UMAP
from utils import prepare_morpho_data, prepare_dna_data, remove_outliers


def select_files():
    """Select the required CSV files through file dialogs."""
    root = tk.Tk()
    root.withdraw()
    
    files = {}
    
    print("Select the first morpho CSV file (morpho1):")
    files['morpho1'] = filedialog.askopenfilename(
        title="Select first morpho CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    print("Select the first DNA CSV file (dna1):")
    files['dna1'] = filedialog.askopenfilename(
        title="Select first DNA CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    print("Select the second morpho CSV file (morpho2):")
    files['morpho2'] = filedialog.askopenfilename(
        title="Select second morpho CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    print("Select the second DNA CSV file (dna2):")
    files['dna2'] = filedialog.askopenfilename(
        title="Select second DNA CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    return files


def get_group_names(file_paths):
    """Extract group names from folder paths."""
    group1_name = os.path.basename(os.path.dirname(file_paths['morpho1']))
    group2_name = os.path.basename(os.path.dirname(file_paths['morpho2']))
    
    print(f"Group 1: {group1_name}")
    print(f"Group 2: {group2_name}")
    
    return group1_name, group2_name


def load_and_prepare_data(file_paths):
    """Load and prepare morpho and DNA data."""
    # Load raw data
    morpho1_raw = pd.read_csv(file_paths['morpho1'])
    morpho2_raw = pd.read_csv(file_paths['morpho2'])
    dna1_raw = pd.read_csv(file_paths['dna1'])
    dna2_raw = pd.read_csv(file_paths['dna2'])
    
    # Prepare data
    morpho1, morpho2, morpho_features = prepare_morpho_data(morpho1_raw, morpho2_raw, features=None)
    dna1, dna2, dna_features = prepare_dna_data(dna1_raw, dna2_raw, features=None)
    
    # Apply filters
    morpho1 = morpho1[morpho1['ACI'] >= 0.6]
    morpho2 = morpho2[morpho2['ACI'] >= 0.6]
    
    print(f"Loaded morpho1 with shape: {morpho1.shape}")
    print(f"Loaded morpho2 with shape: {morpho2.shape}")
    print(f"Loaded dna1 with shape: {dna1.shape}")
    print(f"Loaded dna2 with shape: {dna2.shape}")
    
    return (morpho1, morpho2, morpho_features), (dna1, dna2, dna_features)


def run_analysis(data1, data2, features, group1_name, group2_name, data_type):
    """Run complete analysis pipeline for a dataset."""
    print(f"\n=== {data_type.upper()} ANALYSIS ===")
    
    # Remove mean_intensity from features for some analyses
    features_filtered = [f for f in features if f != 'mean_intensity']
    
    # Normality tests
    normality_tester = NormalityTester(data1, data2, group1=group1_name, group2=group2_name)
    normality_results = normality_tester.run_all_tests()
    
    # PCA/UMAP analysis
    umap_analyzer = PCA_UMAP(data1, data2, group1=group1_name, group2=group2_name, features=features_filtered)
    pca_results, umap_results = umap_analyzer.run_all()
    print(f"UMAP results for {data_type}:")
    print(umap_results)
    
    # Univariate comparison
    univariate_tester = UnivariateComparison(data1, data2, group1=group1_name, group2=group2_name, features=features)
    univariate_results = univariate_tester.run_all(visualization=True)
    print(f"Univariate comparison results for {data_type}:")
    print(univariate_results)
    
    # Multivariate comparison
    multivariate_tester = GlobalMultivariateTests(data1, data2, group1=group1_name, group2=group2_name, features=features_filtered)
    multivariate_results = multivariate_tester.run_all(alpha=0.05, n_permutations=0, random_state=42)
    print(f"Multivariate comparison results for {data_type}:")
    print(multivariate_results)
    
    return {
        'normality': normality_results,
        'pca': pca_results,
        'umap': umap_results,
        'univariate': univariate_results,
        'multivariate': multivariate_results
    }


def main():
    """Main execution function."""
    # Select files
    file_paths = select_files()
    
    # Get group names
    group1_name, group2_name = get_group_names(file_paths)
    
    # Load and prepare data
    (morpho1, morpho2, morpho_features), (dna1, dna2, dna_features) = load_and_prepare_data(file_paths)
    
    # Run morpho analysis
    morpho_results = run_analysis(morpho1, morpho2, morpho_features, group1_name, group2_name, "morpho")
    
    # Run DNA analysis
    dna_results = run_analysis(dna1, dna2, dna_features, group1_name, group2_name, "DNA")
    
    return morpho_results, dna_results


if __name__ == "__main__":
    morpho_results, dna_results = main()