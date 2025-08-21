import pandas as pd
from tkinter import filedialog
import tkinter as tk
import os
from utils import prepare_morpho_data, prepare_dna_data, remove_outliers
from multicomparaison import MultiGroupAnalysis

def select_file(file_type, group_number):
    """Select a CSV file using file dialog."""
    print(f"Select the {file_type} CSV file for group {group_number}:")
    return filedialog.askopenfilename(
        title=f"Select {file_type} CSV file for group {group_number}",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

def get_folder_name(file_path):
    """Extract folder name from file path."""
    return os.path.basename(os.path.dirname(file_path))

def main():
    # Initialize tkinter (hidden window)
    root = tk.Tk()
    root.withdraw()
    
    # File selection for 4 groups (morpho and DNA files)
    print("=== File Selection ===")
    file_paths = {}
    
    for group in range(1, 5):
        file_paths[f'morpho{group}'] = select_file("morpho", group)
        file_paths[f'dna{group}'] = select_file("DNA", group)
    
    # Extract group names from folder structure
    print("\n=== Group Names ===")
    group_names = {}
    for key, path in file_paths.items():
        folder_name = get_folder_name(path)
        group_names[key] = folder_name
        print(f"{key.upper()} folder: {folder_name}")
    
    # Load morphological data
    print("\n=== Loading Morphological Data ===")
    morpho_dataframes = {}
    for group in range(1, 5):
        path = file_paths[f'morpho{group}']
        morpho_dataframes[f'morpho{group}'] = pd.read_csv(path)
        print(f"Loaded morpho{group} with shape: {morpho_dataframes[f'morpho{group}'].shape}")
    
    # Load DNA data
    print("\n=== Loading DNA Data ===")
    dna_dataframes = {}
    for group in range(1, 5):
        path = file_paths[f'dna{group}']
        dna_dataframes[f'dna{group}'] = pd.read_csv(path)
        print(f"Loaded dna{group} with shape: {dna_dataframes[f'dna{group}'].shape}")
    
    # Prepare morphological data (groups 1&2, then 3&4)
    print("\n=== Preparing Morphological Data ===")
    morpho1, morpho2, morpho_features = prepare_morpho_data(
        morpho_dataframes['morpho1'], 
        morpho_dataframes['morpho2'], 
        features=None
    )
    morpho3, morpho4, _ = prepare_morpho_data(
        morpho_dataframes['morpho3'], 
        morpho_dataframes['morpho4'], 
        features=None
    )
    
    # Prepare DNA data (groups 1&2, then 3&4)
    print("\n=== Preparing DNA Data ===")
    dna1, dna2, dna_features = prepare_dna_data(
        dna_dataframes['dna1'], 
        dna_dataframes['dna2'], 
        features=None
    )
    dna3, dna4, _ = prepare_dna_data(
        dna_dataframes['dna3'], 
        dna_dataframes['dna4'], 
        features=None
    )
    
    # Optional filtering (currently commented out)
    # morpho1, morpho2, morpho3, morpho4 = apply_morpho_filters(morpho1, morpho2, morpho3, morpho4)
    
    # Run morphological analysis
    print("\n=== Running Morphological Analysis ===")
    morpho_analyzer = MultiGroupAnalysis(morpho1, morpho2, morpho3, morpho4, morpho_features)
    morpho_results = morpho_analyzer.run_all()
    
    # Run DNA analysis
    print("\n=== Running DNA Analysis ===")
    dna_analyzer = MultiGroupAnalysis(dna1, dna2, dna3, dna4, dna_features)
    dna_results = dna_analyzer.run_all()
    
    print("\n=== Analysis Complete ===")
    return morpho_results, dna_results

def apply_morpho_filters(morpho1, morpho2, morpho3, morpho4):
    """Apply ACI and solidity filters to morphological data."""
    filter_condition = lambda df: df[(df['ACI'] >= 0.65) & (df['solidity'] >= 0.75)]
    
    morpho1_filtered = filter_condition(morpho1)
    morpho2_filtered = filter_condition(morpho2)
    morpho3_filtered = filter_condition(morpho3)
    morpho4_filtered = filter_condition(morpho4)
    
    print(f"After filtering - morpho1 shape: {morpho1_filtered.shape}")
    print(f"After filtering - morpho2 shape: {morpho2_filtered.shape}")
    print(f"After filtering - morpho3 shape: {morpho3_filtered.shape}")
    print(f"After filtering - morpho4 shape: {morpho4_filtered.shape}")
    
    return morpho1_filtered, morpho2_filtered, morpho3_filtered, morpho4_filtered

if __name__ == "__main__":
    morpho_results, dna_results = main()
