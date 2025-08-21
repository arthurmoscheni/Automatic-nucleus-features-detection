import numpy as np
from scipy import stats

def prepare_morpho_data(morpho1, morpho2, features=None):
    """Load and prepare morpho data from CSV files."""
    df1 = morpho1.dropna()
    df2 = morpho2.dropna()

    if features is None:
        exclude = [
            'cell_id', 'image_id', 'label', 'circularity', 'threshold', 
            'boundary_pixels', 'invag_pixels', 'orientation', 'major_axis_length', 
            'minor_axis_length', 'centroid-0', 'centroid-1', 'aspect_ratio', 
            'total_intensity', 'perimeter'
        ]
        features = [
            col for col in df1.select_dtypes(include=np.number).columns 
            if col not in exclude
        ]

    return df1[features], df2[features], features

def prepare_dna_data(dna1, dna2, features=None):
    """Load and prepare DNA data from CSV files."""
    df1 = dna1.dropna()
    df2 = dna2.dropna()

    if features is None:
        exclude = [
            'cell_id', 'image_id', 'std_distance_from_center', 'intensity_skewness', 
            'intensity_kurtosis', 'intensity_entropy', 'lbp_uniformity', 
            'cluster_size_cv', 'intensity_std', 'intensity_cv'
        ]
        features = [
            col for col in df1.select_dtypes(include=np.number).columns 
            if col not in exclude
        ]

    return df1[features], df2[features], features

def remove_outliers(df, method='iqr', threshold=1.5):
    """
    Remove outliers from a DataFrame using IQR or Z-score method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    method : str
        Method to use ('iqr' or 'zscore')
    threshold : float
        Threshold for outlier detection (1.5 for IQR, 3 for Z-score typically)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=np.number).columns
    
    if method == 'iqr':
        for column in numeric_cols:
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_clean = df_clean[
                (df_clean[column] >= lower_bound) & 
                (df_clean[column] <= upper_bound)
            ]
    
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(df_clean[numeric_cols]))
        df_clean = df_clean[(z_scores < threshold).all(axis=1)]
    
    _print_outlier_stats(df, df_clean, method, numeric_cols)
    return df_clean

def _print_outlier_stats(original_df, cleaned_df, method, numeric_cols):
    """Print outlier removal statistics."""
    original_count = len(original_df)
    final_count = len(cleaned_df)
    removed_count = original_count - final_count
    
    print(f"Outlier removal using {method} method:")
    print(f"Original rows: {original_count}")
    print(f"Rows after removal: {final_count}")
    print(f"Removed rows: {removed_count} ({removed_count/original_count*100:.2f}%)")
    
    print("\nDistribution comparison (mean ± std):")
    for col in numeric_cols:
        orig_mean, orig_std = original_df[col].mean(), original_df[col].std()
        clean_mean, clean_std = cleaned_df[col].mean(), cleaned_df[col].std()
        print(f"{col}:")
        print(f"  Before: {orig_mean:.3f} ± {orig_std:.3f}")
        print(f"  After:  {clean_mean:.3f} ± {clean_std:.3f}")