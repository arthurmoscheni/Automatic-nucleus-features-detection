import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ks_2samp, ttest_ind
from statsmodels.stats.multitest import multipletests


class UnivariateComparison:
    def __init__(self, df1, df2, group1=None, group2=None, features=None):
        """
        Initialize with two groups for comparison.
        
        Parameters:
        df1, df2: DataFrames to compare
        group1, group2: str, names for the groups
        features: list, features to analyze (if None, uses all columns)
        """
        self.df1 = df1.copy()
        self.df2 = df2.copy()
        self.features = features if features is not None else self.df1.columns.tolist()
        self.group1_name = group1 if group1 is not None else 'Pop1'
        self.group2_name = group2 if group2 is not None else 'Pop2'
        
        # Replace 'ratio' with 'wrinkle_ratio' for consistency
        self._rename_ratio_column()
        
    def _rename_ratio_column(self):
        """Replace 'ratio' with 'wrinkle_ratio' in dataframes and features list."""
        if 'ratio' in self.df1.columns:
            self.df1 = self.df1.rename(columns={'ratio': 'wrinkle_ratio'})
        if 'ratio' in self.df2.columns:
            self.df2 = self.df2.rename(columns={'ratio': 'wrinkle_ratio'})
        if 'ratio' in self.features:
            self.features = [feat if feat != 'ratio' else 'wrinkle_ratio' for feat in self.features]

    def permutation_test(self, x1, x2, n_permutations=10000):
        """Perform permutation test for difference in means."""
        observed_diff = np.mean(x1) - np.mean(x2)
        combined = np.concatenate([x1, x2])
        n1 = len(x1)
        
        perm_diffs = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_x1 = combined[:n1]
            perm_x2 = combined[n1:]
            perm_diff = np.mean(perm_x1) - np.mean(perm_x2)
            perm_diffs.append(perm_diff)
        
        perm_diffs = np.array(perm_diffs)
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        return observed_diff, p_value

    def _calculate_rank_biserial_correlation(self, x1, x2, mwu_stat):
        """Calculate rank-biserial correlation for effect size."""
        n1, n2 = len(x1), len(x2)
        return (2 * mwu_stat) / (n1 * n2) - 1

    def _get_effect_size_label(self, correlation):
        """Get effect size label based on correlation magnitude."""
        abs_corr = abs(correlation)
        if abs_corr >= 0.5:
            return 'Large'
        elif abs_corr >= 0.3:
            return 'Medium'
        else:
            return 'Small'

    def comprehensive_tests(self, alpha=0.05):
        """
        Perform comprehensive statistical tests with FDR correction.
        
        Args:
            alpha: significance level for FDR correction
            
        Returns:
            results_df: DataFrame with test results and FDR-corrected p-values
        """
        results = []
        p_values_mwu = []
        p_values_ks = []
        
        for feat in self.features:
            x1 = self.df1[feat].dropna().values
            x2 = self.df2[feat].dropna().values
            
            # Statistical tests
            mwu_stat, mwu_p = mannwhitneyu(x1, x2, alternative='two-sided')
            ks_stat, ks_p = ks_2samp(x1, x2)
            perm_diff, perm_p = self.permutation_test(x1, x2)
            
            # Effect size calculation
            rb_corr = self._calculate_rank_biserial_correlation(x1, x2, mwu_stat)
            
            print(f"Feature: {feat}, Permutation diff: {perm_diff:.4f}, Permutation p-value: {perm_p:.2e}")
            
            results.append({
                'feature': feat,
                'mwu_statistic': mwu_stat,
                'perm_pvalue': perm_p,
                'mwu_pvalue': mwu_p,
                'rank_biserial_correlation': rb_corr,
                'rb_effect_size': self._get_effect_size_label(rb_corr)
            })
            
            p_values_mwu.append(mwu_p)
            p_values_ks.append(ks_p)
        
        # FDR correction
        _, mwu_pvals_fdr, _, _ = multipletests(p_values_mwu, alpha=alpha, method='fdr_bh')
        _, ks_pvals_fdr, _, _ = multipletests(p_values_ks, alpha=alpha, method='fdr_bh')
        
        # Add FDR-corrected p-values to results
        for i, result in enumerate(results):
            result['mwu_pvalue_fdr'] = mwu_pvals_fdr[i]
            result['ks_pvalue_fdr'] = ks_pvals_fdr[i]
            result['mwu_significant_fdr'] = mwu_pvals_fdr[i] < alpha
            result['ks_significant_fdr'] = ks_pvals_fdr[i] < alpha
        
        return pd.DataFrame(results)

    def plot_pvalue_histogram(self, results_df, test_type='mwu', corrected=True, bins=20, figsize=(10, 6)):
        """Plot p-values as a histogram to assess their distribution."""
        p_col = f'{test_type}_pvalue_fdr' if corrected else f'{test_type}_pvalue'
        title_suffix = '(FDR Corrected)' if corrected else '(Uncorrected)'
        
        p_values = results_df[p_col].values
        alpha_threshold = 0.05
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Create histogram
        n, bins_edges, patches = ax.hist(p_values, bins=bins, alpha=0.7, 
                                       color='skyblue', edgecolor='black')
        
        # Color significant p-values differently
        for patch, bin_start, bin_end in zip(patches, bins_edges[:-1], bins_edges[1:]):
            if bin_end <= alpha_threshold:
                patch.set_facecolor('lightcoral')
        
        # Formatting
        test_names = {'mwu': 'Mann-Whitney U', 'ks': 'Kolmogorov-Smirnov'}
        ax.axvline(x=alpha_threshold, color='red', linestyle='--', label=f'α = {alpha_threshold}')
        ax.set_xlabel('P-values')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{test_names.get(test_type, test_type)} P-value Distribution {title_suffix}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        n_significant = np.sum(p_values < alpha_threshold)
        n_total = len(p_values)
        stats_text = f'Significant: {n_significant}/{n_total} ({n_significant/n_total*100:.1f}%)'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

    def plot_violinplots(self, results_df, alpha=0.05, max_plots_per_row=4, figsize_per_plot=(4, 3)):
        """Create violin plots for all features comparing the two populations."""
        n_features = len(self.features)
        n_rows = (n_features + max_plots_per_row - 1) // max_plots_per_row
        
        fig, axes = plt.subplots(n_rows, max_plots_per_row, 
                                figsize=(figsize_per_plot[0] * max_plots_per_row, 
                                        figsize_per_plot[1] * n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()
        
        for i, feat in enumerate(self.features):
            ax = axes_flat[i]
            
            # Create violin plot
            data1 = self.df1[feat].dropna().values
            data2 = self.df2[feat].dropna().values
            parts = ax.violinplot([data1, data2], positions=[1, 2], 
                                showmeans=True, showmedians=False)
            
            # Color the violin plots
            colors = ['lightblue', 'lightcoral']
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            # Set labels and title with statistics
            ax.set_xticks([1, 2])
            ax.set_xticklabels([self.group1_name, self.group2_name])
            
            result_row = results_df[results_df['feature'] == feat].iloc[0]
            mwu_p = result_row['mwu_pvalue_fdr']
            rb_corr = result_row['rank_biserial_correlation']
            rb_effect = result_row['rb_effect_size']
            
            # Significance stars
            stars = "***" if mwu_p < 0.001 else "**" if mwu_p < 0.01 else "*" if mwu_p < 0.05 else ""
            
            title = f"{feat}\nMWU p-val: {mwu_p:.3e}{stars}\nrb: {rb_corr:.3f} ({rb_effect})"
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    def run_all(self, alpha=0.05, visualization=False):
        """
        Run all univariate tests and return results.
        
        Args:
            alpha: significance level for FDR correction
            visualization: whether to show plots

        Returns:
            results_df: DataFrame with test results and FDR-corrected p-values
        """
        results_df = self.comprehensive_tests(alpha=alpha)
        
        if visualization:
            self.plot_pvalue_histogram(results_df, test_type='mwu', corrected=True)
            self.plot_violinplots(results_df, alpha=alpha)
            
        return results_df
