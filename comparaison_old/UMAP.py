from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu, ks_2samp
import seaborn as sns


class PCA_UMAP:
    def __init__(self, df1, df2, group1="Group 1", group2="Group 2", features=None):
        """Initialize with two dataframes and optional features to include."""
        self.df1 = df1.dropna()
        self.df2 = df2.dropna()
        self.label1 = group1
        self.label2 = group2
        self.features = features if features is not None else df1.columns.tolist()
        
    def correlation_matrices(self, plot=True):
        """Compute and optionally plot correlation matrices for each population and their difference."""
        corr1 = self.df1[self.features].corr()
        corr2 = self.df2[self.features].corr()
        corr_diff = corr1 - corr2

        if plot:
            self._plot_correlation_matrices(corr1, corr2, corr_diff)

        return corr1, corr2, corr_diff
    
    def _plot_correlation_matrices(self, corr1, corr2, corr_diff):
        """Plot correlation matrices for both populations."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.heatmap(corr1, annot=True, cmap='RdBu_r', center=0,
                   vmin=-1, vmax=1, fmt='.2f', cbar_kws={'shrink': 0.8}, ax=axes[0])
        axes[0].set_title(f'{self.label1} Correlations')

        sns.heatmap(corr2, annot=True, cmap='RdBu_r', center=0,
                   vmin=-1, vmax=1, fmt='.2f', cbar_kws={'shrink': 0.8}, ax=axes[1])
        axes[1].set_title(f'{self.label2} Correlations')

        plt.tight_layout()
        plt.show()

        # Plot difference matrix
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sns.heatmap(corr_diff, annot=True, cmap='RdBu_r', center=0,
                   vmin=-0.5, vmax=0.5, fmt='.2f', cbar_kws={'shrink': 0.8}, ax=ax)
        ax.set_title(f'Correlation Difference ({self.label1} - {self.label2})')
        plt.tight_layout()
        plt.show()
    
    def pca_analysis(self, plot_variance=True):
        """Perform PCA analysis with visualization and statistical testing."""
        # Prepare combined dataset
        combined_data, pca, scaler = self._prepare_pca_data()
        
        # Create PCA DataFrame
        pca_df = self._create_pca_dataframe(combined_data['transformed'], combined_data['labels'])
        
        if plot_variance:
            self._plot_variance_explained(pca)
            self._plot_pca_components(pca_df, pca.explained_variance_ratio_)
        
        # Statistical testing
        pc_test_results = self._test_pca_components(pca_df, pca.explained_variance_ratio_)
        pca_df.pc_test_results = pd.DataFrame(pc_test_results)
        
        # Plot significant components
        self._plot_significant_pcs(pca, pc_test_results)
        self._plot_pc_statistics(pca_df, pc_test_results, pca.explained_variance_ratio_)

        return pca_df, pca
    
    def _prepare_pca_data(self):
        """Prepare and standardize data for PCA."""
        df1 = self.df1[self.features].copy()
        df2 = self.df2[self.features].copy()
        df1['population'] = self.label1
        df2['population'] = self.label2

        combined = pd.concat([df1, df2], ignore_index=True)
        X = combined[self.features].values
        y = combined['population'].values

        # Standardize and apply PCA
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        
        pca = PCA()
        Xp = pca.fit_transform(Xs)

        return {'transformed': Xp, 'labels': y}, pca, scaler
    
    def _create_pca_dataframe(self, transformed_data, labels):
        """Create DataFrame with PCA results."""
        pca_df = pd.DataFrame(transformed_data, 
                             columns=[f'PC{i+1}' for i in range(transformed_data.shape[1])])
        pca_df['population'] = labels
        return pca_df
    
    def _plot_variance_explained(self, pca):
        """Plot explained variance ratios."""
        ratios = pca.explained_variance_ratio_
        cum = np.cumsum(ratios)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        ax[0].plot(range(1, len(ratios) + 1), ratios, 'bo-')
        ax[0].set_xlabel('Principal Component')
        ax[0].set_ylabel('Explained Variance Ratio')
        ax[0].set_title('PCA Explained Variance')
        ax[0].grid(alpha=0.3)

        ax[1].plot(range(1, len(cum) + 1), cum, 'ro-')
        ax[1].set_xlabel('Number of Components')
        ax[1].set_ylabel('Cumulative Variance')
        ax[1].set_title('Cumulative Explained Variance')
        ax[1].axhline(0.8, linestyle='--', alpha=0.7)
        ax[1].axhline(0.95, linestyle=':', alpha=0.7)
        ax[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_pca_components(self, pca_df, variance_ratios):
        """Plot first 4 principal components."""
        if pca_df.shape[1] < 6:  # Less than 4 PCs + population columns
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        pop1_mask = pca_df['population'] == self.label1
        pop2_mask = pca_df['population'] == self.label2
        
        pc_pairs = [('PC1', 'PC2', 0, 1), ('PC1', 'PC3', 0, 2), ('PC1', 'PC4', 0, 3),
                   ('PC2', 'PC3', 1, 2), ('PC2', 'PC4', 1, 3), ('PC3', 'PC4', 2, 3)]
        
        for idx, (pc1, pc2, var1_idx, var2_idx) in enumerate(pc_pairs):
            row, col = idx // 3, idx % 3
            
            axes[row, col].scatter(pca_df.loc[pop1_mask, pc1], pca_df.loc[pop1_mask, pc2], 
                                 alpha=0.6, label=self.label1, s=30)
            axes[row, col].scatter(pca_df.loc[pop2_mask, pc1], pca_df.loc[pop2_mask, pc2], 
                                 alpha=0.6, label=self.label2, s=30)
            
            axes[row, col].set_xlabel(f'{pc1} ({variance_ratios[var1_idx]:.1%} variance)')
            axes[row, col].set_ylabel(f'{pc2} ({variance_ratios[var2_idx]:.1%} variance)')
            axes[row, col].set_title(f'{pc1} vs {pc2}')
            axes[row, col].legend()
            axes[row, col].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _test_pca_components(self, pca_df, variance_ratios):
        """Perform statistical tests on principal components."""
        print("\n--- Statistical Tests on Principal Components ---")
        pc_test_results = []

        for i in range(min(4, len(variance_ratios))):
            pc_col = f'PC{i+1}'
            pc1_values = pca_df[pca_df['population'] == self.label1][pc_col].values
            pc2_values = pca_df[pca_df['population'] == self.label2][pc_col].values
            
            # Perform statistical tests
            t_stat, t_p = ttest_ind(pc1_values, pc2_values)
            mwu_stat, mwu_p = mannwhitneyu(pc1_values, pc2_values, alternative='two-sided')
            ks_stat, ks_p = ks_2samp(pc1_values, pc2_values)
            
            result = {
                'PC': pc_col,
                'variance_explained': variance_ratios[i],
                'mean_pop1': pc1_values.mean(),
                'mean_pop2': pc2_values.mean(),
                't_statistic': t_stat,
                't_pvalue': t_p,
                'mwu_pvalue': mwu_p,
                'ks_pvalue': ks_p
            }
            pc_test_results.append(result)
            
            print(f"{pc_col} ({variance_ratios[i]:.1%} variance): "
                  f"t-test p={t_p:.2e}, MWU p={mwu_p:.2e}, KS p={ks_p:.2e}")

        return pc_test_results
    
    def _plot_significant_pcs(self, pca, pc_test_results):
        """Plot loadings for statistically significant PCs."""
        significant_pcs = [i for i, result in enumerate(pc_test_results) 
                          if result['ks_pvalue'] < 0.05]

        if not significant_pcs:
            print("No significant PCs found (p < 0.05)")
            return

        loadings = pca.components_
        n_sig_pcs = len(significant_pcs)
        fig, axes = plt.subplots(1, n_sig_pcs, figsize=(8 * n_sig_pcs, 6))
        if n_sig_pcs == 1:
            axes = [axes]
        
        for idx, pc_idx in enumerate(significant_pcs):
            feature_importance = pd.DataFrame({
                'feature': self.features,
                'loading': loadings[pc_idx],
                'abs_loading': np.abs(loadings[pc_idx])
            }).sort_values('abs_loading', ascending=True)
            
            colors = ['red' if x < 0 else 'blue' for x in feature_importance['loading']]
            axes[idx].barh(range(len(feature_importance)), feature_importance['loading'], color=colors)
            
            axes[idx].set_yticks(range(len(feature_importance)))
            axes[idx].set_yticklabels(feature_importance['feature'])
            axes[idx].set_xlabel('Loading Value')
            axes[idx].set_title(f'PC{pc_idx+1} Feature Loadings\n'
                              f'(p-value: {pc_test_results[pc_idx]["t_pvalue"]:.2e})')
            axes[idx].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            axes[idx].grid(alpha=0.3, axis='x')
            
            # Add threshold lines
            threshold = 0.1
            axes[idx].axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
            axes[idx].axvline(x=-threshold, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        # Print top contributing features
        print("\n--- Feature Contributions to Significant PCs ---")
        for pc_idx in significant_pcs:
            feature_importance = pd.DataFrame({
                'feature': self.features,
                'loading': loadings[pc_idx],
                'abs_loading': np.abs(loadings[pc_idx])
            }).sort_values('abs_loading', ascending=False)
            
            print(f"\nPC{pc_idx+1} (p-value: {pc_test_results[pc_idx]['t_pvalue']:.2e}):")
            print("Top 5 contributing features:")
            for i in range(min(5, len(feature_importance))):
                feat = feature_importance.iloc[i]
                print(f"  {feat['feature']}: {feat['loading']:.3f}")
    
    def _plot_pc_statistics(self, pca_df, pc_test_results, variance_ratios):
        """Plot statistical test results and distributions for PCs."""
        if not pc_test_results:
            return
            
        # Statistical comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        pc_names = [r['PC'] for r in pc_test_results]
        t_pvals = [r['t_pvalue'] for r in pc_test_results]
        mwu_pvals = [r['mwu_pvalue'] for r in pc_test_results]
        ks_pvals = [r['ks_pvalue'] for r in pc_test_results]
        
        # P-values comparison
        x = np.arange(len(pc_names))
        width = 0.25
        
        axes[0,0].bar(x - width, t_pvals, width, label='t-test', alpha=0.7)
        axes[0,0].bar(x, mwu_pvals, width, label='Mann-Whitney U', alpha=0.7)
        axes[0,0].bar(x + width, ks_pvals, width, label='Kolmogorov-Smirnov', alpha=0.7)
        axes[0,0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
        axes[0,0].set_xlabel('Principal Components')
        axes[0,0].set_ylabel('P-value')
        axes[0,0].set_title('Statistical Test P-values for PCs')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(pc_names)
        axes[0,0].legend()
        axes[0,0].grid(alpha=0.3)
        axes[0,0].set_yscale('log')
        
        # Mean differences
        mean_diffs = [abs(r['mean_pop1'] - r['mean_pop2']) for r in pc_test_results]
        axes[0,1].bar(pc_names, mean_diffs, alpha=0.7, color='orange')
        axes[0,1].set_xlabel('Principal Components')
        axes[0,1].set_ylabel('|Mean Difference|')
        axes[0,1].set_title('Absolute Mean Differences between Populations')
        axes[0,1].grid(alpha=0.3)
        
        # Box plots for first 2 PCs
        for idx, pc_num in enumerate([1, 2]):
            if pc_num <= len(pc_test_results):
                pc_col = f'PC{pc_num}'
                pc1_vals = pca_df[pca_df['population'] == self.label1][pc_col].values
                pc2_vals = pca_df[pca_df['population'] == self.label2][pc_col].values
                
                ax = axes[1, idx]
                bp = ax.boxplot([pc1_vals, pc2_vals], labels=[self.label1, self.label2], 
                               patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][1].set_facecolor('lightcoral')
                ax.set_ylabel(f'{pc_col} Score')
                ax.set_title(f'{pc_col} Distribution by Population\n'
                           f'(p-value: {pc_test_results[pc_num-1]["t_pvalue"]:.2e})')
                ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Violin plots
        self._plot_pc_violins(pca_df, pc_test_results, variance_ratios)
    
    def _plot_pc_violins(self, pca_df, pc_test_results, variance_ratios):
        """Plot violin plots for principal components."""
        n_pcs = len(pc_test_results)
        if n_pcs == 0:
            return
            
        fig, axes = plt.subplots(1, min(4, n_pcs), figsize=(4 * min(4, n_pcs), 6))
        if n_pcs == 1:
            axes = [axes]
        
        for i in range(min(4, n_pcs)):
            pc_col = f'PC{i+1}'
            pc1_vals = pca_df[pca_df['population'] == self.label1][pc_col].values
            pc2_vals = pca_df[pca_df['population'] == self.label2][pc_col].values
            
            parts = axes[i].violinplot([pc1_vals, pc2_vals], positions=[1, 2], showmeans=True)
            parts['bodies'][0].set_facecolor('lightblue')
            parts['bodies'][1].set_facecolor('lightcoral')
            
            axes[i].set_xticks([1, 2])
            axes[i].set_xticklabels([self.label1, self.label2])
            axes[i].set_ylabel(f'{pc_col} Score')
            
            p_val = pc_test_results[i]['t_pvalue']
            significance = ('***' if p_val < 0.001 else '**' if p_val < 0.01 
                          else '*' if p_val < 0.05 else 'ns')
            axes[i].set_title(f'{pc_col}\n({variance_ratios[i]:.1%} var, '
                            f'p={p_val:.2e} {significance})')
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def umap_analysis(self, n_neighbors=10, min_dist=0.05, n_components=2, random_state=42, plot=True):
        """Perform UMAP dimensionality reduction and visualization."""
        # Prepare data
        combined_data, umap_model = self._prepare_umap_data(n_neighbors, min_dist, 
                                                           n_components, random_state)
        
        # Create UMAP DataFrame
        umap_df = self._create_umap_dataframe(combined_data['transformed'], combined_data['labels'])
        
        if plot and n_components >= 2:
            self._plot_umap_results(umap_df)
            
            # Statistical testing
            umap_test_results = self._test_umap_components(umap_df, n_components)
            umap_df.umap_test_results = pd.DataFrame(umap_test_results)
            
            self._plot_umap_statistics(umap_df, umap_test_results, n_components)
            self._plot_umap_feature_importance(umap_df, combined_data['standardized'], n_components)

        return umap_df, umap_model
    
    def _prepare_umap_data(self, n_neighbors, min_dist, n_components, random_state):
        """Prepare and transform data for UMAP."""
        df1 = self.df1[self.features].copy()
        df2 = self.df2[self.features].copy()
        df1['population'] = self.label1
        df2['population'] = self.label2
        combined = pd.concat([df1, df2], ignore_index=True)

        X = combined[self.features].values
        y = combined['population'].values

        # Standardize
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # UMAP
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=random_state
        )
        
        X_umap = umap_model.fit_transform(Xs)
        
        return {'transformed': X_umap, 'labels': y, 'standardized': Xs}, umap_model
    
    def _create_umap_dataframe(self, transformed_data, labels):
        """Create DataFrame with UMAP results."""
        umap_columns = [f'UMAP{i+1}' for i in range(transformed_data.shape[1])]
        umap_df = pd.DataFrame(transformed_data, columns=umap_columns)
        umap_df['population'] = labels
        return umap_df
    
    def _plot_umap_results(self, umap_df):
        """Plot UMAP projection results."""
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        # Seaborn plot
        sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='population', 
                       alpha=0.6, ax=ax)
        ax.set_title('UMAP Projection')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _test_umap_components(self, umap_df, n_components):
        """Perform statistical tests on UMAP components."""
        print("\n--- Statistical Tests on UMAP Components ---")
        umap_test_results = []
        
        for i in range(min(n_components, 4)):
            umap_col = f'UMAP{i+1}'
            umap1_values = umap_df[umap_df['population'] == self.label1][umap_col].values
            umap2_values = umap_df[umap_df['population'] == self.label2][umap_col].values
            
            # Statistical tests
            t_stat, t_p = ttest_ind(umap1_values, umap2_values)
            mwu_stat, mwu_p = mannwhitneyu(umap1_values, umap2_values, alternative='two-sided')
            ks_stat, ks_p = ks_2samp(umap1_values, umap2_values)
            
            result = {
                'Component': umap_col,
                'mean_pop1': umap1_values.mean(),
                'mean_pop2': umap2_values.mean(),
                't_statistic': t_stat,
                't_pvalue': t_p,
                'mwu_pvalue': mwu_p,
                'ks_pvalue': ks_p
            }
            umap_test_results.append(result)
            
            print(f"{umap_col}: t-test p={t_p:.2e}, MWU p={mwu_p:.2e}, KS p={ks_p:.2e}")
        
        return umap_test_results
    
    def _plot_umap_statistics(self, umap_df, umap_test_results, n_components):
        """Plot violin plots for UMAP components."""
        if not umap_test_results:
            return
            
        n_comps = min(n_components, 4)
        fig, axes = plt.subplots(1, n_comps, figsize=(5 * n_comps, 5))
        if n_comps == 1:
            axes = [axes]
        
        for i in range(n_comps):
            umap_col = f'UMAP{i+1}'
            umap1_vals = umap_df[umap_df['population'] == self.label1][umap_col].values
            umap2_vals = umap_df[umap_df['population'] == self.label2][umap_col].values
            
            parts = axes[i].violinplot([umap1_vals, umap2_vals], positions=[1, 2], showmeans=True)
            parts['bodies'][0].set_facecolor('lightblue')
            parts['bodies'][1].set_facecolor('lightcoral')
            
            axes[i].set_xticks([1, 2])
            axes[i].set_xticklabels([self.label1, self.label2])
            axes[i].set_ylabel(f'{umap_col}')
            
            p_val = umap_test_results[i]['mwu_pvalue']
            significance = ('***' if p_val < 0.001 else '**' if p_val < 0.01 
                          else '*' if p_val < 0.05 else 'ns')
            axes[i].set_title(f'{umap_col}\n(p={p_val:.2e} {significance})')
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_umap_feature_importance(self, umap_df, standardized_data, n_components):
        """Analyze and plot feature importance for UMAP components."""
        print("\n--- UMAP Feature Analysis ---")
        
        feature_importance_results = []
        
        for i in range(min(n_components, 2)):
            umap_col = f'UMAP{i+1}'
            component_values = umap_df[umap_col].values
            
            correlations = []
            for j, feature in enumerate(self.features):
                feature_values = standardized_data[:, j]
                corr = np.corrcoef(component_values, feature_values)[0, 1]
                correlations.append({
                    'feature': feature,
                    'correlation': corr,
                    'abs_correlation': abs(corr)
                })
            
            correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
            feature_importance_results.append({
                'component': umap_col,
                'correlations': correlations
            })
            
            print(f"\n{umap_col} - Top 5 correlated features:")
            for k in range(min(5, len(correlations))):
                feat = correlations[k]
                print(f"  {feat['feature']}: {feat['correlation']:.3f}")
        
        # Plot feature correlations
        if feature_importance_results:
            n_plot_comps = min(n_components, 2)
            fig, axes = plt.subplots(1, n_plot_comps, figsize=(8 * n_plot_comps, 6))
            if n_plot_comps == 1:
                axes = [axes]
            
            for idx, result in enumerate(feature_importance_results[:n_plot_comps]):
                correlations = result['correlations']
                corr_df = pd.DataFrame(correlations).sort_values('correlation')
                
                colors = ['red' if x < 0 else 'blue' for x in corr_df['correlation']]
                axes[idx].barh(range(len(corr_df)), corr_df['correlation'], color=colors)
                
                axes[idx].set_yticks(range(len(corr_df)))
                axes[idx].set_yticklabels(corr_df['feature'])
                axes[idx].set_xlabel('Correlation with UMAP Component')
                axes[idx].set_title(f'{result["component"]} Feature Correlations')
                axes[idx].axvline(x=0, color='black', linestyle='-', alpha=0.3)
                axes[idx].grid(alpha=0.3, axis='x')
                
                # Add threshold lines
                threshold = 0.3
                axes[idx].axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
                axes[idx].axvline(x=-threshold, color='gray', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.show()
    
    def run_all(self, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, plot=True):
        """Run both PCA and UMAP analyses and return results."""
        # Run correlation matrices
        self.correlation_matrices(plot=plot)
        
        # Run PCA
        pca_results = self.pca_analysis()
        
        # Run UMAP
        umap_results = self.umap_analysis(n_neighbors, min_dist, n_components, random_state, plot)
        
        return pca_results, umap_results
