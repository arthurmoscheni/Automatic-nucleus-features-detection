# --- Required imports (explicit & consolidated) ---
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from scipy.stats import kruskal, rankdata
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from scikit_posthocs import posthoc_dunn
import warnings
warnings.filterwarnings('ignore')


class MultiGroupAnalysis:
    def __init__(self, young_wt_data, young_app_data, aged_wt_data, aged_app_data, features):
        """
        Initialize the multi-group analysis for 4 groups.

        Args:
            young_wt_data: Data for young wild-type group
            young_app_data: Data for young APP group
            aged_wt_data: Data for aged wild-type group
            aged_app_data: Data for aged APP group
            features: List of feature names to analyze
        """
        self.young_wt = young_wt_data
        self.young_app = young_app_data
        self.aged_wt = aged_wt_data
        self.aged_app = aged_app_data
        self.features = features

        # Store groups in a dictionary for easier access
        self.groups = {
            'young_wt': self.young_wt,
            'young_app': self.young_app,
            'aged_wt': self.aged_wt,
            'aged_app': self.aged_app
        }
        self.group_names = list(self.groups.keys())

    # -------------------------------
    # Helper methods (class methods)
    # -------------------------------

    def cliffs_delta(self, x, y):
        """Cliff's delta via Mann–Whitney U (fast, exact for continuous data). Positive => x > y."""
        from scipy.stats import mannwhitneyu
        x = pd.Series(x).dropna().values
        y = pd.Series(y).dropna().values
        n1, n2 = len(x), len(y)
        if n1 == 0 or n2 == 0:
            return np.nan
        U, _ = mannwhitneyu(x, y, alternative="two-sided")
        return 2.0 * U / (n1 * n2) - 1.0

    def cliffs_delta_with_ci(self, x, y, n_bootstrap=1000, random_state=42):
        """Cliff's delta with bootstrap 95% CI."""
        rng = np.random.default_rng(random_state)
        x = pd.Series(x).dropna().values
        y = pd.Series(y).dropna().values
        if len(x) == 0 or len(y) == 0:
            return np.nan, np.nan, np.nan
        delta = self.cliffs_delta(x, y)
        if n_bootstrap <= 0:
            return delta, np.nan, np.nan
        boot = []
        for _ in range(n_bootstrap):
            xb = rng.choice(x, size=len(x), replace=True)
            yb = rng.choice(y, size=len(y), replace=True)
            boot.append(self.cliffs_delta(xb, yb))
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        return delta, ci_low, ci_high

    def permutation_test_delta_diff(self, young_app, young_wt, aged_app, aged_wt, n_perm=10000, random_state=42):
        """
        Permutation test for Δδ = δ_aged − δ_young.
        Shuffles labels within each age to build the null.
        """
        rng = np.random.default_rng(random_state)

        def _delta(x_app, x_wt):
            return self.cliffs_delta(x_app, x_wt)

        y_app = pd.Series(young_app).dropna().values
        y_wt  = pd.Series(young_wt ).dropna().values
        a_app = pd.Series(aged_app ).dropna().values
        a_wt  = pd.Series(aged_wt  ).dropna().values

        if min(len(y_app), len(y_wt), len(a_app), len(a_wt)) == 0:
            return np.nan, []

        obs_y = _delta(y_app, y_wt)
        obs_a = _delta(a_app, a_wt)
        obs_d = obs_a - obs_y

        young_all = np.concatenate([y_app, y_wt])
        aged_all  = np.concatenate([a_app, a_wt])

        perm_d = []
        for _ in range(n_perm):
            ry = rng.permutation(young_all)
            ra = rng.permutation(aged_all)

            y_app_p = ry[:len(y_app)]
            y_wt_p  = ry[len(y_app):]
            a_app_p = ra[:len(a_app)]
            a_wt_p  = ra[len(a_app):]

            dy = _delta(y_app_p, y_wt_p)
            da = _delta(a_app_p, a_wt_p)
            perm_d.append(da - dy)

        perm_d = np.asarray(perm_d, dtype=float)
        p_val = (1 + np.sum(np.abs(perm_d) >= np.abs(obs_d))) / (1 + len(perm_d))
        return p_val, perm_d.tolist()

    # -----------------------------------
    # Multivariate analysis (standalone)
    # -----------------------------------

    def multivariate_analysis(self, alpha=0.05, n_permutations=0):
        """
        PERMANOVA-like R² comparison of genotype across ages using Euclidean distances.
        Computes overall genotype R², within-age genotype R² (Young/Aged), ΔR² and a permutation p-value.
        """
        from scipy.spatial.distance import pdist, squareform

        print("Performing Multivariate PERMANOVA Analysis")
        print("=" * 50)

        # Build a complete-case matrix across features
        data_list = []
        for group_name, data in self.groups.items():
            age = 'Young' if 'young' in group_name.lower() else 'Aged'
            genotype = 'APP' if 'app' in group_name.lower() else 'WT'
            complete = data[self.features].dropna()
            for idx, row in complete.iterrows():
                row_dict = {
                    'age': age,
                    'genotype': genotype,
                    'group': group_name,
                    'subject_id': f"{group_name}_{idx}",
                }
                for f in self.features:
                    row_dict[f] = row[f]
                data_list.append(row_dict)

        if not data_list:
            print("No complete cases found across all features.")
            return None

        dfc = pd.DataFrame(data_list)
        X = dfc[self.features].values
        D = squareform(pdist(X, metric='euclidean'))

        def r2_for_labels(Dmat, labels):
            """Compute a simple R² from distances for a grouping factor."""
            labels = np.asarray(labels)
            n = len(labels)
            groups = np.unique(labels)
            ss_total = np.sum(Dmat ** 2) / n
            ss_within = 0.0
            for g in groups:
                mask = labels == g
                if np.sum(mask) > 1:
                    G = Dmat[np.ix_(mask, mask)]
                    ss_within += np.sum(G ** 2) / np.sum(mask)
            ss_between = ss_total - ss_within
            r2 = ss_between / ss_total if ss_total > 0 else 0.0
            return r2

        geno = dfc['genotype'].values
        age  = dfc['age'].values

        r2_genotype_overall = r2_for_labels(D, geno)

        # within-age R²
        y_mask = (age == 'Young')
        a_mask = (age == 'Aged')
        if np.sum(y_mask) > 1 and np.sum(a_mask) > 1:
            Dy = D[np.ix_(y_mask, y_mask)]
            Da = D[np.ix_(a_mask, a_mask)]
            r2_genotype_young = r2_for_labels(Dy, geno[y_mask])
            r2_genotype_aged  = r2_for_labels(Da, geno[a_mask])
            delta_r2 = r2_genotype_aged - r2_genotype_young

            # permutation within each age: shuffle genotype labels
            perm_d = []
            rng = np.random.default_rng(42)
            for _ in range(n_permutations):
                gy = rng.permutation(geno[y_mask])
                ga = rng.permutation(geno[a_mask])
                r2y = r2_for_labels(Dy, gy)
                r2a = r2_for_labels(Da, ga)
                perm_d.append(r2a - r2y)
            perm_d = np.asarray(perm_d, dtype=float)
            p_val = (1 + np.sum(np.abs(perm_d) >= np.abs(delta_r2))) / (1 + len(perm_d))
        else:
            r2_genotype_young = np.nan
            r2_genotype_aged  = np.nan
            delta_r2 = np.nan
            p_val = np.nan
            perm_d = np.array([])

        # Age main effect (overall)
        r2_age_overall = r2_for_labels(D, age)

        print(f"\nMultivariate Analysis Results:")
        print(f"Overall Genotype R²: {r2_genotype_overall:.4f}")
        print(f"Young Genotype R²:   {r2_genotype_young:.4f}")
        print(f"Aged Genotype R²:    {r2_genotype_aged:.4f}")
        print(f"Age R²:              {r2_age_overall:.4f}")
        print(f"ΔR² (Aged - Young):  {delta_r2:.4f}")
        print(f"Permutation p-value: {p_val:.4f}")

        return {
            'r2_genotype_overall': r2_genotype_overall,
            'r2_genotype_young': r2_genotype_young,
            'r2_genotype_aged': r2_genotype_aged,
            'r2_age_overall': r2_age_overall,
            'delta_r2': delta_r2,
            'permutation_p': p_val,
            'permutation_distribution': perm_d.tolist(),
            'n_subjects': len(dfc),
            'n_young': int(np.sum(y_mask)),
            'n_aged': int(np.sum(a_mask)),
            'complete_data': dfc
        }

    # -----------------------------------
    # ART ANOVA utilities
    # -----------------------------------

    def aligned_rank_transform(self, df, value_col, factors):
        """
        Perform Aligned Rank Transform for factorial ANOVA.
        """
        df_art = df.copy()
        print("Performing Aligned Rank Transform")
        print("=" * 40)
        effects = factors + [f"{factors[0]}:{factors[1]}"]  # main effects + interaction

        art_data = {}

        for effect in effects:
            if ':' in effect:  # Interaction effect
                factor1, factor2 = effect.split(':')

                # Means
                row_means = df_art.groupby(factor1)[value_col].mean()
                col_means = df_art.groupby(factor2)[value_col].mean()
                grand_mean = df_art[value_col].mean()

                aligned_values = []
                for _, row in df_art.iterrows():
                    row_mean = row_means.loc[row[factor1]]
                    col_mean = col_means.loc[row[factor2]]
                    aligned = row[value_col] - row_mean - col_mean + grand_mean
                    aligned_values.append(aligned)

                ranks = rankdata(aligned_values)
                art_data[f"ART_{effect}"] = ranks

            else:  # Main effect
                other_factor = [f for f in factors if f != effect][0]
                other_means = df_art.groupby(other_factor)[value_col].mean()
                grand_mean = df_art[value_col].mean()

                aligned_values = []
                for _, row in df_art.iterrows():
                    other_mean = other_means.loc[row[other_factor]]
                    aligned = row[value_col] - other_mean + grand_mean
                    aligned_values.append(aligned)

                ranks = rankdata(aligned_values)
                art_data[f"ART_{effect}"] = ranks

        for effect, ranks in art_data.items():
            df_art[effect] = ranks

        return df_art, art_data.keys()

    # -----------------------------------
    # Analyses
    # -----------------------------------

    def kruskal_wallis_analysis(self, alpha=0.05, plot=True, figsize=(15, 10)):
        """
        Perform Kruskal-Wallis test with post-hoc Dunn's test for each feature.

        Returns:
            dict: Results containing KW stats, p-values, effect sizes, and post-hoc tests
        """
        results = {}

        for feature in self.features:
            # Extract feature data for each group
            group_data = []
            group_labels = []

            for group_name, data in self.groups.items():
                if feature in data.columns:
                    feature_values = data[feature].dropna()
                    group_data.append(feature_values)
                    group_labels.extend([group_name] * len(feature_values))

            if len(group_data) < 4:
                print(f"Warning: Feature '{feature}' not found in all groups. Skipping.")
                continue

            # Kruskal-Wallis
            kw_stat, kw_pvalue = kruskal(*group_data)

            # Effect sizes
            n_total = sum(len(group) for group in group_data)
            eta_squared = (kw_stat - len(group_data) + 1) / (n_total - len(group_data))
            epsilon_squared = kw_stat / (n_total - 1)

            # Post-hoc Dunn's (FDR)
            all_values = np.concatenate(group_data)
            df_posthoc = pd.DataFrame({'value': all_values, 'group': group_labels})
            dunn_results = posthoc_dunn(df_posthoc, val_col='value', group_col='group',
                                        p_adjust='fdr_bh')

            # Pairwise Cliff's delta matrix
            cliffs_delta_matrix = pd.DataFrame(index=self.group_names, columns=self.group_names, dtype=float)
            for i, g1 in enumerate(self.group_names):
                for j, g2 in enumerate(self.group_names):
                    if i == j:
                        cliffs_delta_matrix.loc[g1, g2] = np.nan
                    else:
                        data1 = self.groups[g1][feature].dropna()
                        data2 = self.groups[g2][feature].dropna()
                        cliffs_delta_matrix.loc[g1, g2] = self.cliffs_delta(data1, data2)

            # Store
            results[feature] = {
                'kw_statistic': kw_stat,
                'kw_pvalue': kw_pvalue,
                'eta_squared': eta_squared,
                'epsilon_squared': epsilon_squared,
                'significant': kw_pvalue < alpha,
                'dunn_pvalues': dunn_results,
                'cliffs_delta': cliffs_delta_matrix,
                'group_data': group_data,
                'group_names': self.group_names
            }

        # Summary dataframe
        summary_df = pd.DataFrame({
            'Feature': list(results.keys()),
            'KW_Statistic': [results[f]['kw_statistic'] for f in results.keys()],
            'KW_P_value': [results[f]['kw_pvalue'] for f in results.keys()],
            'Eta_Squared': [results[f]['eta_squared'] for f in results.keys()],
            'Epsilon_Squared': [results[f]['epsilon_squared'] for f in results.keys()],
            'Significant': [results[f]['significant'] for f in results.keys()]
        })

        # FDR across features
        if not summary_df.empty:
            _, fdr_pvals, _, _ = multipletests(summary_df['KW_P_value'], method='fdr_bh')
            summary_df['FDR_P_value'] = fdr_pvals
            summary_df['FDR_Significant'] = fdr_pvals < alpha

        if plot and results:
            # Layout
            n_features = len(results)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = np.atleast_1d(axes).ravel()

            colors = {
                'young_wt': '#4A90E2',   # Light blue for young WT
                'young_app': '#1F5F99',  # Dark blue for young APP
                'aged_wt': '#E85D75',    # Light red for aged WT
                'aged_app': '#B73E56'    # Dark red for aged APP
            }

            ordered_groups = ['young_wt', 'young_app', 'aged_wt', 'aged_app']

            for ax, (feature, result) in zip(axes, results.items()):
                plot_data = []
                for group_name in ordered_groups:
                    if group_name in self.groups and feature in self.groups[group_name].columns:
                        group_values = self.groups[group_name][feature].dropna()
                        for value in group_values:
                            age = 'Young' if 'young' in group_name else 'Aged'
                            genotype = 'WT' if 'wt' in group_name else 'APP'
                            plot_data.append({
                                'Value': value,
                                'Age': age,
                                'Genotype': genotype,
                                'Group': group_name
                            })

                df_plot = pd.DataFrame(plot_data)
                if df_plot.empty:
                    ax.set_visible(False)
                    continue

                # grouped positions: (Young: WT, APP) | (Aged: WT, APP)
                x_positions, box_data, box_colors, labels = [], [], [], []
                for i, age in enumerate(['Young', 'Aged']):
                    for j, genotype in enumerate(['WT', 'APP']):
                        gname = f"{age.lower()}_{genotype.lower()}"
                        subset = df_plot[(df_plot['Age'] == age) & (df_plot['Genotype'] == genotype)]
                        if subset.empty:
                            continue
                        x_pos = i * 3 + j
                        x_positions.append(x_pos)
                        box_data.append(subset['Value'].values)
                        box_colors.append(colors[gname])
                        labels.append(f"{age}\n{genotype}")

                bp = ax.boxplot(box_data, positions=x_positions, patch_artist=True,
                                widths=0.6, showfliers=True)

                for patch, color in zip(bp['boxes'], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1)

                for element in ['whiskers', 'fliers', 'medians', 'caps']:
                    plt.setp(bp[element], color='black')
                plt.setp(bp['medians'], linewidth=2)

                ax.set_xticks(x_positions)
                ax.set_xticklabels(labels, fontsize=10)
                ax.set_ylabel('Value', fontsize=12)

                if len(x_positions) >= 3:
                    ax.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5)

                title = f'{feature}\nKW p={result["kw_pvalue"]:.1e}, η²={result["eta_squared"]:.3f}'
                if result['significant']:
                    title += ' *'
                ax.set_title(title, fontsize=11, pad=10)

                if result['significant']:
                    ax.text(0.02, 0.98, '★', transform=ax.transAxes,
                            fontsize=20, color='gold', weight='bold', va='top')

                ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
                ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(0.5)
                ax.spines['bottom'].set_linewidth(0.5)

            # Hide any unused axes
            for ax in axes[len(results):]:
                ax.set_visible(False)

            legend_elements = [
                Patch(facecolor=colors['young_wt'], alpha=0.7, label='Young WT'),
                Patch(facecolor=colors['young_app'], alpha=0.7, label='Young APP'),
                Patch(facecolor=colors['aged_wt'], alpha=0.7, label='Aged WT'),
                Patch(facecolor=colors['aged_app'], alpha=0.7, label='Aged APP')
            ]
            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            plt.show()

            # Heatmap for significant features' post-hoc Dunn p-values
            if any(results[f]['significant'] for f in results.keys()):
                significant_features = [f for f in results.keys() if results[f]['significant']]
                if significant_features:
                    fig, axes_hm = plt.subplots(1, len(significant_features),
                                                figsize=(5 * len(significant_features), 4))
                    axes_hm = np.atleast_1d(axes_hm)
                    for ax_hm, feat in zip(axes_hm, significant_features):
                        dunn_matrix = results[feat]['dunn_pvalues']
                        mask = dunn_matrix >= alpha
                        sns.heatmap(dunn_matrix, annot=True, fmt='.4f',
                                    mask=mask, cmap='RdYlBu_r', center=alpha,
                                    ax=ax_hm, cbar_kws={'label': 'Dunn p-value'})
                        ax_hm.set_title(f'{feat}\nDunn Post-hoc (FDR corrected)')
                    plt.tight_layout()
                    plt.show()

        print("Kruskal-Wallis Analysis Summary:")
        print("=" * 50)
        if not summary_df.empty:
            print(summary_df.to_string(index=False))
            print(f"\nSignificant features (α = {alpha}): {summary_df['Significant'].sum()}")
            print(f"FDR-corrected significant features: {summary_df['FDR_Significant'].sum()}")
        else:
            print("No valid results found.")

        return {
            'summary': summary_df,
            'detailed_results': results,
            'alpha': alpha
        }

    # -----------------------------------
    # ART ANOVA main method
    # -----------------------------------

    def art_anova_analysis(self, alpha=0.05, plot=True, figsize=(15, 10)):
        """
        Run Aligned Rank Transform (ART) ANOVA per feature for Age, Genotype, and Interaction.
        Expects self.aligned_rank_transform(df, value_col, factors) to exist.
        """
        print("Performing ART ANOVA Analysis")
        print("=" * 40)

        results = {}

        for feature in self.features:
            print(f"Analyzing feature: {feature}")

            # Build long dataframe for the feature
            data_list = []

            if feature in self.young_wt.columns:
                for v in self.young_wt[feature].dropna():
                    data_list.append({'value': v, 'age': 'Young', 'genotype': 'WT',  'group': 'young_wt'})
            if feature in self.young_app.columns:
                for v in self.young_app[feature].dropna():
                    data_list.append({'value': v, 'age': 'Young', 'genotype': 'APP', 'group': 'young_app'})
            if feature in self.aged_wt.columns:
                for v in self.aged_wt[feature].dropna():
                    data_list.append({'value': v, 'age': 'Aged',  'genotype': 'WT',  'group': 'aged_wt'})
            if feature in self.aged_app.columns:
                for v in self.aged_app[feature].dropna():
                    data_list.append({'value': v, 'age': 'Aged',  'genotype': 'APP', 'group': 'aged_app'})

            if not data_list:
                print(f"Warning: No data found for feature '{feature}'. Skipping.")
                continue

            df = pd.DataFrame(data_list)
            if df['group'].nunique() < 4:
                print(f"Warning: Not all groups present for feature '{feature}'. Skipping.")
                continue

            # ART transform
            df_art, art_cols = self.aligned_rank_transform(df, 'value', ['age', 'genotype'])

            # Fit ANOVA on each ART column and collect the matching effect
            anova_results = {}
            for art_col in art_cols:
                effect_name = art_col.replace('ART_', '').replace(':', '_x_')

                # Ensure numeric & clean
                df_art[art_col] = pd.to_numeric(df_art[art_col], errors='coerce')
                df_clean = df_art.dropna(subset=[art_col])
                if df_clean.empty:
                    print(f"Warning: No valid data for {art_col} in feature {feature}")
                    continue

                try:
                    model = ols(f"Q('{art_col}') ~ C(age) * C(genotype)", data=df_clean).fit()
                    tab = anova_lm(model, typ=2)
                except Exception as e:
                    print(f"Warning: ANOVA failed for {art_col} in feature {feature}: {e}")
                    continue

                # Map ART column to specific effect row
                if 'age:genotype' in art_col or 'age_x_genotype' in effect_name:
                    row = 'C(age):C(genotype)'
                elif 'age' in art_col and 'genotype' not in art_col:
                    row = 'C(age)'
                elif 'genotype' in art_col and 'age' not in art_col:
                    row = 'C(genotype)'
                else:
                    row = None

                if row and row in tab.index:
                    f_stat = tab.loc[row, 'F']
                    p_val  = tab.loc[row, 'PR(>F)']
                    ss_eff = tab.loc[row, 'sum_sq']
                    ss_err = tab.loc['Residual', 'sum_sq']
                    eta2p  = ss_eff / (ss_eff + ss_err) if (ss_eff + ss_err) > 0 else np.nan

                    anova_results[effect_name] = {
                        'F_statistic': f_stat,
                        'p_value': p_val,
                        'partial_eta_squared': eta2p,
                        'significant': p_val < alpha,
                        'full_anova': tab
                    }

            results[feature] = {
                'anova_results': anova_results,
                'data': df,
                'art_data': df_art,
                'group_means': df.groupby(['age', 'genotype'])['value'].agg(['mean', 'std', 'count'])
            }

        # Post-hoc for significant interaction
        posthoc_results = {}
        for feature, res in results.items():
            ar = res['anova_results']
            if 'age_x_genotype' in ar and ar['age_x_genotype']['significant']:
                dff = res['data']
                y_wt  = dff[(dff.age == 'Young') & (dff.genotype == 'WT')]['value'].values
                y_app = dff[(dff.age == 'Young') & (dff.genotype == 'APP')]['value'].values
                a_wt  = dff[(dff.age == 'Aged')  & (dff.genotype == 'WT')]['value'].values
                a_app = dff[(dff.age == 'Aged')  & (dff.genotype == 'APP')]['value'].values

                dy, dy_lo, dy_hi = self.cliffs_delta_with_ci(y_app, y_wt)
                da, da_lo, da_hi = self.cliffs_delta_with_ci(a_app, a_wt)
                d_diff = da - dy if (np.isfinite(da) and np.isfinite(dy)) else np.nan
                p_perm, perm_dist = self.permutation_test_delta_diff(y_app, y_wt, a_app, a_wt)

                posthoc_results[feature] = {
                    'delta_young': dy, 'delta_young_ci': (dy_lo, dy_hi),
                    'delta_aged' : da, 'delta_aged_ci' : (da_lo, da_hi),
                    'delta_difference': d_diff,
                    'permutation_p': p_perm,
                    'permutation_distribution': perm_dist
                }

        for feature, ph in posthoc_results.items():
            if feature in results:
                results[feature]['within_age_contrasts'] = ph

        # Summaries + FDR + Plots
        summary_rows = []
        for feature, res in results.items():
            for eff, st in res['anova_results'].items():
                summary_rows.append({
                    'Feature': feature,
                    'Effect': eff,
                    'F_statistic': st['F_statistic'],
                    'P_value': st['p_value'],
                    'Partial_Eta_Squared': st['partial_eta_squared'],
                    'Significant': st['significant']
                })
        summary_df = pd.DataFrame(summary_rows)

        if not summary_df.empty:
            for eff in summary_df['Effect'].unique():
                mask = summary_df['Effect'] == eff
                _, q, _, _ = multipletests(summary_df.loc[mask, 'P_value'], method='fdr_bh')
                summary_df.loc[mask, 'FDR_P_value'] = q
                summary_df.loc[mask, 'FDR_Significant'] = q < alpha

        if plot and not summary_df.empty:
            # Interaction plots per feature
            n_features = len(results)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = np.atleast_1d(axes).ravel()

            for ax, (feature, res) in zip(axes, results.items()):
                d = res['data']
                means = d.groupby(['age', 'genotype'])['value'].mean().reset_index()
                sems  = d.groupby(['age', 'genotype'])['value'].sem().reset_index()
                ages = ['Young', 'Aged']
                for geno in ['WT', 'APP']:
                    gmean = [means[(means.age == a) & (means.genotype == geno)]['value'].mean() for a in ages]
                    gsem  = [sems [(sems .age == a) & (sems .genotype == geno)]['value'].mean() for a in ages]
                    ax.errorbar(range(len(ages)), gmean, yerr=gsem, marker='o', label=geno)
                ax.set_xticks(range(len(ages))); ax.set_xticklabels(ages)
                ax.set_xlabel('Age'); ax.set_ylabel(feature)
                title_bits = [feature]
                if 'age' in res['anova_results']:
                    p = res['anova_results']['age']['p_value']; title_bits.append(f"Age p={p:.2e}")
                if 'genotype' in res['anova_results']:
                    p = res['anova_results']['genotype']['p_value']; title_bits.append(f"Geno p={p:.2e}")
                if 'age_x_genotype' in res['anova_results']:
                    p = res['anova_results']['age_x_genotype']['p_value']; title_bits.append(f"Int p={p:.2e}")
                ax.set_title(" | ".join(title_bits), fontsize=9)
                ax.legend()
            for ax in axes[len(results):]:
                fig.delaxes(ax)
            plt.tight_layout(); plt.show()

            # Effect sizes heatmap (η²) with annotations
            pivot_eta = summary_df.pivot(index='Feature', columns='Effect', values='Partial_Eta_Squared')
            fig, ax = plt.subplots(1, 1, figsize=(12, max(6, len(pivot_eta) * 0.6)))
            annot = pivot_eta.copy().astype(str)
            for i in range(len(pivot_eta)):
                for j in range(len(pivot_eta.columns)):
                    eta = pivot_eta.iloc[i, j]
                    annot.iloc[i, j] = f"{eta:.4f}" if pd.notna(eta) else ""
            sns.heatmap(pivot_eta, annot=annot, fmt='', cmap='RdYlBu_r',
                        cbar_kws={'label': 'Partial η²'}, ax=ax, annot_kws={'fontsize': 9})
            ax.set_title('ART ANOVA Results: Effect Sizes (η²)')
            plt.tight_layout(); plt.show()

        print("\nART ANOVA Analysis Summary:")
        print("=" * 50)
        if not summary_df.empty:
            print(summary_df.to_string(index=False))
            print(f"\nEffect Summary (α = {alpha}):")
            for eff in summary_df['Effect'].unique():
                sub = summary_df[summary_df['Effect'] == eff]
                print(f"{eff}: {int(sub['Significant'].sum())}/{len(sub)} significant features")
        else:
            print("No valid results found.")

        return {
            'summary': summary_df,
            'detailed_results': results,
            'alpha': alpha
        }

    def classification_analysis(self, random_state=42, plot=True, figsize=(15, 10),
                                n_permutations=100, use_gpu=True, fast_mode=True):
        """
        Optimized non-linear classification analysis with optional GPU acceleration.
        Uses XGBoost (GPU if available), LightGBM, and RF with robust preprocessing,
        and wraps each task in a label-permutation significance test.

        Returns a dict with metrics, permutation p-values, and runtime info.
        """
        # ---- imports (scoped; keep as-is) ----
        import numpy as np
        import pandas as pd
        from time import perf_counter
        from sklearn.preprocessing import LabelEncoder, RobustScaler, QuantileTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import StratifiedKFold, cross_validate
        from sklearn.metrics import (make_scorer, balanced_accuracy_score,
                                    matthews_corrcoef, roc_auc_score, f1_score)
        from sklearn.model_selection import train_test_split
        from statsmodels.stats.multitest import multipletests
        from sklearn.ensemble import RandomForestClassifier

        # ---- try optional libs ----
        try:
            import xgboost as xgb
            try:
                _ = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=1)
                gpu_available = bool(use_gpu)
            except Exception:
                gpu_available = False
        except Exception:
            xgb = None
            gpu_available = False
            print("XGBoost not available; skipping XGB.")

        try:
            import lightgbm as lgb
            lgb_available = True
        except Exception:
            lgb = None
            lgb_available = False
            print("LightGBM not available; skipping LGBM.")

        print("Performing Optimized Classification Analysis")
        print("=" * 60)
        print(f"GPU mode: {'Enabled' if gpu_available else 'Disabled'}")
        print(f"Fast mode: {'Enabled' if fast_mode else 'Disabled'}")
        t0 = perf_counter()

        # ---- build complete-case dataset (vectorized) ----
        data_list = []
        for group_name, data in self.groups.items():
            complete = data[self.features].dropna()
            if complete.empty:
                continue
            gdf = complete.copy()
            gdf['group'] = group_name
            gdf['age'] = 'Young' if 'young' in group_name.lower() else 'Aged'
            gdf['genotype'] = 'APP' if 'app' in group_name.lower() else 'WT'
            gdf['subject_id'] = [f"{group_name}_{idx}" for idx in complete.index]
            data_list.append(gdf)

        if not data_list:
            print("No complete cases found across all features.")
            return None

        df = pd.concat(data_list, ignore_index=True)
        X = df[self.features].to_numpy(dtype=float)

        # manual encodings (preserve your mapping)
        group_mapping = {'young_wt': 0, 'young_app': 1, 'aged_wt': 2, 'aged_app': 3}
        age_mapping = {'Young': 0, 'Aged': 1}
        genotype_mapping = {'WT': 0, 'APP': 1}

        df['group_enc'] = df['group'].map(group_mapping)
        df['age_enc'] = df['age'].map(age_mapping)
        df['genotype_enc'] = df['genotype'].map(genotype_mapping)

        label_encoders = {
            'group': group_mapping,
            'age': age_mapping,
            'genotype': genotype_mapping
        }

        y_group = df['group_enc'].to_numpy()
        y_age = df['age_enc'].to_numpy()
        y_genotype = df['genotype_enc'].to_numpy()

        print(f"Dataset: {len(df)} subjects, {len(self.features)} features")
        print(f"Group distribution: {df['group'].value_counts().to_dict()}")

        # ---- preprocessing (robust -> quantile-normal) ----
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import RobustScaler, QuantileTransformer

        def create_preprocessor():
            return ColumnTransformer([
                ('robust_quantile', Pipeline([
                    ('robust', RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25, 75))),
                    ('quantile', QuantileTransformer(output_distribution='normal',
                                                    n_quantiles=min(1000, max(10, X.shape[0] // 2)),
                                                    random_state=random_state))
                ]), list(range(len(self.features))))
            ])

        # ---- models ----
        from sklearn.ensemble import RandomForestClassifier
        cv_folds = 3 if fast_mode else 5

        def get_models():
            models = {}
            if xgb is not None:
                models['xgb_gpu' if gpu_available else 'xgb_cpu'] = Pipeline([
                    ('preprocessor', create_preprocessor()),
                    ('classifier', xgb.XGBClassifier(
                        tree_method='gpu_hist' if gpu_available else 'hist',
                        gpu_id=0 if gpu_available else None,
                        n_estimators=100 if fast_mode else 200,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        random_state=random_state,
                        n_jobs=1 if gpu_available else -1,
                        verbosity=1
                    ))
                ])
            if lgb_available:
                models['lightgbm'] = Pipeline([
                    ('preprocessor', create_preprocessor()),
                    ('classifier', lgb.LGBMClassifier(
                        n_estimators=100 if fast_mode else 200,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=0.0,
                        random_state=random_state,
                        n_jobs=-1,
                        verbose=1,
                        class_weight='balanced' if y_group.size > 2 else None
                    ))
                ])
            models['rf_optimized'] = Pipeline([
                ('preprocessor', create_preprocessor()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100 if fast_mode else 200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=0
                ))
            ])
            return models

        models = get_models()

        # ---- scoring helpers ----
        from sklearn.metrics import make_scorer, matthews_corrcoef
        def create_scoring(unique_classes):
            if len(unique_classes) == 2:
                return {
                    'roc_auc': 'roc_auc',
                    'balanced_accuracy': 'balanced_accuracy',
                    'mcc': make_scorer(matthews_corrcoef)
                }
            else:
                return {
                    'balanced_accuracy': 'balanced_accuracy',
                    'macro_f1': 'f1_macro'
                }

        # ---- permutation test ----
        from sklearn.model_selection import StratifiedKFold, cross_validate
        def permutation_test(pipeline, X_, y_, scoring_dict, n_perm=100, cv=3, seed=42):
            rng = np.random.default_rng(seed)
            cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

            # observed
            obs_cv = cross_validate(pipeline, X_, y_, cv=cv_split, scoring=scoring_dict,
                                    return_train_score=False, n_jobs=-1, error_score='raise')
            obs_mean = {k.replace('test_', ''): float(np.mean(v)) for k, v in obs_cv.items() if k.startswith('test_')}

            # permutations
            perm_dists = {metric: [] for metric in obs_mean.keys()}
            for _ in range(int(max(1, n_perm))):
                y_perm = rng.permutation(y_)
                perm_cv = cross_validate(pipeline, X_, y_perm, cv=cv_split, scoring=scoring_dict,
                                         return_train_score=False, n_jobs=-1, error_score='raise')
                for k in obs_mean.keys():
                    perm_dists[k].append(float(np.mean(perm_cv['test_' + k])))

            results_ = {}
            for metric, obs_val in obs_mean.items():
                dist = np.asarray(perm_dists[metric], dtype=float)
                if dist.size == 0:
                    continue
                p_val = (1.0 + np.sum(dist >= obs_val)) / (1.0 + dist.size)
                z = (obs_val - dist.mean()) / (dist.std() + 1e-12)
                results_[metric] = {
                    'observed_score': obs_val,
                    'permutation_mean': float(dist.mean()),
                    'permutation_std': float(dist.std()),
                    'effect_size_z': float(z),
                    'p_value': float(p_val),
                    'n_permutations': int(dist.size),
                    'permutation_distribution': dist[:100].tolist()
                }
            return results_

        # ---- tasks ----
        tasks = {
            'four_groups': {
                'label': '4-Group Classification',
                'y_target': y_group,
                'metrics': ['balanced_accuracy', 'macro_f1'],
                'primary_metric': 'balanced_accuracy'
            },
            'age': {
                'label': 'Age Classification',
                'y_target': y_age,
                'metrics': ['roc_auc', 'balanced_accuracy', 'mcc'],
                'primary_metric': 'roc_auc'
            },
            'genotype': {
                'label': 'Genotype Classification',
                'y_target': y_genotype,
                'metrics': ['roc_auc', 'balanced_accuracy', 'mcc'],
                'primary_metric': 'roc_auc'
            }
        }

        # ---- run tasks ----
        results = {}
        all_p_values = []
        p_value_info = []

        for task_key, task in tasks.items():
            print(f"\n{task['label']}")
            print("-" * 40)

            y_t = task['y_target']
            classes = np.unique(y_t)
            if classes.size < 2:
                print("Insufficient classes; skipping.")
                continue

            preferred = ['xgb_gpu', 'xgb_cpu', 'lightgbm', 'rf_optimized']
            model_name = next((m for m in preferred if m in models), None)
            if model_name is None:
                print("No model available; skipping.")
                continue
            print(f"Using model: {model_name}")

            pipe = models[model_name]
            scoring_dict = create_scoring(classes)
            scoring_use = {k: v for k, v in scoring_dict.items() if k in task['metrics']}

            if model_name.startswith('xgb') and classes.size == 2:
                clf = pipe.named_steps['classifier']
                pos_weight = (y_t == 0).sum() / max(1, (y_t == 1).sum())
                try:
                    clf.set_params(scale_pos_weight=pos_weight)
                except Exception:
                    pass

            perm_res = permutation_test(pipe, X, y_t, scoring_use,
                                        n_perm=n_permutations, cv=cv_folds, seed=random_state)
            results[task_key] = {
                'task': task,
                'model': model_name,
                'metrics': perm_res
            }

            for metric, r in perm_res.items():
                all_p_values.append(r['p_value'])
                p_value_info.append({
                    'task': task_key,
                    'metric': metric,
                    'model': model_name,
                    'p_value': r['p_value']
                })
                print(f"  {metric}: {r['observed_score']:.3f} "
                      f"(p={r['p_value']:.3f}, z={r['effect_size_z']:.2f}, n_perm={r['n_permutations']})")

        # ---- per-age genotype analysis ----
        print("\nPer-Age Genotype Classification")
        print("-" * 40)
        per_age_results = {}
        for age_lbl in ['Young', 'Aged']:
            mask = (df['age'] == age_lbl).to_numpy()
            if mask.sum() < 10:
                continue
            X_age = X[mask]
            y_geno_age = df.loc[mask, 'genotype_enc'].to_numpy()
            if np.unique(y_geno_age).size < 2:
                continue
            print(f"{age_lbl} subjects (n={mask.sum()}):")

            model_name = next((m for m in ['xgb_gpu', 'xgb_cpu', 'lightgbm', 'rf_optimized'] if m in models), None)
            pipe = models[model_name]
            scoring_use = {'roc_auc': 'roc_auc', 'balanced_accuracy': 'balanced_accuracy'}
            age_perm = permutation_test(pipe, X_age, y_geno_age, scoring_use,
                                        n_perm=max(20, n_permutations // 2), cv=3, seed=random_state)
            per_age_results[age_lbl.lower()] = age_perm
            for metric, r in age_perm.items():
                all_p_values.append(r['p_value'])
                p_value_info.append({
                    'task': f'genotype_{age_lbl.lower()}',
                    'metric': metric,
                    'model': model_name,
                    'p_value': r['p_value']
                })
                print(f"  {metric}: {r['observed_score']:.3f} "
                      f"(p={r['p_value']:.3f}, n_perm={r['n_permutations']})")

        # ---- multiple testing (primary vs secondary) ----
        primary_p_values, primary_info = [], []
        secondary_p_values, secondary_info = [], []

        for info in p_value_info:
            task_name = info['task']
            metric_name = info['metric']
            is_primary = False
            if task_name in results:
                is_primary = (metric_name == results[task_name]['task']['primary_metric'])
            elif task_name.startswith('genotype_'):
                is_primary = (metric_name == 'roc_auc')
            if is_primary:
                primary_p_values.append(info['p_value'])
                primary_info.append(info)
            else:
                secondary_p_values.append(info['p_value'])
                secondary_info.append(info)

        if primary_p_values:
            _, fdr_primary, _, _ = multipletests(primary_p_values, method='fdr_bh')
            for i, info in enumerate(primary_info):
                info['fdr_p_value'] = float(fdr_primary[i])
                info['fdr_significant'] = bool(fdr_primary[i] < 0.05)

        for info in secondary_info:
            info['fdr_p_value'] = info['p_value']  # no correction for secondary
            info['fdr_significant'] = bool(info['p_value'] < 0.05)
            info['is_secondary'] = True

        print("\nMultiple Testing Correction Results")
        print("=" * 50)

        if primary_p_values:
            print("PRIMARY METRICS (FDR corrected):")
            print("-" * 35)
            for info in primary_info:
                stars = "***" if info['fdr_p_value'] < 0.001 else "**" if info['fdr_p_value'] < 0.01 else "*" if info['fdr_p_value'] < 0.05 else ""
                print(f"  {info['task']} - {info['metric']}: p={info['p_value']:.3f}, q={info['fdr_p_value']:.3f} {stars}")

        if secondary_p_values:
            print("\nSECONDARY METRICS (uncorrected):")
            print("-" * 35)
            for info in secondary_info:
                stars = "***" if info['p_value'] < 0.001 else "**" if info['p_value'] < 0.01 else "*" if info['p_value'] < 0.05 else ""
                print(f"  {info['task']} - {info['metric']}: p={info['p_value']:.3f} {stars}")

        all_p_values = primary_p_values + secondary_p_values
        p_value_info = primary_info + secondary_info

        total_time = perf_counter() - t0

        return {
            'results': results,
            'per_age_results': per_age_results,
            'multiple_testing': {
                'all_p_values': all_p_values,
                'p_value_info': p_value_info,
                'fdr_corrected': bool(all_p_values)
            },
            'dataset_info': {
                'n_subjects': int(len(df)),
                'n_features': int(len(self.features)),
                'group_distribution': df['group'].value_counts().to_dict()
            },
            'optimization_info': {
                'gpu_used': bool(gpu_available),
                'fast_mode': bool(fast_mode),
                'cv_folds': int(cv_folds),
                'total_time_seconds': float(total_time)
            },
            'label_encoders': label_encoders
        }

    def _plot_optimized_results(self, results, per_age_results, df, X, figsize):
        """Streamlined visualization for optimized results"""
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import RobustScaler

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Performance overview
        ax = axes[0, 0]
        tasks = list(results.keys())
        scores, p_values = [], []
        for task in tasks:
            task_results = results[task]['metrics']
            primary_metric = results[task]['task']['primary_metric']
            if primary_metric in task_results:
                scores.append(task_results[primary_metric]['observed_score'])
                p_values.append(task_results[primary_metric]['p_value'])
            else:
                scores.append(0)
                p_values.append(1)
        colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_values]
        bars = ax.bar(range(len(tasks)), scores, color=colors, alpha=0.7)
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([results[task]['task']['label'] for task in tasks], rotation=45)
        ax.set_ylabel('Classification Score')
        ax.set_title('Classification Performance Overview')
        for bar, score, p_val in zip(bars, scores, p_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{score:.3f}\np={p_val:.3f}', ha='center', va='bottom', fontsize=9)

        # 2. PCA visualization
        ax = axes[0, 1]
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        colors_dict = {'young_wt': '#4A90E2', 'young_app': '#1F5F99',
                       'aged_wt': '#E85D75', 'aged_app': '#B73E56'}
        for group in df['group'].unique():
            mask = df['group'] == group
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=colors_dict.get(group, 'gray'), label=group, alpha=0.7)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('PCA - Group Separation')
        ax.legend(fontsize=8)

        # 3. Per-age comparison
        if per_age_results:
            ax = axes[1, 0]
            ages = list(per_age_results.keys())
            for i, age in enumerate(ages):
                if 'roc_auc' in per_age_results[age]:
                    score = per_age_results[age]['roc_auc']['observed_score']
                    p_val = per_age_results[age]['roc_auc']['p_value']
                    color = 'green' if p_val < 0.05 else 'orange' if p_val < 0.1 else 'red'
                    bar = ax.bar(i, score, color=color, alpha=0.7)
                    ax.text(i, score + 0.02, f'{score:.3f}\np={p_val:.3f}',
                            ha='center', va='bottom')
            ax.set_xticks(range(len(ages)))
            ax.set_xticklabels([age.title() for age in ages])
            ax.set_ylabel('ROC-AUC')
            ax.set_title('Genotype Classification by Age')
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

        # 4. Effect sizes
        ax = axes[1, 1]
        all_z_scores, all_labels = [], []
        for task_name, task_data in results.items():
            for metric_name, metric_data in task_data['metrics'].items():
                all_z_scores.append(metric_data['effect_size_z'])
                all_labels.append(f"{task_name[:8]}\n{metric_name}")
        if all_z_scores:
            colors = ['green' if abs(z) > 1.96 else 'gray' for z in all_z_scores]
            ax.barh(range(len(all_z_scores)), all_z_scores, color=colors, alpha=0.7)
            ax.set_yticks(range(len(all_z_scores)))
            ax.set_yticklabels(all_labels, fontsize=8)
            ax.set_xlabel('Effect Size (z-score)')
            ax.set_title('Classification Effect Sizes')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.axvline(x=1.96, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=-1.96, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

    def run_all(self, alpha=0.05, plot=True, figsize=(15, 10)):
        """
        Run the complete multi-group analysis workflow.
        """
        print("Starting Multi-Group Analysis")
        print("=" * 50)
        print(f"Groups: {', '.join(self.group_names)}")
        print(f"Features: {len(self.features)}")
        print(f"Significance level: {alpha}")
        print()

        kw_results = self.kruskal_wallis_analysis(alpha=alpha, plot=plot, figsize=figsize)
        art_anova_results = self.art_anova_analysis(alpha=alpha, plot=plot, figsize=figsize)
        multivariate_results = self.multivariate_analysis(alpha=alpha)
        classifier_results = self.classification_analysis(n_permutations=0, use_gpu=True, fast_mode=True, plot=plot, figsize=figsize)

        # Only call additional plotting if classifier_results is available and has the required data
        if classifier_results and 'results' in classifier_results and 'per_age_results' in classifier_results:
            data_list = []
            for group_name, data in self.groups.items():
                complete_cases = data[self.features].dropna()
                if len(complete_cases) == 0:
                    continue
                group_data = complete_cases.copy()
                group_data['group'] = group_name
                group_data['age'] = 'Young' if 'young' in group_name.lower() else 'Aged'
                group_data['genotype'] = 'APP' if 'app' in group_name.lower() else 'WT'
                group_data['subject_id'] = [f"{group_name}_{idx}" for idx in complete_cases.index]
                data_list.append(group_data)

            if data_list:
                df = pd.concat(data_list, ignore_index=True)
                X = df[self.features].values
                self._plot_optimized_results(
                    results=classifier_results['results'],
                    per_age_results=classifier_results['per_age_results'],
                    df=df,
                    X=X,
                    figsize=figsize
                )

        return {
            'groups_info': {
                'group_names': self.group_names,
                'features': self.features,
                'group_sizes': {name: len(data) for name, data in self.groups.items()}
            }
        }
