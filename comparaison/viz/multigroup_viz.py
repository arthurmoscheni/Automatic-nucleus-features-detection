import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from typing import Dict

class MultiGroupViz:
    """All plotting utilities for the 4-group workflow."""
    
    def __init__(self, groups: Dict[str, pd.DataFrame]):
        self.groups = groups
        self.group_names = list(groups.keys())
        # Define colors for each group
        self.colors = {
            group_name: plt.cm.Set1(i) for i, group_name in enumerate(self.group_names)
        }
        


    def plot_kw(self, kw_results, figsize=(15, 10), alpha=0.05):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        results = kw_results
        if not results:
            print("[KW Viz] No results to plot.")
            return

        n_features = len(results)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_1d(axes).ravel()

        for ax, (feature, res) in zip(axes, results.items()):
            group_data  = res["group_data"]
            group_order = res["group_order"]

            positions = np.arange(len(group_data))
            box_parts = ax.boxplot(group_data, positions=positions, patch_artist=True, widths=0.6, showfliers=True)

            # Color boxes according to the *actual* order
            for patch, gname in zip(box_parts["boxes"], group_order):
                patch.set_facecolor(self.colors.get(gname, "gray"))
                patch.set_alpha(0.7)
                patch.set_edgecolor("black")
                patch.set_linewidth(1)

            for element in ["whiskers", "fliers", "medians", "caps"]:
                plt.setp(box_parts[element], color="black")
            plt.setp(box_parts["medians"], linewidth=2)

            ax.set_xticks(positions)
            ax.set_xticklabels(group_order, rotation=45, ha="right")
            ax.set_ylabel(feature)
            ax.set_title(f"{feature}\nKW p={res['kw_pvalue']:.1e}, η²={res['eta_squared']:.3f}")
            ax.grid(axis="y", alpha=0.3)

        # Hide unused axes
        for ax in axes[len(results):]:
            ax.set_visible(False)

        # legend_elements = [
        #     Patch(facecolor=self.colors.get(group_name, "gray"), alpha=0.7, label=group_name)
        #     for group_name in self.group_names
        # ]
        # fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98))
        plt.tight_layout()
        plt.show()


    def plot_art_interactions(self, art_results, figsize=(15, 10)):
        results = art_results['detailed_results']
        if not results:
            return
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
            bits = [feature]
            if 'age' in res['anova_results']:
                bits.append(f"Age p={res['anova_results']['age']['p_value']:.2e}")
            if 'genotype' in res['anova_results']:
                bits.append(f"Geno p={res['anova_results']['genotype']['p_value']:.2e}")
            if 'age_x_genotype' in res['anova_results']:
                bits.append(f"Int p={res['anova_results']['age_x_genotype']['p_value']:.2e}")
            ax.set_title(" | ".join(bits), fontsize=9)
            ax.legend()
        for ax in axes[len(results):]:
            ax.set_visible(False)
        plt.tight_layout(); plt.show()

    def plot_art_eta_heatmap(self, art_results):
        summary_df = art_results['summary']
        if summary_df.empty:
            return
        pivot_eta = summary_df.pivot(index='Feature', columns='Effect', values='Partial_Eta_Squared')
        fig, ax = plt.subplots(1, 1, figsize=(12, max(6, len(pivot_eta) * 0.6)))
        annot = pivot_eta.copy().astype(str)
        for i in range(len(pivot_eta)):
            for j in range(len(pivot_eta.columns)):
                val = pivot_eta.iloc[i, j]
                annot.iloc[i, j] = f"{val:.4f}" if pd.notna(val) else ""
        sns.heatmap(pivot_eta, annot=annot, fmt='', cmap='RdYlBu_r',
                    cbar_kws={'label': 'Partial η²'}, ax=ax, annot_kws={'fontsize': 9})
        ax.set_title('ART ANOVA Effect Sizes (η²)')
        plt.tight_layout(); plt.show()

    # ---- Compact classification plots (moved from the class) ----
    def plot_classification_summary(self, clf_results, df, X, figsize=(15, 10)):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import RobustScaler

        results = clf_results['results']
        per_age = clf_results.get('per_age_results', {})

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1) Performance overview
        ax = axes[0, 0]
        tasks = list(results.keys())
        scores, p_values = [], []
        labels = []
        for task in tasks:
            task_results = results[task]['metrics']
            primary = results[task]['task']['primary']
            if primary in task_results:
                scores.append(task_results[primary]['observed_score'])
                p_values.append(task_results[primary]['p_value'])
            else:
                scores.append(0); p_values.append(1)
            labels.append(results[task]['task']['label'])
        colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_values]
        bars = ax.bar(range(len(tasks)), scores, color=colors, alpha=0.7)
        ax.set_xticks(range(len(tasks))); ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel('Score'); ax.set_title('Classification Overview')
        for bar, score, p_val in zip(bars, scores, p_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{score:.3f}\np={p_val:.3f}', ha='center', va='bottom', fontsize=9)

        # 2) PCA scatter
        ax = axes[0, 1]
        scaler = RobustScaler(); Xs = scaler.fit_transform(X)
        pca = PCA(n_components=2, random_state=42)
        Xp = pca.fit_transform(Xs)
        colors_dict = {group: self.colors[group] for group in self.group_names}
        for group in df['group'].unique():
            mask = df['group'] == group
            ax.scatter(Xp[mask, 0], Xp[mask, 1], c=colors_dict.get(group, 'gray'),
                       label=group, alpha=0.7)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('PCA - Group Separation'); ax.legend(fontsize=8)

        # 3) Per-age ROC-AUC bars (if provided)
        ax = axes[1, 0]
        if per_age:
            for i, age in enumerate(['young', 'aged']):
                if age in per_age and 'roc_auc' in per_age[age]:
                    score = per_age[age]['roc_auc']['observed_score']
                    p_val = per_age[age]['roc_auc']['p_value']
                    color = 'green' if p_val < 0.05 else 'orange' if p_val < 0.1 else 'red'
                    ax.bar(i, score, color=color, alpha=0.7)
                    ax.text(i, score + 0.02, f'{score:.3f}\np={p_val:.3f}', ha='center', va='bottom')
            ax.set_xticks([0, 1]); ax.set_xticklabels(['Young', 'Aged'])
            ax.set_ylabel('ROC-AUC'); ax.set_title('Genotype Classification by Age'); ax.axhline(0.5, color='red', ls='--', alpha=0.5)

        # 4) Effect sizes (z-scores) among metrics
        ax = axes[1, 1]
        all_z, labels = [], []
        for task_name, task_data in results.items():
            for metric_name, metric_data in task_data['metrics'].items():
                all_z.append(metric_data['effect_size_z'])
                labels.append(f"{task_name[:8]}\n{metric_name}")
        if all_z:
            colors = ['green' if abs(z) > 1.96 else 'gray' for z in all_z]
            ax.barh(range(len(all_z)), all_z, color=colors, alpha=0.7)
            ax.set_yticks(range(len(all_z))); ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Effect Size (z)'); ax.set_title('Classification Effect Sizes')
            ax.axvline(0, color='black', ls='-', alpha=0.3)
            ax.axvline(1.96, color='red', ls='--', alpha=0.5); ax.axvline(-1.96, color='red', ls='--', alpha=0.5)

        plt.tight_layout(); plt.show()
