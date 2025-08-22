from typing import List, Optional
import pandas as pd
from typing import Dict
from analyzers.multigroup_analyzer import MultiGroupAnalyzer
from utils.stats import build_long_dataframe
from viz.multigroup_viz import MultiGroupViz


class MultiGroupPipeline:
    """
    Orchestrates analyzer + viz. Keeps your original logic,
    just split cleanly into layers.
    """

    def __init__(self, groups: Dict[str, pd.DataFrame], features: List[str]):
        self.groups = groups
        self.group_names = list(groups.keys())
        g = self.group_names
        self.analyzer = MultiGroupAnalyzer(groups, groups[g[0]], groups[g[1]], groups[g[2]], groups[g[3]], features)
        self.features = features
        # self.groups = {
        #     'young_wt': young_wt,
        #     'young_app': young_app,
        #     'aged_wt': aged_wt,
        #     'aged_app': aged_app,
        # }

    def run(self, alpha: float = 0.05, n_permutations: int = 0, plot: bool = True):
        results = self.analyzer.run_all(alpha=alpha, n_permutations=n_permutations)

        if plot:
            viz = MultiGroupViz(self.groups)
            # KW plots + Dunn heatmaps
            viz.plot_kw(results['kw'], alpha=alpha)
            # ART plots
            viz.plot_art_interactions(results['art'])
            viz.plot_art_eta_heatmap(results['art'])
            # Classification summary plots
            df = build_long_dataframe(self.groups, self.features)
            if not df.empty:
                if 'group' in df.columns:
                    df['group'] = pd.Categorical(df['group'], categories=self.group_names, ordered=True)
                X = df[self.features].values
                viz.plot_classification_summary(results['classification'], df, X)
                
        return results
