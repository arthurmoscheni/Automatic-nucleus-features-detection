import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from scipy.stats import kruskal
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from scikit_posthocs import posthoc_dunn
from scipy.spatial.distance import pdist, squareform

# local utils
from utils.stats import (
    cliffs_delta, cliffs_delta_with_ci, permutation_test_delta_diff,
    aligned_rank_transform, build_long_dataframe, r2_from_distance
)

import warnings
warnings.filterwarnings('ignore')


class MultiGroupAnalyzer:
    """
    Computes all statistics for the 4-group workflow (no visualization).
    Groups: young_wt, young_app, aged_wt, aged_app.
    """

    def __init__(self, groups, young_wt_data: pd.DataFrame, young_app_data: pd.DataFrame,
                 aged_wt_data: pd.DataFrame, aged_app_data: pd.DataFrame,
                 features: List[str]):
        self.group_names = list(groups.keys())

        self.young_wt = young_wt_data
        self.young_app = young_app_data
        self.aged_wt = aged_wt_data
        self.aged_app = aged_app_data
        self.features = features

        self.groups: Dict[str, pd.DataFrame] = groups
        self.group_names = list(groups.keys())

    # ------------- Kruskal–Wallis + Dunn + Cliff's -------------
    def kruskal_wallis_analysis(self, alpha: float = 0.05):
        

        results = {}
        skipped = []

        # Option 1 (recommended): restrict to features present in all groups
        common = set(self.features)
        for gdf in self.groups.values():
            common &= set(gdf.columns)
        features_to_test = [f for f in self.features if f in common]
        if len(features_to_test) < len(self.features):
            missing = sorted(set(self.features) - set(features_to_test))
            print(f"[KW] Skipping {len(missing)} features not present in all groups (e.g. {missing[:5]}...)")

        for feature in features_to_test:
            group_data = []
            group_order = []
            group_labels = []

            # Collect clean numeric arrays for each group (must be non-empty)
            for group_name, df in self.groups.items():
                if feature not in df.columns:
                    continue
                vals = pd.to_numeric(df[feature], errors="coerce").dropna().values
                if len(vals) == 0:
                    # empty sample → cannot test this feature
                    break
                group_data.append(vals)
                group_order.append(group_name)
                group_labels.extend([group_name] * len(vals))
            else:
                # Only executes if we didn't break (i.e., all 4 groups provided non-empty data)
                if len(group_data) != len(self.groups):
                    # Safety: require all 4 groups
                    skipped.append(feature)
                    continue

                # Kruskal–Wallis
                kw_stat, kw_pvalue = kruskal(*group_data)

                # Effect sizes (guard against tiny n)
                n_total = sum(len(g) for g in group_data)
                k = len(group_data)
                if n_total > k:
                    eta_squared = (kw_stat - k + 1) / (n_total - k)
                    epsilon_squared = kw_stat / max(n_total - 1, 1)
                else:
                    eta_squared = np.nan
                    epsilon_squared = np.nan

                # Dunn post-hoc (FDR)
                all_values = np.concatenate(group_data)
                df_post = pd.DataFrame({"value": all_values, "group": group_labels})
                dunn_results = posthoc_dunn(
                    df_post, val_col="value", group_col="group", p_adjust="fdr_bh"
                )
                # Reindex to the actual order for nicer heatmaps
                dunn_results = dunn_results.reindex(index=group_order, columns=group_order)

                # Pairwise Cliff’s delta (use the class method!)
                cd_mat = pd.DataFrame(index=group_order, columns=group_order, dtype=float)
                for i, g1 in enumerate(group_order):
                    for j, g2 in enumerate(group_order):
                        if i == j:
                            cd_mat.loc[g1, g2] = np.nan
                        else:
                            x1 = pd.to_numeric(self.groups[g1][feature], errors="coerce").dropna().values
                            x2 = pd.to_numeric(self.groups[g2][feature], errors="coerce").dropna().values
                            cd_mat.loc[g1, g2] = cliffs_delta(x1, x2)

                results[feature] = {
                    "kw_statistic": kw_stat,
                    "kw_pvalue": kw_pvalue,
                    "eta_squared": eta_squared,
                    "epsilon_squared": epsilon_squared,
                    "significant": kw_pvalue < alpha,
                    "dunn_pvalues": dunn_results,
                    "cliffs_delta": cd_mat,
                    "group_data": group_data,   # list of np arrays in the SAME order as group_order
                    "group_order": group_order, # <- store actual order used
                }
                continue

            # If we hit break (some group empty), record skip
            skipped.append(feature)

        if skipped:
            print(f"[KW] Skipped {len(skipped)} features with missing/empty data in ≥1 group "
                f"(e.g. {skipped[:5]}...)")

        return results


    # ------------- ART ANOVA -------------
    def art_anova_analysis(self, alpha: float = 0.05):
        results = {}

        for feature in self.features:
            # Build long dataframe (value, age, genotype, group)
            rows = []
            if feature in self.young_wt.columns:
                rows += [{'value': v, 'age': 'Young', 'genotype': 'WT',  'group': 'young_wt'}
                         for v in self.young_wt[feature].dropna()]
            if feature in self.young_app.columns:
                rows += [{'value': v, 'age': 'Young', 'genotype': 'APP', 'group': 'young_app'}
                         for v in self.young_app[feature].dropna()]
            if feature in self.aged_wt.columns:
                rows += [{'value': v, 'age': 'Aged',  'genotype': 'WT',  'group': 'aged_wt'}
                         for v in self.aged_wt[feature].dropna()]
            if feature in self.aged_app.columns:
                rows += [{'value': v, 'age': 'Aged',  'genotype': 'APP', 'group': 'aged_app'}
                         for v in self.aged_app[feature].dropna()]

            if not rows:
                continue
            df = pd.DataFrame(rows)
            if df['group'].nunique() < 4:
                continue

            # ART
            df_art, art_cols = aligned_rank_transform(df, 'value', ['age', 'genotype'])

            anova_results = {}
            for art_col in art_cols:
                # ANOVA on ranks
                df_art[art_col] = pd.to_numeric(df_art[art_col], errors='coerce')
                dclean = df_art.dropna(subset=[art_col])
                if dclean.empty:
                    continue
                model = ols(f"Q('{art_col}') ~ C(age) * C(genotype)", data=dclean).fit()
                tab = anova_lm(model, typ=2)

                # Map column to effect row
                if 'age:genotype' in art_col:
                    row = 'C(age):C(genotype)'
                    key = 'age_x_genotype'
                elif 'age' in art_col and 'genotype' not in art_col:
                    row = 'C(age)'
                    key = 'age'
                elif 'genotype' in art_col and 'age' not in art_col:
                    row = 'C(genotype)'
                    key = 'genotype'
                else:
                    row = None
                    key = None

                if row and row in tab.index:
                    f = tab.loc[row, 'F']
                    p = tab.loc[row, 'PR(>F)']
                    ss_eff = tab.loc[row, 'sum_sq']
                    ss_err = tab.loc['Residual', 'sum_sq']
                    eta2p = ss_eff / (ss_eff + ss_err) if (ss_eff + ss_err) > 0 else np.nan
                    anova_results[key] = {
                        'F_statistic': f,
                        'p_value': p,
                        'partial_eta_squared': eta2p,
                        'significant': p < alpha,
                        'full_anova': tab
                    }

            # Post-hoc contrasts if interaction is significant
            if 'age_x_genotype' in anova_results and anova_results['age_x_genotype']['significant']:
                dff = df
                y_wt  = dff[(dff.age == 'Young') & (dff.genotype == 'WT')]['value'].values
                y_app = dff[(dff.age == 'Young') & (dff.genotype == 'APP')]['value'].values
                a_wt  = dff[(dff.age == 'Aged')  & (dff.genotype == 'WT')]['value'].values
                a_app = dff[(dff.age == 'Aged')  & (dff.genotype == 'APP')]['value'].values

                dy, dy_lo, dy_hi = cliffs_delta_with_ci(y_app, y_wt)
                da, da_lo, da_hi = cliffs_delta_with_ci(a_app, a_wt)
                d_diff = da - dy if (np.isfinite(da) and np.isfinite(dy)) else np.nan
                p_perm, perm_dist = permutation_test_delta_diff(y_app, y_wt, a_app, a_wt)

                anova_results['within_age_contrasts'] = {
                    'delta_young': dy, 'delta_young_ci': (dy_lo, dy_hi),
                    'delta_aged' : da, 'delta_aged_ci' : (da_lo, da_hi),
                    'delta_difference': d_diff,
                    'permutation_p': p_perm,
                    'permutation_distribution': perm_dist
                }

            results[feature] = {
                'anova_results': anova_results,
                'data': df,
                'art_data': df_art,
                'group_means': df.groupby(['age', 'genotype'])['value'].agg(['mean', 'std', 'count'])
            }

        # Summaries + FDR over features per effect
        summary_rows = []
        for feature, res in results.items():
            for eff, st in res['anova_results'].items():
                if eff == 'within_age_contrasts':
                    continue
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

        return {'summary': summary_df, 'detailed_results': results, 'alpha': alpha}

    # ------------- Multivariate (PERMANOVA-like) -------------
    def multivariate_analysis(self, n_permutations: int = 0):
        dfc = build_long_dataframe(self.groups, self.features)
        if dfc.empty:
            return None

        X = dfc[self.features].values
        D = squareform(pdist(X, metric='euclidean'))
        geno = dfc['genotype'].values
        age  = dfc['age'].values

        r2_genotype_overall = r2_from_distance(D, geno)
        y_mask = (age == 'Young')
        a_mask = (age == 'Aged')
        if np.sum(y_mask) > 1 and np.sum(a_mask) > 1:
            Dy = D[np.ix_(y_mask, y_mask)]
            Da = D[np.ix_(a_mask, a_mask)]
            r2_genotype_young = r2_from_distance(Dy, geno[y_mask])
            r2_genotype_aged  = r2_from_distance(Da, geno[a_mask])
            delta_r2 = r2_genotype_aged - r2_genotype_young

            # permutations by shuffling genotype within age
            rng = np.random.default_rng(42)
            perm_d = []
            for _ in range(n_permutations):
                gy = rng.permutation(geno[y_mask])
                ga = rng.permutation(geno[a_mask])
                r2y = r2_from_distance(Dy, gy)
                r2a = r2_from_distance(Da, ga)
                perm_d.append(r2a - r2y)
            perm_d = np.asarray(perm_d, float)
            p_val = (1 + np.sum(np.abs(perm_d) >= np.abs(delta_r2))) / (1 + len(perm_d))
        else:
            r2_genotype_young = np.nan
            r2_genotype_aged  = np.nan
            delta_r2 = np.nan
            p_val = np.nan
            perm_d = np.array([])

        r2_age_overall = r2_from_distance(D, age)

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

    # ------------- Classification (no plots) -------------
    def classification_analysis(self, random_state: int = 42, n_permutations: int = 0,
                                use_gpu: bool = True, fast_mode: bool = True):
        """
        Same modeling logic as before (XGB/LightGBM/RF), returns metrics & permutation p-values.
        """
        from time import perf_counter
        from sklearn.preprocessing import RobustScaler, QuantileTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import StratifiedKFold, cross_validate
        from sklearn.metrics import make_scorer, matthews_corrcoef
        from sklearn.ensemble import RandomForestClassifier

        # Optional libs
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

        try:
            import lightgbm as lgb

            lgb_available = True
        except Exception:
            lgb = None
            lgb_available = False

        t0 = perf_counter()
        df = build_long_dataframe(self.groups, self.features)
        if df.empty:
            return None

        X = df[self.features].to_numpy(dtype=float)

        group_mapping = {'young_wt': 0, 'young_app': 1, 'aged_wt': 2, 'aged_app': 3}
        age_mapping = {'Young': 0, 'Aged': 1}
        genotype_mapping = {'WT': 0, 'APP': 1}

        df['group_enc'] = df['group'].map(group_mapping)
        df['age_enc'] = df['age'].map(age_mapping)
        df['genotype_enc'] = df['genotype'].map(genotype_mapping)

        y_group = df['group_enc'].to_numpy()
        y_age = df['age_enc'].to_numpy()
        y_genotype = df['genotype_enc'].to_numpy()

        # Preprocessing
        def create_preprocessor():
            return ColumnTransformer([
                ('robust_quantile', Pipeline([
                    ('robust', RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25, 75))),
                    ('quantile', QuantileTransformer(output_distribution='normal',
                                                    n_quantiles=min(1000, max(10, X.shape[0] // 2)),
                                                    random_state=random_state))
                ]), list(range(len(self.features))))
            ])

        # Models
        cv_folds = 3 if fast_mode else 5
        models = {}
        if xgb is not None:
            models['xgb_gpu' if gpu_available else 'xgb_cpu'] = Pipeline([
                ('preprocessor', create_preprocessor()),
                ('classifier', xgb.XGBClassifier(
                    tree_method='gpu_hist' if gpu_available else 'hist',
                    gpu_id=0 if gpu_available else None,
                    n_estimators=100 if fast_mode else 200,
                    max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0, random_state=random_state,
                    n_jobs=1 if gpu_available else -1, verbosity=1
                ))
            ])
        if lgb_available:
            models['lightgbm'] = Pipeline([
                ('preprocessor', create_preprocessor()),
                ('classifier', lgb.LGBMClassifier(
                    n_estimators=100 if fast_mode else 200,
                    max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=0.0, random_state=random_state,
                    n_jobs=-1, verbose=1,
                    class_weight='balanced' if len(np.unique(y_group)) > 2 else None
                ))
            ])
        models['rf_optimized'] = Pipeline([
            ('preprocessor', create_preprocessor()),
            ('classifier', RandomForestClassifier(
                n_estimators=100 if fast_mode else 200,
                max_depth=10, min_samples_split=5, min_samples_leaf=2,
                class_weight='balanced', random_state=random_state, n_jobs=-1
            ))
        ])

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

        def permutation_test(pipeline, X_, y_, scoring_dict, n_perm=100, cv=3, seed=42):
            rng = np.random.default_rng(seed)
            cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

            obs_cv = cross_validate(pipeline, X_, y_, cv=cv_split, scoring=scoring_dict,
                                    return_train_score=False, n_jobs=-1, error_score='raise')
            obs_mean = {k.replace('test_', ''): float(np.mean(v)) for k, v in obs_cv.items() if k.startswith('test_')}

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

        tasks = {
            'four_groups': {'label': '4-Group Classification', 'y': y_group,
                            'metrics': ['balanced_accuracy', 'macro_f1'], 'primary': 'balanced_accuracy'},
            'age': {'label': 'Age Classification', 'y': y_age,
                    'metrics': ['roc_auc', 'balanced_accuracy', 'mcc'], 'primary': 'roc_auc'},
            'genotype': {'label': 'Genotype Classification', 'y': y_genotype,
                         'metrics': ['roc_auc', 'balanced_accuracy', 'mcc'], 'primary': 'roc_auc'},
        }

        results = {}
        primary_p, primary_info = [], []
        secondary_p, secondary_info = [], []

        for key, task in tasks.items():
            y_t = task['y']
            classes = np.unique(y_t)
            if classes.size < 2:
                continue

            model_name = next((m for m in ['xgb_gpu', 'xgb_cpu', 'lightgbm', 'rf_optimized'] if m in models), None)
            if model_name is None:
                continue
            pipe = models[model_name]
            scoring = create_scoring(classes)
            scoring = {k: v for k, v in scoring.items() if k in task['metrics']}

            metrics = permutation_test(pipe, X, y_t, scoring, n_perm=n_permutations, cv=cv_folds, seed=random_state)
            results[key] = {'task': task, 'model': model_name, 'metrics': metrics}

            for metric, r in metrics.items():
                if metric == task['primary']:
                    primary_p.append(r['p_value']); primary_info.append({'task': key, 'metric': metric, 'p_value': r['p_value']})
                else:
                    secondary_p.append(r['p_value']); secondary_info.append({'task': key, 'metric': metric, 'p_value': r['p_value']})

        # FDR on primary metrics only
        if primary_p:
            _, q, _, _ = multipletests(primary_p, method='fdr_bh')
            for info, qv in zip(primary_info, q):
                info['fdr_p_value'] = float(qv)
                info['fdr_significant'] = bool(qv < 0.05)

        for info in secondary_info:
            info['fdr_p_value'] = info['p_value']
            info['fdr_significant'] = bool(info['p_value'] < 0.05)
            info['is_secondary'] = True

        total_time = perf_counter() - t0
        return {
            'results': results,
            'multiple_testing': {
                'all_p_values': primary_p + secondary_p,
                'p_value_info': primary_info + secondary_info,
                'fdr_corrected': bool(primary_p)
            },
            'dataset_info': {
                'n_subjects': int(len(df)),
                'n_features': int(len(self.features)),
                'group_distribution': df['group'].value_counts().to_dict()
            },
            'optimization_info': {
                'gpu_used': bool('xgb_gpu' in models),
                'fast_mode': bool(fast_mode),
                'cv_folds': int(cv_folds),
                'total_time_seconds': float(total_time)
            }
        }

    # ------------- Orchestrator (no plots) -------------
    def run_all(self, alpha=0.05, n_permutations=0):
        kw = self.kruskal_wallis_analysis(alpha=alpha)
        art = self.art_anova_analysis(alpha=alpha)
        mv = self.multivariate_analysis(n_permutations=n_permutations)
        clf = self.classification_analysis(n_permutations=n_permutations, use_gpu=True, fast_mode=True)
        return {'kw': kw, 'art': art, 'multivariate': mv, 'classification': clf}
