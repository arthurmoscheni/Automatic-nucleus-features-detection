import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata
from typing import Dict, List, Tuple

# ---------- Effect sizes & helpers ----------

def cliffs_delta(x, y) -> float:
    """Cliff's delta via Mann–Whitney U (two-sided). Positive => x > y."""
    x = pd.Series(x).dropna().values
    y = pd.Series(y).dropna().values
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    U, _ = mannwhitneyu(x, y, alternative="two-sided")
    return 2.0 * U / (n1 * n2) - 1.0


def cliffs_delta_with_ci(x, y, n_bootstrap=1000, random_state=42):
    """Cliff's delta with bootstrap 95% CI."""
    rng = np.random.default_rng(random_state)
    x = pd.Series(x).dropna().values
    y = pd.Series(y).dropna().values
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan, np.nan
    delta = cliffs_delta(x, y)
    if n_bootstrap <= 0:
        return delta, np.nan, np.nan
    boot = []
    for _ in range(n_bootstrap):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        boot.append(cliffs_delta(xb, yb))
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    return delta, ci_low, ci_high


def permutation_test_delta_diff(young_app, young_wt, aged_app, aged_wt,
                                n_perm=10000, random_state=42):
    """
    Permutation test for Δδ = δ_aged − δ_young, shuffling labels within each age.
    Returns p-value and permutation distribution (list).
    """
    rng = np.random.default_rng(random_state)

    def _delta(x_app, x_wt):
        return cliffs_delta(x_app, x_wt)

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


# ---------- ART ANOVA helper ----------

def aligned_rank_transform(df: pd.DataFrame, value_col: str, factors: List[str]):
    """
    Aligned Rank Transform (ART) for two factors (main effects + interaction).
    Returns transformed dataframe and an iterable of ART column names.
    """
    df_art = df.copy()
    effects = factors + [f"{factors[0]}:{factors[1]}"]  # main effects + interaction
    art_data: Dict[str, np.ndarray] = {}

    for effect in effects:
        if ':' in effect:  # Interaction
            factor1, factor2 = effect.split(':')
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


# ---------- Data prep helpers ----------

def build_long_dataframe(groups: Dict[str, pd.DataFrame], features: List[str]) -> pd.DataFrame:
    """
    Build a long dataframe with all complete cases across features.
    Adds columns: group, age, genotype, subject_id, and the feature columns.
    """
    data_list = []
    for group_name, data in groups.items():
        complete = data[features].dropna()
        if complete.empty:
            continue
        gdf = complete.copy()
        gdf['group'] = group_name
        gdf['age'] = 'Young' if 'young' in group_name.lower() else 'Aged'
        gdf['genotype'] = 'APP' if 'app' in group_name.lower() else 'WT'
        gdf['subject_id'] = [f"{group_name}_{idx}" for idx in complete.index]
        data_list.append(gdf)

    if not data_list:
        return pd.DataFrame(columns=features + ['group', 'age', 'genotype', 'subject_id'])

    return pd.concat(data_list, ignore_index=True)


def r2_from_distance(D: np.ndarray, labels: np.ndarray) -> float:
    """
    Simple R² from a distance matrix for a grouping factor.
    """
    labels = np.asarray(labels)
    n = len(labels)
    groups = np.unique(labels)
    ss_total = np.sum(D**2) / n
    ss_within = 0.0
    for g in groups:
        mask = labels == g
        if np.sum(mask) > 1:
            G = D[np.ix_(mask, mask)]
            ss_within += np.sum(G**2) / np.sum(mask)
    ss_between = ss_total - ss_within
    return ss_between / ss_total if ss_total > 0 else 0.0
