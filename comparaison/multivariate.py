import numpy as np
from math import comb
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import rankdata
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


class GlobalMultivariateTests:
    """
    Distance-based PERMANOVA + MMD + RF classification tests,
    with animal-aware aggregation and permutations when feasible.
    """

    def __init__(self, df1, df2, group1=None, group2=None, features=None,
                 standardize=True, drop_constant=True):
        """Initialize with two dataframes and preprocessing options."""
        self.df1 = df1.copy()
        self.df2 = df2.copy()
        self.label1 = group1 if group1 is not None else 'Pop1'
        self.label2 = group2 if group2 is not None else 'Pop2'
        self.features = features if features is not None else df1.columns.tolist()

        # Extract and combine data
        X1_raw = self.df1[self.features].to_numpy(float)
        X2_raw = self.df2[self.features].to_numpy(float)
        X_raw = np.vstack([X1_raw, X2_raw])

        # Drop constant features
        if drop_constant:
            std = X_raw.std(axis=0, ddof=1)
            keep = std > 0
            if not np.all(keep):
                dropped = [f for f, k in zip(self.features, keep) if not k]
                print(f"[WARN] Dropping constant features: {dropped}")
            self.features = [f for f, k in zip(self.features, keep) if k]
            X1_raw, X2_raw = X1_raw[:, keep], X2_raw[:, keep]
            X_raw = X_raw[:, keep]

        # Standardize if requested
        self.scaler = None
        if standardize:
            self.scaler = StandardScaler().fit(X_raw)
            X1 = self.scaler.transform(X1_raw)
            X2 = self.scaler.transform(X2_raw)
        else:
            X1, X2 = X1_raw, X2_raw

        self.X1 = X1
        self.X2 = X2
        self.X = np.vstack([X1, X2])
        self.y = np.array([0] * len(X1) + [1] * len(X2))

    def _robust_transform(self, X, method="rank", robust_scale=True, winsor=None):
        """Apply robust transformations to data."""
        Z = X.astype(float).copy()
        
        if method == "rank":
            for j in range(Z.shape[1]):
                r = rankdata(Z[:, j], method="average")
                Z[:, j] = (r - 0.5) / Z.shape[0]
        elif method != "none":
            raise ValueError("Unknown transform method")
        
        if winsor is not None and 0 < winsor < 0.5:
            for j in range(Z.shape[1]):
                Z[:, j] = winsorize(Z[:, j], limits=winsor)
        
        if robust_scale:
            med = np.nanmedian(Z, axis=0)
            q1 = np.nanpercentile(Z, 25, axis=0)
            q3 = np.nanpercentile(Z, 75, axis=0)
            iqr = np.where((q3 - q1) == 0, 1.0, (q3 - q1))
            Z = (Z - med) / iqr
        
        return Z

    def _aggregate_by_animal(self, X, y, animal_ids, method='mean'):
        """Aggregate data by animal using specified method."""
        import pandas as pd
        
        df = pd.DataFrame(X, columns=self.features)
        df['animal'] = np.asarray(animal_ids)
        df['y'] = np.asarray(y)
        
        agg = {f: method for f in self.features}
        agg['y'] = 'first'  # assumes consistent label per animal
        
        A = df.groupby('animal').agg(agg)
        X_agg = A[self.features].to_numpy(float)
        y_agg = A['y'].to_numpy()
        animal_order = A.index.to_numpy()
        
        return X_agg, y_agg, animal_order

    def _perm_count_binary(self, y_agg):
        """Count number of distinct labelings for permutation test feasibility."""
        u, cnts = np.unique(y_agg, return_counts=True)
        if len(u) != 2:
            return 0, np.inf
        L = comb(cnts.sum(), cnts[0])
        pmin = 1.0 / (L + 1)
        return L, pmin

    def permanova_descriptive(self, metric="euclidean", transform="rank", 
                            robust_scale=True, winsor=0.01, animal_ids=None, 
                            aggregate_method='mean'):
        """
        PERMANOVA descriptive statistics without permutation test.
        Returns F-statistic, R², and additional information.
        """
        # Data preprocessing
        Xr = self._robust_transform(self.X, method=transform, 
                                  robust_scale=robust_scale, winsor=winsor)
        y = self.y.copy()
        
        # Optional animal aggregation
        if animal_ids is not None:
            Xr, y, _ = self._aggregate_by_animal(Xr, y, animal_ids, method=aggregate_method)
        
        # Distance matrix and Gower-centering
        D = squareform(pdist(Xr, metric=metric))
        n = D.shape[0]
        A = -0.5 * (D**2)
        H = np.eye(n) - np.ones((n, n))/n
        G = H @ A @ H
        
        # Design matrix and sum of squares
        g = y.astype(int)
        Xd = np.column_stack([np.ones(n), g])
        XtX_inv = np.linalg.pinv(Xd.T @ Xd)
        H1 = Xd @ XtX_inv @ Xd.T
        
        SSb = np.trace(H1 @ G)      # Between-group
        SSt = np.trace(G)           # Total
        SSw = SSt - SSb             # Within-group
        
        # Degrees of freedom and statistics
        dfb, dfw = 1, n - 2
        msb = SSb / max(dfb, 1)
        msw = SSw / max(dfw, 1)
        
        F_statistic = msb / msw if msw > 0 else np.nan
        R_squared = SSb / SSt if SSt > 0 else np.nan
        
        # Additional information
        info = {
            'n_samples': n,
            'SS_between': SSb, 'SS_within': SSw, 'SS_total': SSt,
            'df_between': dfb, 'df_within': dfw,
            'MS_between': msb, 'MS_within': msw,
            'metric': metric, 'transform': transform,
            'aggregated': animal_ids is not None
        }
        
        # Print results
        print(f"PERMANOVA Descriptive Statistics:")
        print(f"  F-statistic = {F_statistic:.4f}")
        print(f"  R² = {R_squared:.4f} ({R_squared*100:.2f}% of variance explained)")
        print(f"  Samples: n = {n}")
        if animal_ids is not None:
            print(f"  (Aggregated by animal using '{aggregate_method}' method)")
        print(f"  Distance metric: {metric}")
        print(f"  Data transform: {transform}")
        
        return F_statistic, R_squared, info

    def permanova_test(self, metric="euclidean", transform="rank", robust_scale=True, 
                      winsor=0.01, n_permutations=999, alpha=0.05, random_state=None,
                      animal_ids=None, aggregate_method='mean'):
        """
        Full PERMANOVA test with permutations for p-value calculation.
        Returns F-statistic, p-value, R², and additional information.
        """
        rng = np.random.default_rng(random_state)
        
        # Get descriptive statistics first
        F_obs, R2, info = self.permanova_descriptive(
            metric=metric, transform=transform, robust_scale=robust_scale, 
            winsor=winsor, animal_ids=animal_ids, aggregate_method=aggregate_method
        )
        
        # Check permutation feasibility
        y = self.y.copy()
        if animal_ids is not None:
            _, y, _ = self._aggregate_by_animal(
                self._robust_transform(self.X, method=transform, 
                                     robust_scale=robust_scale, winsor=winsor),
                y, animal_ids, method=aggregate_method
            )
            L, pmin = self._perm_count_binary(y)
            feasible = (L >= 20)
            if not feasible:
                print(f"\n[WARNING] Permutation test not feasible:")
                print(f"  Only {L} possible animal-level labelings (need ≥20)")
                print(f"  Minimum achievable p-value = {pmin:.4f}")
                print(f"  Returning descriptive statistics only.")
                return F_obs, np.nan, R2, info
        
        # Run permutation test
        print(f"\nRunning {n_permutations} permutations...")
        
        # Prepare data for permutation
        Xr = self._robust_transform(self.X, method=transform, 
                                  robust_scale=robust_scale, winsor=winsor)
        if animal_ids is not None:
            Xr, y, _ = self._aggregate_by_animal(Xr, y, animal_ids, method=aggregate_method)
        
        D = squareform(pdist(Xr, metric=metric))
        n = D.shape[0]
        A = -0.5 * (D**2)
        H = np.eye(n) - np.ones((n, n))/n
        G = H @ A @ H
        msw = info['MS_within']
        
        # Permutation loop
        ge = 0
        for i in range(n_permutations):
            if (i + 1) % 100 == 0:
                print(f"  Permutation {i + 1}/{n_permutations}")
            
            yp = rng.permutation(y)
            Xdp = np.column_stack([np.ones(n), yp.astype(int)])
            H1p = Xdp @ np.linalg.pinv(Xdp.T @ Xdp) @ Xdp.T
            SSb_p = np.trace(H1p @ G)
            msb_p = SSb_p / max(1, 1)
            Fp = msb_p / msw if msw > 0 else np.nan
            
            if not np.isnan(Fp) and Fp >= F_obs:
                ge += 1
        
        # Calculate p-value and print results
        p_value = (ge + 1) / (n_permutations + 1)
        
        print(f"\nPERMANOVA Test Results:")
        print(f"  F-statistic = {F_obs:.4f}")
        print(f"  p-value = {p_value:.6f}")
        print(f"  R² = {R2:.4f}")
        
        if p_value < alpha:
            print(f"  → Significant difference (p < {alpha})")
            print("  → Reject H₀: group centroids are equal")
        else:
            print(f"  → No significant difference (p ≥ {alpha})")
            print("  → Fail to reject H₀: group centroids are equal")
        
        # Add permutation info
        info.update({
            'n_permutations': n_permutations,
            'p_value': p_value,
            'alpha': alpha
        })
        
        return F_obs, p_value, R2, info

    def _rbf_mmd2_unbiased(self, X, Y, sigma=None):
        """Calculate unbiased MMD² with RBF kernel."""
        Z = np.vstack([X, Y])
        D = squareform(pdist(Z, metric='euclidean'))
        
        if sigma is None:
            # Median heuristic on pooled distances
            dists = D[np.triu_indices_from(D, k=1)]
            sigma = np.median(dists)
            if sigma <= 0:
                sigma = 1.0
        
        K = np.exp(-(D**2) / (2*sigma**2))
        m, n = len(X), len(Y)
        
        Kxx = K[:m, :m]
        Kyy = K[m:, m:]
        Kxy = K[:m, m:]
        
        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)
        
        mmd2 = (Kxx.sum()/(m*(m-1))) + (Kyy.sum()/(n*(n-1))) - (2*Kxy.sum()/(m*n))
        return mmd2, sigma

    def mmd_test(self, alpha=0.05, n_permutations=0, random_state=None,
                 animal_ids=None, aggregate_method='mean'):
        """Maximum Mean Discrepancy test for two-sample comparison."""
        rng = np.random.default_rng(random_state)
        X1, X2 = self.X1, self.X2
        y = self.y.copy()

        if animal_ids is not None:
            X_all = np.vstack([self.X])
            X_agg, y_agg, _ = self._aggregate_by_animal(X_all, y, animal_ids, method=aggregate_method)
            X1 = X_agg[y_agg==0]
            X2 = X_agg[y_agg==1]

        stat, sigma = self._rbf_mmd2_unbiased(X1, X2, sigma=None)

        # Scaled MMD for interpretability
        Z = np.vstack([X1, X2])
        md = np.median(pdist(Z, metric='euclidean'))
        scaled = stat / md if md > 0 else stat

        # Permutation test
        p = np.nan
        if n_permutations and n_permutations > 0:
            lab = np.array([0]*len(X1) + [1]*len(X2))
            feasible = True
            
            if animal_ids is not None:
                L, pmin = self._perm_count_binary(lab)
                feasible = (L >= 20)
                if not feasible:
                    print(f"[INFO] MMD permutations skipped (only {L} animal-labelings; p_min={pmin:.3f}).")
            
            if feasible:
                ge = 0
                for _ in range(n_permutations):
                    lp = rng.permutation(lab)
                    stat_p, _ = self._rbf_mmd2_unbiased(Z[lp==0], Z[lp==1], sigma=sigma)
                    if stat_p >= stat:
                        ge += 1
                p = (ge + 1) / (n_permutations + 1)

        print(f"MMD (RBF, σ={sigma:.4f}): MMD²={stat:.4f}, scaled={scaled:.4f}, p={p if not np.isnan(p) else 'NA'}")
        return {"mmd2": stat, "scaled_mmd": scaled, "sigma": sigma, "p_value": p}

    def classification_permutation_test(self, classifier=None, cv=5, n_permutations=0, 
                                       alpha=0.05, random_state=None, animal_ids=None,
                                       aggregate_method='mean'):
        """
        Random Forest classification with animal-aware aggregation and permutation testing.
        """
        rng = np.random.default_rng(random_state)
        
        if classifier is None:
            classifier = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                              random_state=random_state)

        X_use, y_use = self.X, self.y
        
        # Animal-level aggregation if provided
        if animal_ids is not None:
            X_use, y_use, animal_order = self._aggregate_by_animal(
                self.X, self.y, animal_ids, method=aggregate_method)
            cv_split = StratifiedKFold(n_splits=min(cv, len(np.unique(animal_order))), 
                                     shuffle=True, random_state=random_state)
        else:
            cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

        # Choose appropriate scoring
        n_classes = len(np.unique(y_use))
        auc_scoring = "roc_auc" if n_classes == 2 else "roc_auc_ovr_weighted"

        # Observed metrics
        obs_acc = cross_val_score(classifier, X_use, y_use, cv=cv_split, scoring="accuracy").mean()
        obs_bacc = cross_val_score(classifier, X_use, y_use, cv=cv_split, scoring="balanced_accuracy").mean()
        obs_auc = cross_val_score(classifier, X_use, y_use, cv=cv_split, scoring=auc_scoring).mean()
        y_pred = cross_val_predict(classifier, X_use, y_use, cv=cv_split)
        obs_mcc = matthews_corrcoef(y_use, y_pred)
        cm_obs = confusion_matrix(y_use, y_pred)

        # Permutation test
        p_acc = p_bacc = p_auc = p_mcc = np.nan
        if n_permutations and n_permutations > 0:
            feasible = True
            if animal_ids is not None:
                L, pmin = self._perm_count_binary(y_use)
                feasible = (L >= 20)
                if not feasible:
                    print(f"[INFO] RF permutations skipped (only {L} animal-labelings; p_min={pmin:.3f}).")
            
            if feasible:
                ge_acc = ge_bacc = ge_auc = ge_mcc = 0
                for _ in range(n_permutations):
                    yp = rng.permutation(y_use)
                    acc = cross_val_score(classifier, X_use, yp, cv=cv_split, scoring="accuracy").mean()
                    bacc = cross_val_score(classifier, X_use, yp, cv=cv_split, scoring="balanced_accuracy").mean()
                    auc = cross_val_score(classifier, X_use, yp, cv=cv_split, scoring=auc_scoring).mean()
                    ypp = cross_val_predict(classifier, X_use, yp, cv=cv_split)
                    mcc = matthews_corrcoef(yp, ypp)
                    
                    ge_acc += (acc >= obs_acc)
                    ge_bacc += (bacc >= obs_bacc)
                    ge_auc += (auc >= obs_auc)
                    ge_mcc += (mcc >= obs_mcc)
                
                denom = n_permutations + 1
                p_acc = (ge_acc + 1) / denom
                p_bacc = (ge_bacc + 1) / denom
                p_auc = (ge_auc + 1) / denom
                p_mcc = (ge_mcc + 1) / denom

        # Feature importances
        classifier.fit(X_use, y_use)
        top = []
        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
            top = sorted(zip(self.features, importances), 
                        key=lambda t: t[1], reverse=True)[:10]

        # Print results
        print("RF (descriptive metrics):")
        print(f"  Acc={obs_acc:.3f} (p={p_acc if not np.isnan(p_acc) else 'NA'}) | "
              f"BalAcc={obs_bacc:.3f} (p={p_bacc if not np.isnan(p_bacc) else 'NA'}) | "
              f"AUC={obs_auc:.3f} (p={p_auc if not np.isnan(p_auc) else 'NA'}) | "
              f"MCC={obs_mcc:.3f} (p={p_mcc if not np.isnan(p_mcc) else 'NA'})")
        print(f"  Confusion matrix:\n{cm_obs}")
        if top:
            print("  Top features:")
            for i, (f, v) in enumerate(top, 1):
                print(f"    {i:2d}. {f}: {v:.4f}")

        return {
            "accuracy": obs_acc, "p_acc": p_acc,
            "balanced_accuracy": obs_bacc, "p_balanced_accuracy": p_bacc,
            "auc": obs_auc, "p_auc": p_auc,
            "mcc": obs_mcc, "p_mcc": p_mcc,
            "confusion_matrix": cm_obs,
            "top_features": top
        }

    def run_all(self, alpha=0.05, n_permutations=0, random_state=None,
                animal_ids=None, aggregate_method='mean'):
        """
        Run all multivariate tests. Set n_permutations>0 only when you have 
        enough animals (>=3 per group ideally).
        """
        results = {}
        print("=" * 50)
        print("MULTIVARIATE COMPARISON")
        print("=" * 50)

        # PERMANOVA descriptive
        print("\n--- PERMANOVA (distance-based) ---")
        F, R2, info = self.permanova_descriptive(
            metric="euclidean", 
            animal_ids=animal_ids,
            aggregate_method=aggregate_method
        )
        results["PERMANOVA"] = {"F": F, "R2": R2, "info": info}

        # MMD test
        print("\n--- MMD TEST ---")
        mmd_results = self.mmd_test(
            alpha=alpha, n_permutations=n_permutations,
            random_state=random_state, animal_ids=animal_ids,
            aggregate_method=aggregate_method
        )
        results["MMD"] = mmd_results

        # Random Forest classification
        print("\n--- RF CLASSIFICATION ---")
        rf_results = self.classification_permutation_test(
            cv=5, n_permutations=n_permutations, alpha=alpha,
            random_state=random_state, animal_ids=animal_ids,
            aggregate_method=aggregate_method
        )
        results["RF"] = rf_results
        
        return results
