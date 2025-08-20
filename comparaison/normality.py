import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import chi2, shapiro, anderson, kstest, normaltest, levene, bartlett
import warnings
import matplotlib.pyplot as plt

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


class NormalityTester:
    def __init__(self, df1, df2, group1=None, group2=None):
        """
        Initialize with two dataframes for comparison
        
        Args:
            df1 (pd.DataFrame): First dataframe
            df2 (pd.DataFrame): Second dataframe
            group1 (str): Name for first group (optional)
            group2 (str): Name for second group (optional)
        """
        self.df1 = df1
        self.df2 = df2
        self.group1 = group1 or "Group 1"
        self.group2 = group2 or "Group 2"
        self.feature_names = self.df1.columns.tolist()

    def univariate_normality_tests(self, data, group_name=""):
        """
        Perform comprehensive univariate normality tests
        
        Args:
            data (pd.DataFrame): Data to test
            group_name (str): Name of the group for reporting
            
        Returns:
            dict: Results of all normality tests
        """
        results = {}
        
        for feature in data.columns:
            feature_data = data[feature].dropna()
            
            if len(feature_data) < 3:
                results[feature] = {"error": "Insufficient data points"}
                continue
            
            feature_results = {}
            
            # Shapiro-Wilk test (best for smaller samples)
            if len(feature_data) <= 5000:
                try:
                    stat, p_val = shapiro(feature_data)
                    feature_results['shapiro_wilk'] = {
                        'statistic': stat, 
                        'p_value': p_val,
                        'normal': p_val > 0.05
                    }
                except Exception:
                    feature_results['shapiro_wilk'] = {"error": "Test failed"}
            
            # Anderson-Darling test
            try:
                result = anderson(feature_data, dist='norm')
                feature_results['anderson_darling'] = {
                    'statistic': result.statistic,
                    'critical_values': result.critical_values,
                    'significance_levels': result.significance_levels,
                    'normal': result.statistic <= result.critical_values[2]  # 5% level
                }
            except Exception:
                feature_results['anderson_darling'] = {"error": "Test failed"}
            
            # Kolmogorov-Smirnov test
            try:
                stat, p_val = kstest(feature_data, 'norm', 
                                   args=(np.mean(feature_data), np.std(feature_data)))
                feature_results['kolmogorov_smirnov'] = {
                    'statistic': stat, 
                    'p_value': p_val,
                    'normal': p_val > 0.05
                }
            except Exception:
                feature_results['kolmogorov_smirnov'] = {"error": "Test failed"}
            
            # D'Agostino-Pearson test
            if len(feature_data) >= 8:  # Minimum sample size for this test
                try:
                    stat, p_val = normaltest(feature_data)
                    feature_results['dagostino_pearson'] = {
                        'statistic': stat, 
                        'p_value': p_val,
                        'normal': p_val > 0.05
                    }
                except Exception:
                    feature_results['dagostino_pearson'] = {"error": "Test failed"}
            
            # Descriptive statistics
            feature_results['descriptive'] = {
                'mean': np.mean(feature_data),
                'median': np.median(feature_data),
                'std': np.std(feature_data),
                'skewness': stats.skew(feature_data),
                'kurtosis': stats.kurtosis(feature_data),
                'n_samples': len(feature_data)
            }
            
            results[feature] = feature_results
        
        return results

    def variance_equality_tests(self):
        """
        Test equality of variances between the two groups
        
        Returns:
            dict: Results of variance equality tests
        """
        results = {}
        
        for feature in self.feature_names:
            data1 = self.df1[feature].dropna()
            data2 = self.df2[feature].dropna()
            
            if len(data1) < 2 or len(data2) < 2:
                results[feature] = {"error": "Insufficient data for variance tests"}
                continue
            
            feature_results = {}
            
            # Levene's test (robust to non-normality)
            try:
                stat, p_val = levene(data1, data2)
                feature_results['levene'] = {
                    'statistic': stat,
                    'p_value': p_val,
                    'equal_variances': p_val > 0.05
                }
            except Exception:
                feature_results['levene'] = {"error": "Test failed"}
            
            # Bartlett's test (assumes normality)
            try:
                stat, p_val = bartlett(data1, data2)
                feature_results['bartlett'] = {
                    'statistic': stat,
                    'p_value': p_val,
                    'equal_variances': p_val > 0.05
                }
            except Exception:
                feature_results['bartlett'] = {"error": "Test failed"}
            
            # F-test for equality of variances
            try:
                var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
                f_stat = var1 / var2 if var1 > var2 else var2 / var1
                df1, df2 = len(data1) - 1, len(data2) - 1
                p_val = 2 * (1 - stats.f.cdf(f_stat, df1, df2))
                
                feature_results['f_test'] = {
                    'statistic': f_stat,
                    'p_value': p_val,
                    'equal_variances': p_val > 0.05,
                    'variance_ratio': var1 / var2
                }
            except Exception:
                feature_results['f_test'] = {"error": "Test failed"}
            
            results[feature] = feature_results
        
        return results

    def mardia_test(self, data, group_name=""):
        """
        Mardia's test for multivariate normality
        
        Args:
            data (pd.DataFrame): Data to test
            group_name (str): Name of the group
            
        Returns:
            dict: Test statistics and p-values
        """
        if data.empty:
            return {"error": "Empty dataset"}
            
        n, p = data.shape
        
        if n < p + 1:
            return {"error": f"Insufficient samples (n={n}) for {p} variables. Need at least {p+1} samples."}
        
        if n < 20:
            return {"warning": f"Small sample size (n={n}). Results may be unreliable."}
        
        try:
            # Center the data
            data_centered = data - data.mean()
            
            # Calculate covariance matrix with regularization if needed
            S = np.cov(data_centered.T)
            
            # Check for singularity and add small regularization if needed
            eigenvals = np.linalg.eigvals(S)
            if np.min(eigenvals) < 1e-10:
                S += np.eye(p) * 1e-8  # Small regularization
            
            S_inv = np.linalg.inv(S)
            
            # Calculate Mahalanobis distances
            mahal_dist = []
            for i in range(n):
                d = data_centered.iloc[i].values
                mahal_dist.append(d.T @ S_inv @ d)
            
            mahal_dist = np.array(mahal_dist)
            
            # Multivariate skewness test
            b1p = np.sum(mahal_dist**3) / n
            skew_stat = n * b1p / 6
            skew_df = p * (p + 1) * (p + 2) / 6
            skew_p_value = 1 - chi2.cdf(skew_stat, skew_df)
            
            # Multivariate kurtosis test
            b2p = np.sum(mahal_dist**2) / n
            expected_b2p = p * (p + 2)
            kurt_stat = (b2p - expected_b2p) / np.sqrt(8 * p * (p + 2) / n)
            kurt_p_value = 2 * (1 - stats.norm.cdf(abs(kurt_stat)))
            
            # Combined test
            combined_stat = skew_stat + kurt_stat**2
            combined_df = skew_df + 1
            combined_p_value = 1 - chi2.cdf(combined_stat, combined_df)
            
            return {
                "n_samples": n,
                "n_variables": p,
                "skewness_stat": skew_stat,
                "skewness_df": skew_df,
                "skewness_p_value": skew_p_value,
                "skewness_normal": skew_p_value > 0.05,
                "kurtosis_stat": kurt_stat,
                "kurtosis_p_value": kurt_p_value,
                "kurtosis_normal": kurt_p_value > 0.05,
                "combined_stat": combined_stat,
                "combined_df": combined_df,
                "combined_p_value": combined_p_value,
                "multivariate_normal": combined_p_value > 0.05,
                "condition_number": np.linalg.cond(S)
            }
            
        except np.linalg.LinAlgError as e:
            return {"error": f"Linear algebra error: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def box_m_test(self):
        """
        Box's M test for equality of covariance matrices
        
        Returns:
            dict: Test statistic and p-value
        """
        if self.df1.empty or self.df2.empty:
            return {"error": "One or both datasets are empty"}
            
        n1, p = self.df1.shape
        n2, _ = self.df2.shape
        
        if n1 <= p or n2 <= p:
            return {"error": f"Insufficient samples for Box's M test. Need > {p} samples per group, have {n1} and {n2}"}
        
        try:
            # Calculate covariance matrices
            S1 = np.cov(self.df1.T)
            S2 = np.cov(self.df2.T)
            
            # Check for numerical issues
            cond1 = np.linalg.cond(S1)
            cond2 = np.linalg.cond(S2)
            
            if cond1 > 1e12 or cond2 > 1e12:
                return {"error": "Covariance matrices are nearly singular (high condition number)"}
            
            # Pooled covariance matrix
            S_pooled = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
            
            # Calculate determinants with numerical stability check
            det_S1 = np.linalg.det(S1)
            det_S2 = np.linalg.det(S2)
            det_Sp = np.linalg.det(S_pooled)
            
            if det_S1 <= 0 or det_S2 <= 0 or det_Sp <= 0:
                return {"error": "Non-positive determinant encountered"}
            
            # Box's M statistic
            M = (n1 - 1) * np.log(det_S1) + (n2 - 1) * np.log(det_S2) - (n1 + n2 - 2) * np.log(det_Sp)
            
            # Correction factor for small samples
            c = (2 * p**2 + 3 * p - 1) / (6 * (p + 1)) * (1/(n1-1) + 1/(n2-1) - 1/(n1+n2-2))
            
            # Chi-square approximation
            chi2_stat = -M * (1 - c)  # Note: negative sign for proper test direction
            df = p * (p + 1) / 2
            p_value = 1 - chi2.cdf(chi2_stat, df)
            
            return {
                "M_statistic": M,
                "chi2_statistic": chi2_stat,
                "degrees_of_freedom": df,
                "p_value": p_value,
                "equal_covariances": p_value > 0.05,
                "condition_numbers": {"group1": cond1, "group2": cond2},
                "sample_sizes": {"group1": n1, "group2": n2}
            }
            
        except np.linalg.LinAlgError as e:
            return {"error": f"Linear algebra error: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def create_comprehensive_plots(self):
        """Create comprehensive diagnostic plots"""
        n_features = len(self.feature_names)
        
        # Create Q-Q plots
        fig, axes = plt.subplots(2, min(4, n_features), figsize=(16, 8))
        if n_features == 1:
            axes = axes.reshape(2, 1)
        
        for i, feature in enumerate(self.feature_names[:min(4, n_features)]):
            # Group 1 Q-Q plot
            stats.probplot(self.df1[feature].dropna(), dist="norm", plot=axes[0, i])
            axes[0, i].set_title(f'{self.group1} - {feature}')
            axes[0, i].grid(True, alpha=0.3)
            
            # Group 2 Q-Q plot
            stats.probplot(self.df2[feature].dropna(), dist="norm", plot=axes[1, i])
            axes[1, i].set_title(f'{self.group2} - {feature}')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Create distribution comparison plots
        fig, axes = plt.subplots((n_features + 2) // 3, 3, figsize=(15, 5 * ((n_features + 2) // 3)))
        if n_features <= 3:
            axes = axes.reshape(1, -1) if n_features > 1 else [axes]
        
        for i, feature in enumerate(self.feature_names):
            row, col = i // 3, i % 3
            ax = axes[row, col] if n_features > 3 else axes[i] if n_features > 1 else axes
            
            # Plot histograms
            ax.hist(self.df1[feature].dropna(), alpha=0.6, bins=30, 
                   label=self.group1, density=True, color='skyblue')
            ax.hist(self.df2[feature].dropna(), alpha=0.6, bins=30, 
                   label=self.group2, density=True, color='lightcoral')
            ax.set_title(f'{feature} Distribution')
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        if n_features > 1:
            for i in range(n_features, len(axes.flat) if n_features > 3 else len(axes)):
                ax = axes.flat[i] if n_features > 3 else axes[i]
                ax.set_visible(False)
        
        plt.tight_layout()
        plt.show()

    def print_summary_results(self, normality1, normality2, variance_results, mardia1, mardia2, box_m):
        """Print a comprehensive summary of all test results"""
        print("\n" + "="*80)
        print("COMPREHENSIVE NORMALITY AND COVARIANCE ANALYSIS SUMMARY")
        print("="*80)
        
        # Univariate normality summary
        print("\n--- UNIVARIATE NORMALITY TEST SUMMARY ---")
        for feature in self.feature_names:
            print(f"\n{feature.upper()}:")
            
            # Helper function to format test results
            def format_test_results(results, group_name):
                if 'error' in results:
                    return f"  {group_name}: No valid tests"
                
                tests = []
                if 'shapiro_wilk' in results and 'error' not in results['shapiro_wilk']:
                    is_normal = results['shapiro_wilk']['normal']
                    p_val = results['shapiro_wilk']['p_value']
                    tests.append(f"Shapiro-Wilk: {'Normal' if is_normal else 'NOT Normal'} (p={p_val:.2e})")
                
                if 'anderson_darling' in results and 'error' not in results['anderson_darling']:
                    is_normal = results['anderson_darling']['normal']
                    tests.append(f"Anderson-Darling: {'Normal' if is_normal else 'NOT Normal'}")
                
                return f"  {group_name}: {', '.join(tests) if tests else 'No valid tests'}"
            
            # Print results for both groups
            print(format_test_results(normality1[feature], self.group1))
            print(format_test_results(normality2[feature], self.group2))

            # Variance equality
            if feature in variance_results and 'error' not in variance_results[feature]:
                var_tests = []
                if 'levene' in variance_results[feature] and 'error' not in variance_results[feature]['levene']:
                    equal_var = variance_results[feature]['levene']['equal_variances']
                    p_val = variance_results[feature]['levene']['p_value']
                    var_tests.append(f"Levene: {'Equal' if equal_var else 'UNEQUAL'} variances (p={p_val:.2e})")
                
                print(f"  Variance: {', '.join(var_tests) if var_tests else 'No valid tests'}")
        
        # Multivariate normality
        print("\n--- MULTIVARIATE NORMALITY (MARDIA TEST) ---")
        self._print_mardia_result(mardia1, self.group1)
        self._print_mardia_result(mardia2, self.group2)
        
        # Covariance equality
        print("\n--- COVARIANCE MATRIX EQUALITY (BOX'S M TEST) ---")
        if 'error' not in box_m:
            print(f"Result: {'Equal covariances' if box_m['equal_covariances'] else 'UNEQUAL covariances'}")
            print(f"P-value: {box_m['p_value']:.2e}")
            print(f"Chi-square statistic: {box_m['chi2_statistic']:.2e}")
        else:
            print(f"Error: {box_m['error']}")
        
        # Recommendations
        self._print_recommendations(normality1, normality2, variance_results, mardia1, mardia2, box_m)

    def _print_mardia_result(self, mardia_result, group_name):
        """Helper method to print Mardia test results"""
        if 'error' not in mardia_result and 'warning' not in mardia_result:
            print(f"{group_name}: {'Multivariate Normal' if mardia_result['multivariate_normal'] else 'NOT Multivariate Normal'}")
            print(f"  Combined p-value: {mardia_result['combined_p_value']:.3e}")
        else:
            print(f"{group_name}: {mardia_result.get('error', mardia_result.get('warning', 'Unknown issue'))}")

    def _print_recommendations(self, normality1, normality2, variance_results, mardia1, mardia2, box_m):
        """Helper method to print analysis recommendations"""
        print("\n--- ANALYSIS RECOMMENDATIONS ---")
        
        # Check for violations
        non_normal_count = self._count_non_normal_features(normality1, normality2)
        unequal_var_count = self._count_unequal_variance_features(variance_results)
        multivariate_non_normal = self._check_multivariate_non_normal(mardia1, mardia2)
        unequal_covariances = 'error' not in box_m and not box_m['equal_covariances']
        
        print(f"Non-normal features: {non_normal_count}/{len(self.feature_names)}")
        print(f"Unequal variance features: {unequal_var_count}/{len(self.feature_names)}")
        print(f"Multivariate non-normality: {multivariate_non_normal}")
        print(f"Unequal covariance matrices: {unequal_covariances}")
        
        print("\nRECOMMENDED STATISTICAL TESTS:")
        if non_normal_count > 0 or unequal_var_count > 0:
            print("✓ Mann-Whitney U test (for individual features)")
            print("✓ Permutation tests")
            print("✓ PERMANOVA (multivariate)")
        else:
            print("✓ t-tests may be appropriate")
            print("✓ MANOVA (if covariances are equal)")
        
        if unequal_covariances:
            print("✓ Avoid MANOVA (use PERMANOVA instead)")
            print("✓ Use Welch's t-test instead of Student's t-test")
        
        print("✓ Effect size calculations (Cliff's Delta)")
        print("✓ Multiple comparison corrections (FDR/Bonferroni)")

    def _count_non_normal_features(self, normality1, normality2):
        """Count features that violate normality assumptions"""
        non_normal_count = 0
        for feature in self.feature_names:
            for results in [normality1, normality2]:
                if (feature in results and 'shapiro_wilk' in results[feature] and 
                    'error' not in results[feature]['shapiro_wilk'] and 
                    not results[feature]['shapiro_wilk']['normal']):
                    non_normal_count += 1
                    break
        return non_normal_count

    def _count_unequal_variance_features(self, variance_results):
        """Count features with unequal variances"""
        unequal_var_count = 0
        for feature in self.feature_names:
            if (feature in variance_results and 'levene' in variance_results[feature] and 
                'error' not in variance_results[feature]['levene'] and 
                not variance_results[feature]['levene']['equal_variances']):
                unequal_var_count += 1
        return unequal_var_count

    def _check_multivariate_non_normal(self, mardia1, mardia2):
        """Check if either group violates multivariate normality"""
        return (('error' not in mardia1 and 'warning' not in mardia1 and not mardia1['multivariate_normal']) or 
                ('error' not in mardia2 and 'warning' not in mardia2 and not mardia2['multivariate_normal']))

    def run_all_tests(self):
        """
        Run comprehensive analysis with all tests and visualizations
        
        Returns:
            dict: All test results
        """
        print("Running comprehensive normality and covariance analysis...")
        print(f"Analyzing {len(self.feature_names)} features: {self.feature_names}")
        
        # Run all tests
        print("\n1. Running univariate normality tests...")
        normality1 = self.univariate_normality_tests(self.df1, self.group1)
        normality2 = self.univariate_normality_tests(self.df2, self.group2)

        print("2. Running variance equality tests...")
        variance_results = self.variance_equality_tests()
        
        print("3. Running multivariate normality tests...")
        mardia1 = self.mardia_test(self.df1, self.group1)
        mardia2 = self.mardia_test(self.df2, self.group2)

        print("4. Running Box's M test...")
        box_m = self.box_m_test()
        
        print("5. Creating diagnostic plots...")
        self.create_comprehensive_plots()
        
        # Print comprehensive summary
        self.print_summary_results(normality1, normality2, variance_results, mardia1, mardia2, box_m)
        
        return {
            "univariate_normality_group1": normality1,
            "univariate_normality_group2": normality2,
            "variance_equality": variance_results,
            "multivariate_normality_group1": mardia1,
            "multivariate_normality_group2": mardia2,
            "box_m_test": box_m,
            "feature_names": self.feature_names,
            "sample_sizes": {"group1": len(self.df1), "group2": len(self.df2)}
        }
