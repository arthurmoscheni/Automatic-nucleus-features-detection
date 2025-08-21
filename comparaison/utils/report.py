from __future__ import annotations

from typing import Dict, List
import pandas as pd


def print_summary_results(
    feature_names: List[str],
    group1: str,
    group2: str,
    normality1: Dict,
    normality2: Dict,
    variance_results: Dict,
    mardia1: Dict,
    mardia2: Dict,
    box_m: Dict,
) -> None:
    """
    Print a comprehensive summary (same text/logic as your class).
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE NORMALITY AND COVARIANCE ANALYSIS SUMMARY")
    print("=" * 80)

    # --- Univariate normality
    print("\n--- UNIVARIATE NORMALITY TEST SUMMARY ---")
    for feature in feature_names:
        print(f"\n{feature.upper()}:")

        def fmt(results, gname):
            if "error" in results:
                return f"  {gname}: No valid tests"
            tests = []
            if "shapiro_wilk" in results and "error" not in results["shapiro_wilk"]:
                is_normal = results["shapiro_wilk"]["normal"]
                p = results["shapiro_wilk"]["p_value"]
                tests.append(f"Shapiro-Wilk: {'Normal' if is_normal else 'NOT Normal'} (p={p:.2e})")
            if "anderson_darling" in results and "error" not in results["anderson_darling"]:
                is_normal = results["anderson_darling"]["normal"]
                tests.append(f"Anderson-Darling: {'Normal' if is_normal else 'NOT Normal'}")
            return f"  {gname}: {', '.join(tests) if tests else 'No valid tests'}"

        print(fmt(normality1[feature], group1))
        print(fmt(normality2[feature], group2))

        if feature in variance_results and "error" not in variance_results[feature]:
            var_tests = []
            if "levene" in variance_results[feature] and "error" not in variance_results[feature]["levene"]:
                eq = variance_results[feature]["levene"]["equal_variances"]
                p = variance_results[feature]["levene"]["p_value"]
                var_tests.append(f"Levene: {'Equal' if eq else 'UNEQUAL'} variances (p={p:.2e})")
            print(f"  Variance: {', '.join(var_tests) if var_tests else 'No valid tests'}")

    # --- Multivariate normality
    print("\n--- MULTIVARIATE NORMALITY (MARDIA TEST) ---")
    _print_mardia_result(mardia1, group1)
    _print_mardia_result(mardia2, group2)

    # --- Covariance equality
    print("\n--- COVARIANCE MATRIX EQUALITY (BOX'S M TEST) ---")
    if "error" not in box_m:
        print(f"Result: {'Equal covariances' if box_m['equal_covariances'] else 'UNEQUAL covariances'}")
        print(f"P-value: {box_m['p_value']:.2e}")
        print(f"Chi-square statistic: {box_m['chi2_statistic']:.2e}")
    else:
        print(f"Error: {box_m['error']}")

    # --- Recommendations
    _print_recommendations(feature_names, normality1, normality2, variance_results, mardia1, mardia2, box_m)


def _print_mardia_result(mardia_result: Dict, group_name: str) -> None:
    if "error" not in mardia_result and "warning" not in mardia_result:
        ok = mardia_result["multivariate_normal"]
        print(f"{group_name}: {'Multivariate Normal' if ok else 'NOT Multivariate Normal'}")
        print(f"  Combined p-value: {mardia_result['combined_p_value']:.3e}")
    else:
        print(f"{group_name}: {mardia_result.get('error', mardia_result.get('warning', 'Unknown issue'))}")


def _print_recommendations(
    feature_names: list[str],
    normality1: Dict,
    normality2: Dict,
    variance_results: Dict,
    mardia1: Dict,
    mardia2: Dict,
    box_m: Dict,
) -> None:
    print("\n--- ANALYSIS RECOMMENDATIONS ---")

    non_normal_count = _count_non_normal_features(feature_names, normality1, normality2)
    unequal_var_count = _count_unequal_variance_features(feature_names, variance_results)
    multivar_non_normal = _check_multivariate_non_normal(mardia1, mardia2)
    unequal_covariances = ("error" not in box_m) and (not box_m["equal_covariances"])

    print(f"Non-normal features: {non_normal_count}/{len(feature_names)}")
    print(f"Unequal variance features: {unequal_var_count}/{len(feature_names)}")
    print(f"Multivariate non-normality: {multivar_non_normal}")
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


def _count_non_normal_features(feature_names: list[str], normality1: Dict, normality2: Dict) -> int:
    cnt = 0
    for feature in feature_names:
        for results in (normality1, normality2):
            if (
                feature in results
                and "shapiro_wilk" in results[feature]
                and "error" not in results[feature]["shapiro_wilk"]
                and not results[feature]["shapiro_wilk"]["normal"]
            ):
                cnt += 1
                break
    return cnt


def _count_unequal_variance_features(feature_names: list[str], variance_results: Dict) -> int:
    cnt = 0
    for feature in feature_names:
        if (
            feature in variance_results
            and "levene" in variance_results[feature]
            and "error" not in variance_results[feature]["levene"]
            and not variance_results[feature]["levene"]["equal_variances"]
        ):
            cnt += 1
    return cnt


def _check_multivariate_non_normal(mardia1: Dict, mardia2: Dict) -> bool:
    return (
        ("error" not in mardia1 and "warning" not in mardia1 and not mardia1["multivariate_normal"])
        or ("error" not in mardia2 and "warning" not in mardia2 and not mardia2["multivariate_normal"])
    )
