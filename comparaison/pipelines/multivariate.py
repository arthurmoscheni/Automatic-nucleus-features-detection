from __future__ import annotations
from typing import Optional, List, Dict, Any
import pandas as pd

from analyzers.multivariate import GlobalMultivariateTests

def run_multivariate(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    group1: Optional[str] = None,
    group2: Optional[str] = None,
    features: Optional[List[str]] = None,
    alpha: float = 0.05,
    n_permutations: int = 0,
    random_state: Optional[int] = None,
    animal_ids: Optional[list] = None,
    aggregate_method: str = "mean",
    standardize: bool = True,
    drop_constant: bool = True,
) -> Dict[str, Any]:
    """
    Thin orchestration wrapper so mains stay tiny.
    Mirrors GlobalMultivariateTests(...).run_all(...)
    """
    tester = GlobalMultivariateTests(
        df1, df2, group1=group1, group2=group2, features=features,
        standardize=standardize, drop_constant=drop_constant
    )
    return tester.run_all(
        alpha=alpha,
        n_permutations=n_permutations,
        random_state=random_state,
        animal_ids=animal_ids,
        aggregate_method=aggregate_method,
    )
