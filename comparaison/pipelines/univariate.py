from __future__ import annotations
from typing import Optional, List
import pandas as pd

from analyzers.univariate import UnivariateComparison

def run_univariate(df1: pd.DataFrame, df2: pd.DataFrame,
                   group1: Optional[str] = None, group2: Optional[str] = None,
                   features: Optional[List[str]] = None,
                   alpha: float = 0.05, visualization: bool = False) -> pd.DataFrame:
    """Thin orchestration wrapper so mains stay tiny."""
    uc = UnivariateComparison(df1, df2, group1=group1, group2=group2, features=features)
    return uc.run_all(alpha=alpha, visualization=visualization)
