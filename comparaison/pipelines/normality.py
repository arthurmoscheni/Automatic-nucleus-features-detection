from __future__ import annotations

from typing import Optional, Dict
import pandas as pd


from analyzers.normality import NormalityTester


def run_normality_tests(df1: pd.DataFrame, df2: pd.DataFrame, g1: str, g2: str, plot: bool):

    normality = NormalityTester(df1, df2, group1=g1, group2=g2, plot=plot)
    
    return normality.run_all_tests()