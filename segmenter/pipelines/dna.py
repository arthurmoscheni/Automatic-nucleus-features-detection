from __future__ import annotations

import os
from typing import Dict, List
import pandas as pd

from analyzers.dna import DNADistributionAnalyzer
from viz.dna import visualize_clusters_overlay


def run_dna_analysis(
    dna_data_list: List[Dict],
    output_dir: str,
    visualization: bool = False,
    overlay_method: str = "original",
) -> pd.DataFrame:
    """
    Orchestrate per-image analysis. Returns a combined DataFrame.
    (No CSV saving here; use io.save.save_dna_results separately.)
    """
    results: List[pd.DataFrame] = []

    for data in dna_data_list:
        original_image = data['original_image']
        preprocessed_image = data['preprocessed_image']
        mask = data['mask']
        image_id = data['image_id']

        analyzer = DNADistributionAnalyzer(original_image, preprocessed_image, mask)
        cell_results = analyzer.analyze_all_cells()

        # optional overlay(s)
        out_dir_img = os.path.join(output_dir, image_id)
        os.makedirs(out_dir_img, exist_ok=True)
        if visualization:
            save_path = os.path.join(out_dir_img, "dna_analysis_summary.png")
            visualize_clusters_overlay(analyzer, threshold_method=overlay_method, show_image=False, save_path=save_path)

        if cell_results is not None and not cell_results.empty:
            cell_results['image_id'] = image_id
            results.append(cell_results)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()
