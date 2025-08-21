from __future__ import annotations
import os
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from analyzers.morphology import ImageAnalyzer
from viz.morphology import (
    annotate_cells_with_features,
    visualize_arcs_on_image,
)
from io_utils.save import save_morpho_combined_results


def process_single_image(
    image_data: Dict,
    pixel_size_um: float = 0.124,
    min_circularity: float = 0.4,
    min_area_um2: float = 20,
    min_arc_length: int = 40,
    visualization: bool = False,
    output_dir: Optional[str] = None
) -> Dict:
    """Orchestrates the same steps as your class method `process_single_image`."""
    analyzer = ImageAnalyzer(pixel_size_um=pixel_size_um)

    original_image = image_data['original_image']
    preprocessed_image = image_data['preprocessed_image']
    label_mask = image_data['mask']
    image_id = image_data['image_id']

    print(f"Processing image {image_id}...")

    final_mask = analyzer.get_final_mask(preprocessed_image, label_mask, min_circularity=min_circularity, min_area_um2=min_area_um2)
    overlaid, feats = analyzer.get_image_with_contours_and_features(preprocessed_image, final_mask)
    feats = analyzer.add_morphology_features(feats, final_mask)
    feats['image_id'] = image_id

    intens = analyzer.calculate_cell_intensity(original_image, final_mask, image_id)
    aci = analyzer.analyze_arcs(final_mask, min_arc_length, image_id)

    out_dir = None
    if output_dir:
        out_dir = os.path.join(output_dir, image_id)
        os.makedirs(out_dir, exist_ok=True)

    # arcs viz
    _ = visualize_arcs_on_image(
        analyzer, preprocessed_image, final_mask,
        min_arc_length=min_arc_length, linewidth=2,
        show_image=visualization,
        save_path=(os.path.join(out_dir, "arcs.png") if out_dir else None)
    )

    # annotated features
    annotate_cells_with_features(
        original_image, final_mask, feats,
        show_image=visualization,
        save_path=(os.path.join(out_dir, "with_annotation.png") if out_dir else None)
    )

    # wrinkle overlay
    wrinkles_df, wrinkle_rgb = analyzer.analyze_image_wrinkle_2(
        original_image,
        final_mask,
        image_id=image_id,
        boundary_width=5,
        min_inv_size=25,
        min_ridge_length=25,
        sigma=1.0
    )
    if out_dir:
        import cv2
        cv2.imwrite(os.path.join(out_dir, "wrinkles.png"), cv2.cvtColor(wrinkle_rgb, cv2.COLOR_RGB2BGR))

    if not wrinkles_df.empty:
        print(f"\nWrinkles analysis for {image_id}:")
        for _, row in wrinkles_df.iterrows():
            print(
                f"  Cell {row['label']}: Ratio = {row['ratio']:.3f} "
                f"(Boundary pixels: {row['boundary_pixels']}, "
                f"Invagination pixels: {row['invag_pixels']})"
            )
    else:
        print(f"No wrinkles data available for {image_id}")

    return {
        'image_id': image_id,
        'original_image': original_image,
        'preprocessed_image': preprocessed_image,
        'final_mask': final_mask,
        'features_df': feats,
        'overlaid_image': overlaid,
        'aci_results': aci,
        'signal_intensities': intens,
        'wrinkles_results': wrinkles_df
    }


def process_multiple_images(
    image_data_list: List[Dict],
    pixel_size_um: float = 0.124,
    min_circularity: float = 0.4,
    min_area_um2: float = 20,
    min_arc_length: int = 40,
    visualization: bool = False,
    output_dir: Optional[str] = None
) -> Dict:
    """Same aggregator as your class method, just a free function."""
    print(f"Found {len(image_data_list)} images with valid red masks for analysis.")
    all_results = []
    feats_all = []
    aci_all = []
    wrinkles_all = []
    intens_all = []

    for item in image_data_list:
        res = process_single_image(
            item,
            pixel_size_um=pixel_size_um,
            min_circularity=min_circularity,
            min_area_um2=min_area_um2,
            min_arc_length=min_arc_length,
            visualization=visualization,
            output_dir=output_dir
        )
        all_results.append(res)
        feats_all.append(res['features_df'])
        if not res['aci_results'].empty: aci_all.append(res['aci_results'])
        if not res['signal_intensities'].empty: intens_all.append(res['signal_intensities'])
        if not res['wrinkles_results'].empty: wrinkles_all.append(res['wrinkles_results'])

    combined_features_df = pd.concat(feats_all, ignore_index=True) if feats_all else pd.DataFrame()
    combined_aci_df = pd.concat(aci_all, ignore_index=True) if aci_all else pd.DataFrame()
    combined_signal_intensities_df = pd.concat(intens_all, ignore_index=True) if intens_all else pd.DataFrame()
    combined_wrinkles_df = pd.concat(wrinkles_all, ignore_index=True) if wrinkles_all else pd.DataFrame()

    return {
        'individual_results': all_results,
        'combined_features_df': combined_features_df,
        'combined_aci_df': combined_aci_df,
        'combined_wrinkles_df': combined_wrinkles_df,
        'combined_signal_intensities_df': combined_signal_intensities_df
    }


def run_morphological_analysis(
    image_data_list: List[Dict],
    output_dir: str,
    visualization: bool = False,
    pixel_size_um: Optional[float] = None
):
    """Drop-in replacement for your function name."""
    results = process_multiple_images(
        image_data_list=image_data_list,
        pixel_size_um=(pixel_size_um if pixel_size_um is not None else 0.124),
        min_circularity=0.4,
        min_area_um2=20,
        min_arc_length=40,
        visualization=visualization,
        output_dir=output_dir
    )
    # write combined CSVs
    _ = save_morpho_combined_results(results, output_dir)
    return results
