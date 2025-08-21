from __future__ import annotations
from typing import Dict, List



def prepare_image_data_for_analysis(batch_results: Dict) -> List[Dict]:
    """
    Collect red-channel triplets for morphology analysis:
    original, preprocessed, and mask per image_id.
    """
    out: List[Dict] = []
    for image_id, data in batch_results.items():
        if 'red_preprocessed_image' in data and 'red_mask' in data:
            out.append({
                'original_image': data['red_original_image'],
                'preprocessed_image': data['red_preprocessed_image'],
                'mask': data['red_mask'],
                'image_id': image_id
            })
    return out


def print_analysis_summary(morpho_results: Dict) -> None:
    """Same output as your helper, just moved here."""
    feats = morpho_results['combined_features_df']
    aci = morpho_results['combined_aci_df']
    print(f"Total cells analyzed: {len(feats)}")
    print(f"Total arcs analyzed:  {len(aci)}")
    print(f"Images processed:     {feats['image_id'].nunique()}")
    print("\nCells per image:")
    print(feats.groupby('image_id').size())


def prepare_dna_data_for_analysis(batch_results: Dict) -> List[Dict]:
    """
    Collect blue-channel triplets (original, preprocessed, mask) from a dict of results.
    """
    dna_data_list: List[Dict] = []
    for image_id, data in batch_results.items():
        if 'blue_preprocessed_image' in data and 'blue_mask' in data:
            dna_data_list.append({
                'original_image': data['blue_original_image'],
                'preprocessed_image': data['blue_preprocessed_image'],
                'mask': data['blue_mask'],
                'image_id': image_id
            })

    if dna_data_list:
        print(f"Found {len(dna_data_list)} images for DNA analysis")
    else:
        print("No DNA data found for analysis")
    return dna_data_list
