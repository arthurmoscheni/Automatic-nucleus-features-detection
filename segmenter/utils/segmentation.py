from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

def calculate_overlap(mask1: np.ndarray, mask2: np.ndarray, label1: int, label2: int) -> float:
    region1 = mask1 == label1
    region2 = mask2 == label2
    inter = np.sum(region1 & region2)
    union = np.sum(region1 | region2)
    return float(inter / union) if union > 0 else 0.0

def filter_masks_by_overlap(
    target_masks: np.ndarray,
    reference_masks: np.ndarray,
    overlap_threshold: float = 0.3,
) -> Tuple[np.ndarray, Dict[int, int]]:
    target_labels = np.unique(target_masks)[1:]   # drop background
    reference_labels = np.unique(reference_masks)[1:]

    out = np.zeros_like(target_masks)
    mapping: Dict[int, int] = {}
    new_label = 1

    for t in target_labels:
        best = 0.0
        for r in reference_labels:
            ov = calculate_overlap(target_masks, reference_masks, int(t), int(r))
            if ov > best:
                best = ov
        if best > overlap_threshold:
            out[target_masks == t] = new_label
            mapping[int(t)] = new_label
            new_label += 1

    return out, mapping

def flatten_segmentation_results(results: Dict) -> Dict:
    """Flatten your result dict to a flat dict (unchanged behavior)."""
    flattened = {"original_image": results["original_image"]}
    for ch, img in results["original_image"].items():
        flattened[f"{ch}_original_image"] = img
    for ch, img in results["preprocessed_channels"].items():
        flattened[f"{ch}_preprocessed_image"] = img
    for name, mask in results["masks"].items():
        flattened[f"{name}_mask"] = mask
    for ch, flow in results["flows"].items():
        flattened[f"{ch}_flow"] = flow
    for ch, style in results["styles"].items():
        flattened[f"{ch}_style"] = style
    return flattened
