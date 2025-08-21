from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from skimage import io as skio, util
from scipy.ndimage import gaussian_filter

# lazy import to avoid loading Cellpose at package import time
def _load_cellpose(model_type: str):
    from cellpose import models
    return models.CellposeModel(model_type=model_type)

from utils.segmentation import (
    calculate_overlap,
    filter_masks_by_overlap,
    flatten_segmentation_results,
)
from io_utils.segmentation import (
    save_segmentation_files,
    save_preprocessed_images,
    save_original_image,
)
from viz.segmentation import display_results


class ImageSegmentationPipeline:
    """A clean pipeline for multi-channel image segmentation using Cellpose."""

    def __init__(self, model_type: str = "nuclei"):
        self.model_type = model_type
        self.model = _load_cellpose(model_type)
        print("Using CPU")

    # --------- preprocessing ---------
    @staticmethod
    def preprocess_image(
        img: np.ndarray,
        apply_gaussian: bool = True,
        sigma: float = 1.0,
        enhance_contrast: bool = True,
        alpha: float = 1.0,
        beta: float = 0.1,
    ) -> np.ndarray:
        """Preprocess one channel; identical behavior."""
        img_norm = img.astype(np.float32)
        if img_norm.max() > 0:
            img_norm /= img_norm.max()
        if enhance_contrast:
            img_norm = np.clip(alpha * img_norm + beta, 0, 1)
        if apply_gaussian and img_norm.max() > 1e-3:
            img_norm = gaussian_filter(img_norm, sigma=sigma)
        return img_norm

    def load_and_preprocess_multichannel(
        self,
        image_path: str,
        preprocess_params: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Load RGB-like image, return original + dict of preprocessed channels."""
        if preprocess_params is None:
            preprocess_params = dict(
                apply_gaussian=True, sigma=1.0, enhance_contrast=True, alpha=1.0, beta=0.1
            )
        original_img = skio.imread(image_path)
        print(f"Loaded image shape: {original_img.shape}, dtype: {original_img.dtype}")

        if original_img.ndim != 3 or original_img.shape[-1] < 3:
            raise ValueError("Image must have at least 3 channels (RGB)")

        channels = {}
        names = ["blue", "green", "red"]
        for i, name in enumerate(names):
            ch = original_img[:, :, i]
            channels[name] = self.preprocess_image(ch, **preprocess_params)
            print(
                f"Preprocessed {name}: shape {channels[name].shape}, "
                f"range [{channels[name].min():.3f}, {channels[name].max():.3f}]"
            )
        return original_img, channels

    # --------- segmentation ---------
    def segment_channel(
        self,
        image: np.ndarray,
        diameter: Optional[float] = None,
        flow_threshold: float = 0.5,
        cellprob_threshold: float = 0,
        progress: bool = False,
    ):
        """Run Cellpose on a single channel; identical behavior."""
        masks, flows, styles = self.model.eval(
            image,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            progress=progress,
        )
        num_objects = len(np.unique(masks)) - 1
        print(f"Segmented {num_objects} objects")
        return masks, flows, styles

    # --------- pipeline ---------
    def process_multichannel_image(
        self,
        image_path: str,
        segment_params: Optional[Dict] = None,
        filter_by_green: bool = True,
        overlap_threshold: float = 0.3,
    ) -> Dict:
        """Load → preprocess → segment each channel → optional overlap filtering."""
        if segment_params is None:
            segment_params = dict(
                diameter=None, flow_threshold=0.5, cellprob_threshold=0, progress=True
            )

        original_img, channels = self.load_and_preprocess_multichannel(image_path)

        masks, flows, styles = {}, {}, {}
        original_imgs = {
            "blue": original_img[:, :, 0],
            "green": original_img[:, :, 1],
            "red": original_img[:, :, 2],
        }

        for name, ch in channels.items():
            print(f"\nSegmenting {name} channel...")
            mask, flow, style = self.segment_channel(ch, **segment_params)
            masks[name], flows[name], styles[name] = mask, flow, style

        if filter_by_green and "green" in masks:
            print(f"\nFiltering masks by green channel overlap (threshold={overlap_threshold})...")
            if "red" in masks:
                masks["red"], _ = filter_masks_by_overlap(masks["red"], masks["green"], overlap_threshold)
                print(f"Red masks: {len(np.unique(masks['red']))-1} kept after filtering")
            if "blue" in masks:
                # original logic: blue filtered by *red* (kept)
                masks["blue"], _ = filter_masks_by_overlap(masks["blue"], masks["red"], overlap_threshold)
                print(f"Blue masks: {len(np.unique(masks['blue']))-1} kept after filtering")

        return {
            "original_image": original_imgs,
            "preprocessed_channels": channels,
            "masks": masks,
            "flows": flows,
            "styles": styles,
        }


def run_batch_segmentation(
    image_paths: List[str], output_dir: str, visualization: bool = False
) -> Dict[str, Dict]:
    """Run the pipeline on multiple images; save artifacts; return flattened results per image."""
    os.makedirs(output_dir, exist_ok=True)
    pipe = ImageSegmentationPipeline(model_type="nuclei")
    batch_results: Dict[str, Dict] = {}

    for image_path in image_paths:
        base = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\nProcessing {base}…")
        try:
            results = pipe.process_multichannel_image(image_path)

            image_out = os.path.join(output_dir, base)
            seg_dir = os.path.join(image_out, "segmentation")
            orig_dir = os.path.join(image_out, "original")
            prep_dir = os.path.join(image_out, "preprocessed")
            for d in (seg_dir, orig_dir, prep_dir):
                os.makedirs(d, exist_ok=True)

            save_segmentation_files(results, base, seg_dir)
            save_preprocessed_images(results, base, prep_dir)
            save_original_image(results, base, orig_dir)

            flat = flatten_segmentation_results(results)
            batch_results[base] = flat
            print(f"  → Processed {base} successfully")
            if visualization:
                display_results(results)

        except Exception as e:
            print(f"  → Failed on {base}: {e}")

    return batch_results
