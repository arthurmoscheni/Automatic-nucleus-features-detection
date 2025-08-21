from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff

from scipy import ndimage
from skimage import measure, morphology, filters, feature, segmentation
from skimage.filters import threshold_otsu, threshold_local, gaussian
from skimage.measure import regionprops, label
from skimage.segmentation import clear_border


class DNADistributionAnalyzer:
    """
    Core analysis class (no file I/O and no high-level plotting).
    """

    def __init__(
        self,
        original_image: np.ndarray,
        preprocessed_path: np.ndarray | str,
        mask_path: np.ndarray | str,
        pixel_area_um2: float = 0.124**2,
    ):
        self.image = tiff.imread(preprocessed_path) if isinstance(preprocessed_path, str) else preprocessed_path
        self.mask = tiff.imread(mask_path) if isinstance(mask_path, str) else mask_path
        self.pixel_area_um2 = pixel_area_um2
        self.original_image = original_image

        # Channels (assumes laminB1, hoechst, neun)
        if self.image.ndim == 3:
            if self.image.shape[0] == 3:  # channels first
                self.laminb1, self.hoechst, self.neun = self.image
            else:  # channels last
                self.laminb1 = self.image[:, :, 0]
                self.hoechst = self.image[:, :, 1]
                self.neun = self.image[:, :, 2]
        else:
            self.hoechst = self.image
            self.laminb1 = self.image
            self.neun = self.image

        self.results: List[Dict] = []

    # ---------- mask cleaning ----------
    def remove_edge_touching_nuclei(self, label_mask: np.ndarray) -> np.ndarray:
        cleaned_mask = clear_border(label_mask)
        cleaned_labels = measure.label(cleaned_mask, connectivity=1)

        props = regionprops(cleaned_labels)
        cells_to_remove: set[int] = set()

        for i, prop1 in enumerate(props):
            if prop1.label in cells_to_remove:
                continue

            touching_cells: List[Tuple] = []
            for j, prop2 in enumerate(props):
                if i >= j or prop2.label in cells_to_remove:
                    continue

                mask1 = cleaned_labels == prop1.label
                mask2 = cleaned_labels == prop2.label
                mask1_d = morphology.binary_dilation(mask1, morphology.disk(1))
                mask2_d = morphology.binary_dilation(mask2, morphology.disk(1))
                if np.any(mask1_d & mask2_d):
                    p1, p2 = prop1.perimeter, prop2.perimeter
                    b1 = mask1 & ~morphology.binary_erosion(mask1, morphology.disk(1))
                    b2 = mask2 & ~morphology.binary_erosion(mask2, morphology.disk(1))
                    c1 = np.sum(b1 & mask2_d) / p1 if p1 > 0 else 0
                    c2 = np.sum(b2 & mask1_d) / p2 if p2 > 0 else 0
                    if c1 > 0.3 or c2 > 0.3:
                        touching_cells.append((prop1, c1))
                        touching_cells.append((prop2, c2))

            if len(touching_cells) >= 2:
                unique: Dict[int, Tuple] = {}
                for prop, _ in touching_cells:
                    if prop.label not in unique:
                        circ = 4 * np.pi * prop.area / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
                        unique[prop.label] = (prop, circ)

                if len(unique) == 2:
                    items = list(unique.values())
                    cells_to_remove.add(items[0][0].label if items[0][1] < items[1][1] else items[1][0].label)
                elif len(unique) > 2:
                    best = max(unique.values(), key=lambda x: x[1])
                    for lbl, (prop, _circ) in unique.items():
                        if lbl != best[0].label:
                            cells_to_remove.add(lbl)

        for cell_label in cells_to_remove:
            cleaned_labels[cleaned_labels == cell_label] = 0

        return measure.label(cleaned_mask, connectivity=1).astype(label_mask.dtype)

    # ---------- preprocessing ----------
    def preprocess_hoechst_adaptive(self, cell_mask: np.ndarray, method: str = 'adaptive', debug: bool = False) -> np.ndarray:
        hoechst_cell = self.hoechst * cell_mask
        hoechst_smoothed = gaussian(hoechst_cell, sigma=0.5)

        if method == 'adaptive':
            hoechst_binary = self._adaptive_threshold(hoechst_smoothed, cell_mask)
        elif method == 'otsu':
            hoechst_binary = self._otsu_threshold(hoechst_smoothed, cell_mask)
        elif method == 'percentile':
            hoechst_binary = self._percentile_threshold(hoechst_smoothed, cell_mask, percentile=75)
        elif method == 'local_otsu':
            hoechst_binary = self._local_otsu_threshold(hoechst_smoothed, cell_mask)
        elif method == 'multi_threshold':
            hoechst_binary = self._multi_threshold(hoechst_smoothed, cell_mask)
        else:
            hoechst_binary = self._original_threshold(hoechst_smoothed, cell_mask)

        if debug:
            self._show_preprocessing_steps(hoechst_cell, hoechst_smoothed, hoechst_binary)

        hoechst_binary = morphology.remove_small_objects(hoechst_binary, min_size=5)
        hoechst_binary = morphology.binary_closing(hoechst_binary, morphology.disk(1))
        hoechst_binary = morphology.binary_opening(hoechst_binary, morphology.disk(1))
        return hoechst_binary

    def _adaptive_threshold(self, hoechst_smoothed: np.ndarray, cell_mask: np.ndarray, debug: bool = False) -> np.ndarray:
        filters.rank.mean(hoechst_smoothed * cell_mask, morphology.disk(5))  # local mean (kept for parity)
        local_mean_conv = ndimage.uniform_filter(hoechst_smoothed * cell_mask, size=5)
        local_mean_sq = ndimage.uniform_filter((hoechst_smoothed * cell_mask) ** 2, size=5)
        local_std = np.sqrt(np.maximum(local_mean_sq - local_mean_conv**2, 0))
        thr = local_mean_conv + 0.5 * local_std
        return (hoechst_smoothed > thr) & (cell_mask > 0)

    def _otsu_threshold(self, hoechst_smoothed: np.ndarray, cell_mask: np.ndarray, debug: bool = False) -> np.ndarray:
        values = hoechst_smoothed[cell_mask > 0]
        if len(values) == 0:
            return np.zeros_like(hoechst_smoothed, dtype=bool)
        try:
            thr = threshold_otsu(values)
        except Exception:
            thr = np.percentile(values, 80)
        return (hoechst_smoothed > thr) & (cell_mask > 0)

    def _percentile_threshold(self, hoechst_smoothed: np.ndarray, cell_mask: np.ndarray, percentile: int = 75, debug: bool = False) -> np.ndarray:
        values = hoechst_smoothed[cell_mask > 0]
        if len(values) == 0:
            return np.zeros_like(hoechst_smoothed, dtype=bool)
        thr = np.percentile(values, percentile)
        return (hoechst_smoothed > thr) & (cell_mask > 0)

    def _local_otsu_threshold(self, hoechst_smoothed: np.ndarray, cell_mask: np.ndarray, debug: bool = False) -> np.ndarray:
        local_thresh = threshold_local(hoechst_smoothed, block_size=15, method='gaussian')
        return (hoechst_smoothed > local_thresh) & (cell_mask > 0)

    def _multi_threshold(self, hoechst_smoothed: np.ndarray, cell_mask: np.ndarray, debug: bool = False) -> np.ndarray:
        b1 = self._otsu_threshold(hoechst_smoothed, cell_mask)
        b2 = self._percentile_threshold(hoechst_smoothed, cell_mask, percentile=70)
        b3 = self._adaptive_threshold(hoechst_smoothed, cell_mask)
        return b1 | b2 | b3

    def _original_threshold(self, hoechst_smoothed: np.ndarray, cell_mask: np.ndarray, debug: bool = False) -> np.ndarray:
        values = hoechst_smoothed[cell_mask > 0]
        if len(values) == 0:
            return np.zeros_like(hoechst_smoothed, dtype=bool)
        mean_i = np.mean(values)
        std_i = np.std(values)
        thr = mean_i + 1.0 * std_i
        return (hoechst_smoothed > thr) & (cell_mask > 0)

    def _show_preprocessing_steps(self, original: np.ndarray, smoothed: np.ndarray, binary: np.ndarray) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original, cmap='gray');   axes[0].set_title('Original Hoechst'); axes[0].axis('off')
        axes[1].imshow(smoothed, cmap='gray');   axes[1].set_title('Smoothed Hoechst'); axes[1].axis('off')
        axes[2].imshow(binary,   cmap='gray');   axes[2].set_title('Binary Hoechst');   axes[2].axis('off')
        plt.tight_layout(); plt.show()

    # ---------- clustering + features ----------
    def detect_dna_clusters(
        self,
        hoechst_binary: np.ndarray,
        min_cluster_size: int = 5,
        use_watershed: bool = True,
    ) -> Tuple[Optional[np.ndarray], List]:
        if hoechst_binary is None or not np.any(hoechst_binary):
            return None, []

        if use_watershed:
            dist = ndimage.distance_transform_edt(hoechst_binary)
            peaks = feature.peak_local_max(dist, min_distance=5, threshold_abs=0.5)
            markers = np.zeros_like(hoechst_binary, dtype=int)
            if len(peaks) > 0:
                coords = tuple(peaks.T)
                markers[coords] = np.arange(len(peaks)) + 1
                labeled_clusters = segmentation.watershed(-dist, markers, mask=hoechst_binary)
            else:
                labeled_clusters = label(hoechst_binary)
        else:
            labeled_clusters = measure.label(hoechst_binary)

        props = regionprops(labeled_clusters)
        valid = [p for p in props if p.area >= min_cluster_size]
        return labeled_clusters, valid

    def analyze_cluster_localization(self, clusters: List, cell_mask: np.ndarray) -> Dict:
        if not clusters:
            return {}
        cell_props = regionprops(cell_mask.astype(int))[0]
        cy0, cx0 = cell_props.centroid
        area = cell_props.area

        dists = []
        for cl in clusters:
            cy, cx = cl.centroid
            dists.append(np.sqrt((cy - cy0) ** 2 + (cx - cx0) ** 2))

        radius = np.sqrt(area / np.pi)
        ndists = [d / radius for d in dists]

        return {
            'mean_distance_from_center': float(np.mean(ndists)),
            'std_distance_from_center': float(np.std(ndists)),
            'peripheral_clusters': int(sum(1 for d in ndists if d > 0.7)),
            'central_clusters': int(sum(1 for d in ndists if d < 0.3)),
            'cluster_dispersion': float(np.std(ndists)),
        }

    def analyze_texture_features(self, hoechst_cell: np.ndarray, cell_mask: np.ndarray) -> Dict:
        hoechst_masked = self.hoechst * cell_mask
        vals = hoechst_masked[cell_mask > 0]
        if len(vals) == 0:
            return {}

        feats = {
            'mean_intensity': float(np.mean(vals)),
            'intensity_std': float(np.std(vals)),
            'intensity_skewness': float(self._calculate_skewness(vals)),
            'intensity_kurtosis': float(self._calculate_kurtosis(vals)),
            'intensity_entropy': float(self._calculate_entropy(vals)),
            'intensity_cv': float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 0.0,
        }

        if hoechst_masked.shape[0] > 16 and hoechst_masked.shape[1] > 16:
            lbp = feature.local_binary_pattern(hoechst_masked, P=8, R=1, method='uniform')
            lbp_masked = lbp[cell_mask > 0]
            feats['lbp_uniformity'] = float(len(np.unique(lbp_masked)) / len(lbp_masked))

        return feats

    def _calculate_skewness(self, data: np.ndarray) -> float:
        if len(data) < 3: return 0.0
        mean, std = np.mean(data), np.std(data)
        if std == 0: return 0.0
        return float(np.mean(((data - mean) / std) ** 3))

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        if len(data) < 4: return 0.0
        mean, std = np.mean(data), np.std(data)
        if std == 0: return 0.0
        return float(np.mean(((data - mean) / std) ** 4) - 3)

    def _calculate_entropy(self, data: np.ndarray) -> float:
        hist, _ = np.histogram(data, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))

    # ---------- per-cell / all-cells ----------
    def analyze_single_cell_improved(self, cell_id: int, threshold_method: str = 'original', debug: bool = False) -> Optional[Dict]:
        cell_mask = (self.mask == cell_id).astype(int)
        if np.sum(cell_mask) == 0:
            return None

        hoechst_binary = self.preprocess_hoechst_adaptive(cell_mask, method=threshold_method, debug=debug)

        if debug:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1); plt.imshow(self.hoechst * cell_mask, cmap='gray'); plt.title('Original Hoechst in Cell'); plt.axis('off')
            plt.subplot(1, 2, 2); plt.imshow(hoechst_binary, cmap='gray');           plt.title('Binary Hoechst');          plt.axis('off')
            plt.tight_layout(); plt.show()

        _, cluster_props = self.detect_dna_clusters(hoechst_binary, use_watershed=True)
        localization = self.analyze_cluster_localization(cluster_props, cell_mask)
        texture = self.analyze_texture_features(self.original_image, cell_mask)

        areas_um2 = [prop.area * self.pixel_area_um2 for prop in cluster_props]
        result = {
            'cell_id': cell_id,
            'num_clusters': len(cluster_props),
            'total_dna_area_um2': float(sum(areas_um2)),
            'mean_cluster_size_um2': float(np.mean(areas_um2)) if areas_um2 else 0.0,
            'cluster_size_cv': float(np.std(areas_um2) / np.mean(areas_um2)) if areas_um2 and np.mean(areas_um2) > 0 else 0.0,
        }
        result.update(localization)
        result.update(texture)
        return result

    def analyze_all_cells(self) -> pd.DataFrame:
        self.mask = self.remove_edge_touching_nuclei(self.mask)
        unique_cells = np.unique(self.mask)
        unique_cells = unique_cells[unique_cells > 0]

        for cid in unique_cells:
            res = self.analyze_single_cell_improved(cid)
            if res:
                self.results.append(res)

        return pd.DataFrame(self.results)
