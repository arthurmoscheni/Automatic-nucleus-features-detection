from __future__ import annotations

import os
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


# ------------------------------------------------------------
# DNADistributionAnalyzer
# ------------------------------------------------------------
class DNADistributionAnalyzer:
    def __init__(
        self,
        original_image: np.ndarray,
        preprocessed_path: np.ndarray | str,
        mask_path: np.ndarray | str,
        pixel_area_um2: float = 0.124**2,
    ):
        """
        Initialize analyzer with image & mask (paths or arrays).
        Channel order assumed: laminB1, hoechst, neun.
        """
        self.image = tiff.imread(preprocessed_path) if isinstance(preprocessed_path, str) else preprocessed_path
        self.mask = tiff.imread(mask_path) if isinstance(mask_path, str) else mask_path
        self.pixel_area_um2 = pixel_area_um2
        self.original_image = original_image

        # Channels
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

    def remove_edge_touching_nuclei(self, label_mask: np.ndarray) -> np.ndarray:
        """Remove regions touching border, then relabel (keeps your original logic)."""
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
                overlap = mask1_d & mask2_d
                if np.any(overlap):
                    perimeter1 = prop1.perimeter
                    perimeter2 = prop2.perimeter
                    boundary1 = mask1 & ~morphology.binary_erosion(mask1, morphology.disk(1))
                    boundary2 = mask2 & ~morphology.binary_erosion(mask2, morphology.disk(1))
                    contact1 = boundary1 & mask2_d
                    contact2 = boundary2 & mask1_d
                    c1 = np.sum(contact1) / perimeter1 if perimeter1 > 0 else 0
                    c2 = np.sum(contact2) / perimeter2 if perimeter2 > 0 else 0
                    if c1 > 0.3 or c2 > 0.3:
                        touching_cells.append((prop1, c1))
                        touching_cells.append((prop2, c2))

            if len(touching_cells) >= 2:
                unique: Dict[int, Tuple] = {}
                for prop, _pct in touching_cells:
                    if prop.label not in unique:
                        circ = 4 * np.pi * prop.area / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
                        unique[prop.label] = (prop, circ)

                if len(unique) == 2:
                    items = list(unique.values())
                    if items[0][1] < items[1][1]:
                        cells_to_remove.add(items[0][0].label)
                    else:
                        cells_to_remove.add(items[1][0].label)
                elif len(unique) > 2:
                    best = max(unique.values(), key=lambda x: x[1])
                    for lbl, (prop, _circ) in unique.items():
                        if lbl != best[0].label:
                            cells_to_remove.add(lbl)

        for cell_label in cells_to_remove:
            cleaned_labels[cleaned_labels == cell_label] = 0

        # NOTE: This returns a relabeled version of `cleaned_mask`, matching the original logic.
        return measure.label(cleaned_mask, connectivity=1).astype(label_mask.dtype)

    # ----------------- Preprocessing -----------------
    def preprocess_hoechst_adaptive(self, cell_mask: np.ndarray, method: str = 'adaptive', debug: bool = False) -> np.ndarray:
        """
        Apply one of several thresholding methods to the Hoechst channel, then clean up.
        Methods: 'adaptive', 'otsu', 'percentile', 'local_otsu', 'multi_threshold', 'original'
        """
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
        local_mean = filters.rank.mean(hoechst_smoothed * cell_mask, morphology.disk(5))
        _kernel = morphology.disk(5)
        local_mean_conv = ndimage.uniform_filter(hoechst_smoothed * cell_mask, size=5)
        local_mean_sq = ndimage.uniform_filter((hoechst_smoothed * cell_mask) ** 2, size=5)
        local_std = np.sqrt(np.maximum(local_mean_sq - local_mean_conv**2, 0))
        thr = local_mean + 0.5 * local_std
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
        thr = mean_i + 1.0 * std_i  # matches your original
        return (hoechst_smoothed > thr) & (cell_mask > 0)

    def _show_preprocessing_steps(self, original: np.ndarray, smoothed: np.ndarray, binary: np.ndarray) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original, cmap='gray');   axes[0].set_title('Original Hoechst'); axes[0].axis('off')
        axes[1].imshow(smoothed, cmap='gray');   axes[1].set_title('Smoothed Hoechst'); axes[1].axis('off')
        axes[2].imshow(binary,   cmap='gray');   axes[2].set_title('Binary Hoechst');   axes[2].axis('off')
        plt.tight_layout(); plt.show()

    # ----------------- Detection & analysis -----------------
    def detect_dna_clusters(
        self,
        hoechst_binary: np.ndarray,
        min_cluster_size: int = 5,
        use_watershed: bool = True,
    ) -> Tuple[Optional[np.ndarray], List]:
        """
        Find clusters via watershed (or CC as fallback) and return labels + properties.
        """
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
        """Relative localization metrics vs cell center."""
        if not clusters:
            return {}
        cell_props = regionprops(cell_mask.astype(int))[0]
        cell_centroid = cell_props.centroid
        cell_area = cell_props.area

        dists, areas = [], []
        for cl in clusters:
            cy, cx = cl.centroid
            d = np.sqrt((cy - cell_centroid[0])**2 + (cx - cell_centroid[1])**2)
            dists.append(d); areas.append(cl.area)

        radius = np.sqrt(cell_area / np.pi)
        ndists = [d / radius for d in dists]

        return {
            'mean_distance_from_center': float(np.mean(ndists)),
            'std_distance_from_center': float(np.std(ndists)),
            'peripheral_clusters': int(sum(1 for d in ndists if d > 0.7)),
            'central_clusters': int(sum(1 for d in ndists if d < 0.3)),
            'cluster_dispersion': float(np.std(ndists)),
        }

    def analyze_texture_features(self, hoechst_cell: np.ndarray, cell_mask: np.ndarray) -> Dict:
        """
        Intensity/texture stats within the cell (uses self.hoechst exactly as in your code).
        """
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

    # ----------------- Single/all cells -----------------
    def analyze_single_cell_improved(self, cell_id: int, threshold_method: str = 'original', debug: bool = False) -> Optional[Dict]:
        """Analyze one cell: threshold → clusters → localization → texture."""
        cell_mask = (self.mask == cell_id).astype(int)
        if np.sum(cell_mask) == 0:
            return None

        hoechst_binary = self.preprocess_hoechst_adaptive(cell_mask, method=threshold_method, debug=debug)

        if debug:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1); plt.imshow(self.hoechst * cell_mask, cmap='gray'); plt.title('Original Hoechst in Cell'); plt.axis('off')
            plt.subplot(1, 2, 2); plt.imshow(hoechst_binary, cmap='gray');           plt.title('Binary Hoechst');          plt.axis('off')
            plt.tight_layout(); plt.show()

        labeled_clusters, cluster_props = self.detect_dna_clusters(hoechst_binary, use_watershed=True)
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
        """Analyze every labeled cell."""
        self.mask = self.remove_edge_touching_nuclei(self.mask)
        unique_cells = np.unique(self.mask)
        unique_cells = unique_cells[unique_cells > 0]

        for cid in unique_cells:
            res = self.analyze_single_cell_improved(cid)
            if res:
                self.results.append(res)

        return pd.DataFrame(self.results)

    # ----------------- Visualizations kept in use -----------------
    def visualize_clusters_overlay(self, threshold_method: str = 'original', show_image: bool = False, save_path: Optional[str] = None) -> None:
        """
        Show original Hoechst with DNA clusters (orange) and all cell boundaries (cyan) for all cells.
        """
        unique_cells = np.unique(self.mask)
        unique_cells = unique_cells[unique_cells > 0]
        if len(unique_cells) == 0:
            print("No cells found in the mask")
            return

        all_clusters = np.zeros_like(self.hoechst)
        all_cell_boundaries = np.zeros_like(self.hoechst, dtype=bool)
        total_clusters = 0

        print(f"Processing {len(unique_cells)} cells...")
        for cid in unique_cells:
            cell_mask = (self.mask == cid).astype(int)
            if np.sum(cell_mask) == 0:
                continue

            hoechst_binary = self.preprocess_hoechst_adaptive(cell_mask, method=threshold_method)
            labeled_clusters, cluster_props = self.detect_dna_clusters(hoechst_binary, use_watershed=True)

            if labeled_clusters is not None and np.max(labeled_clusters) > 0:
                all_clusters[labeled_clusters > 0] = 1
                total_clusters += len(cluster_props)

            cell_boundary = cell_mask - morphology.binary_erosion(cell_mask, morphology.disk(1))
            all_cell_boundaries |= cell_boundary.astype(bool)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(self.hoechst, cmap='gray'); axes[0].set_title('Original Hoechst Image'); axes[0].axis('off')

        axes[1].imshow(self.hoechst, cmap='gray')
        axes[1].contour(all_cell_boundaries.astype(int), colors='cyan', linewidths=1)
        axes[1].set_title(f'All Cell Boundaries\n({len(unique_cells)} cells)'); axes[1].axis('off')

        axes[2].imshow(self.hoechst, cmap='gray')
        if np.max(all_clusters) > 0:
            overlay = np.zeros((*self.hoechst.shape, 4))
            overlay[all_clusters > 0] = [1, 0.5, 0, 0.7]  # orange RGBA
            axes[2].imshow(overlay)
            axes[2].set_title(f'All DNA Clusters Highlighted\n({total_clusters} clusters total)')
        else:
            axes[2].set_title('No Clusters Detected')
        axes[2].contour(all_cell_boundaries.astype(int), colors='cyan', linewidths=1, alpha=0.8)
        axes[2].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_image:
            plt.show()
        else:
            plt.close()

        print("\nWhole Image DNA Cluster Analysis:")
        print(f"  - Total cells processed: {len(unique_cells)}")
        print(f"  - Total clusters detected: {total_clusters}")
        print(f"  - Average clusters per cell: {total_clusters / len(unique_cells):.2f}")


# ------------------------------------------------------------
# Batch helpers
# ------------------------------------------------------------
def prepare_dna_data_for_analysis(batch_results: Dict) -> List[Dict]:
    """Collect blue-channel image + mask triplets for analysis."""
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


def run_dna_analysis(dna_data_list: List[Dict], output_dir: str, visualization: bool = False) -> pd.DataFrame:
    """
    Run analysis for each image in `dna_data_list`.
    """
    results: List[pd.DataFrame] = []

    for data in dna_data_list:
        original_image = data['original_image']
        preprocessed_image = data['preprocessed_image']
        mask = data['mask']
        image_id = data['image_id']

        analyzer = DNADistributionAnalyzer(original_image, preprocessed_image, mask)
        cell_results = analyzer.analyze_all_cells()

        out_dir_img = os.path.join(output_dir, image_id)
        os.makedirs(out_dir_img, exist_ok=True)
        save_path = os.path.join(out_dir_img, "dna_analysis_summary.png")
        analyzer.visualize_clusters_overlay(threshold_method='original', show_image=visualization, save_path=save_path)

        if cell_results is not None and not cell_results.empty:
            cell_results['image_id'] = image_id
            results.append(cell_results)

    # NOTE: The following concat mirrors your original logic and will raise if `results` is empty.
    final_df = pd.concat(results, ignore_index=True)
    save_dna_results(final_df, output_dir)

    if results:
        return final_df
    else:
        return pd.DataFrame()


def save_dna_results(dna_df: pd.DataFrame, output_dir: str) -> None:
    """Save combined features to CSV."""
    print("\nSaving analysis results...")
    os.makedirs(output_dir, exist_ok=True)
    features_path = os.path.join(output_dir, "dna_features.csv")
    dna_df.to_csv(features_path, index=False)
    print(f"Dna features saved to: {features_path}")


def quick_dna_insights(results_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Quick 1×3 panel of cluster count, DNA area, and localization."""
    if results_df.empty:
        print("No data to visualize")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(results_df['num_clusters'], bins=15, alpha=0.7, color='skyblue')
    axes[0].set_title(f'DNA Clusters per Cell\n(Mean: {results_df["num_clusters"].mean():.1f})')
    axes[0].set_xlabel('Number of Clusters')

    axes[1].scatter(results_df['num_clusters'], results_df['total_dna_area_um2'], alpha=0.6, color='coral')
    axes[1].set_xlabel('Number of Clusters'); axes[1].set_ylabel('Total DNA Area (μm²)')
    axes[1].set_title('Clusters vs DNA Area')

    axes[2].hist(results_df['mean_distance_from_center'], bins=15, alpha=0.7, color='lightgreen')
    axes[2].set_title('DNA Localization\n(0=center, 1=periphery)')
    axes[2].set_xlabel('Distance from Center')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
