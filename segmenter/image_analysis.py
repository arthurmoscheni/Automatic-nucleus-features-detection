from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops_table, regionprops
from skimage.morphology import (
    disk, binary_erosion, remove_small_objects, binary_closing, binary_opening
)
from skimage.filters import threshold_otsu, threshold_li, frangi
from scipy.optimize import least_squares
from scipy import ndimage


class ImageAnalyzer:
    """
    Analyze nuclei in microscopy images with morphology and arc circularity metrics.
    """

    def __init__(self, pixel_size_um: float = 0.124, threshold_method: str = 'original', threshold_value: Optional[float] = None):
        """
        Parameters
        ----------
        pixel_size_um : float
            Size of one pixel in micrometers (default for x100 confocal).
        threshold_method : str
            Kept for backward compatibility (not used in current pipeline).
        threshold_value : float | None
            Kept for backward compatibility (not used in current pipeline).
        """
        self.pixel_size_um = pixel_size_um
        self.pixel_area_um2 = pixel_size_um ** 2
        self.all_features = pd.DataFrame()
        self.all_aci_results = pd.DataFrame()
        self.threshold_method = threshold_method     # kept (unused)
        self.threshold_value = threshold_value       # kept (unused)

    # ================== PREPROCESSING ==================

    def remove_edge_touching_nuclei(self, label_mask: np.ndarray) -> np.ndarray:
        """Remove labeled regions that touch the image border; then relabel."""
        cleaned = clear_border(label_mask)
        return label(cleaned, connectivity=1).astype(label_mask.dtype)

    def remove_touching_cells(
        self,
        label_mask: np.ndarray,
        features_df: pd.DataFrame,
        min_circularity: Optional[float] = None,
        min_area_um2: Optional[float] = None
    ) -> np.ndarray:
        """
        Remove cells with low circularity or small area, and resolve heavily touching groups
        by retaining only the cell with the highest circularity.
        """
        low_circ = float(0.4) if min_circularity is None else float(min_circularity)
        low_area = float(10)   if min_area_um2   is None else float(min_area_um2)

        low_circ_labels = set(features_df.loc[features_df['circularity'] < low_circ, 'label'].astype(label_mask.dtype))
        small_area_labels = set(features_df.loc[features_df['area'] < low_area, 'label'].astype(label_mask.dtype))
        remove_labels = low_circ_labels | small_area_labels

        cleaned_mask = np.zeros_like(label_mask)
        current_label = 1
        for region in regionprops(label_mask):
            if region.label not in remove_labels:
                coords = tuple(zip(*region.coords))
                cleaned_mask[coords] = current_label
                current_label += 1

        final_labels = np.unique(cleaned_mask)
        final_labels = final_labels[final_labels != 0]

        touching_info: Dict[int, float] = {}
        for label_id in final_labels:
            if label_id not in cleaned_mask:  # (kept exact logic)
                continue

            region_mask = (cleaned_mask == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue

            contour = max(contours, key=lambda c: cv2.arcLength(c, True)).reshape(-1, 2)
            flags = self.touches_other(cleaned_mask, contour, label_id)
            touching_percentage = np.sum(flags) / len(flags) if len(flags) > 0 else 0.0
            touching_info[label_id] = touching_percentage

        high_touching_cells = [lid for lid, pct in touching_info.items() if pct > 0.3]
        if len(high_touching_cells) > 1:
            circ_tbl = features_df[features_df['label'].isin(high_touching_cells)][['label', 'circularity']]
            best_cell = circ_tbl.loc[circ_tbl['circularity'].idxmax(), 'label']
            cells_to_remove = [cell for cell in high_touching_cells if cell != best_cell]
            for cell_id in cells_to_remove:
                cleaned_mask[cleaned_mask == cell_id] = 0

        return cleaned_mask

    # ================== FEATURE EXTRACTION ==================

    def get_image_with_contours_and_features(
        self,
        original_img: np.ndarray,
        label_mask: np.ndarray,
        contour_color: Tuple[int, int, int] = (0, 255, 0),
        linewidth: int = 1
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Return RGB image with contours; compute area/perimeter features (μm / μm²)."""
        original_norm = ((original_img - np.min(original_img)) /
                         (np.max(original_img) - np.min(original_img)) * 255).astype(np.uint8)
        rgb_img = cv2.cvtColor(original_norm, cv2.COLOR_GRAY2BGR)

        rows = []
        for region_label in np.unique(label_mask):
            if region_label == 0:
                continue
            mask = (label_mask == region_label).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb_img, contours, -1, contour_color, thickness=linewidth)

            area = int(np.sum(mask)) * self.pixel_area_um2
            perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours) * self.pixel_size_um
            rows.append({'label': region_label, 'area': area, 'perimeter': perimeter})

        return rgb_img, pd.DataFrame(rows)

    def add_morphology_features(self, features_df: pd.DataFrame, label_mask: np.ndarray) -> pd.DataFrame:
        """Add regionprops-based morphology and derived features."""
        props = regionprops_table(
            label_mask,
            properties=[
                'label', 'eccentricity', 'solidity', 'orientation',
                'major_axis_length', 'minor_axis_length', 'centroid', 'area', 'perimeter'
            ],
            spacing=(self.pixel_size_um, self.pixel_size_um)
        )
        df_sk = pd.DataFrame(props)

        # Convert axes to μm (spacing already scales them, but keep identical behavior)
        df_sk['major_axis_length'] *= self.pixel_size_um
        df_sk['minor_axis_length'] *= self.pixel_size_um

        df = df_sk.copy()  # (kept: not merging with input features_df)
        df['circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)
        df['aspect_ratio'] = df['major_axis_length'] / df['minor_axis_length']
        return df

    def get_final_mask(
        self,
        original_image: np.ndarray,
        label_mask: np.ndarray,
        min_circularity: Optional[float] = None,
        min_area_um2: Optional[float] = None
    ) -> np.ndarray:
        """Get final mask after removing border-touching and filtering by morphology."""
        cleaned_mask = self.remove_edge_touching_nuclei(label_mask)
        _overlay, features_df = self.get_image_with_contours_and_features(original_image, cleaned_mask)
        features_df = self.add_morphology_features(features_df, cleaned_mask)
        final_mask = self.remove_touching_cells(cleaned_mask, features_df, min_circularity, min_area_um2)
        return final_mask

    # ================== WRINKLES (ridge-based) ==================

    def analyze_image_wrinkle_2(
        self,
        original_image: np.ndarray,
        label_mask: np.ndarray,
        image_id: Optional[str] = None,
        boundary_width: Optional[int] = None,
        min_inv_size: Optional[int] = None,
        sigma: float = 1.0,
        min_ridge_length: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute invagination features for each labeled object using ridge-following.
        Returns DataFrame with label, boundary_pixels, invag_pixels, ratio.
        """
        results = []
        final_mask = label_mask.copy()
        labels = np.unique(final_mask)
        labels = labels[labels != 0]

        norm_img = ((original_image - np.min(original_image)) /
                    (np.max(original_image) - np.min(original_image)) * 255).astype(np.uint8)
        rgb_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2RGB)
        img = original_image.astype(float)

        for lab in labels:
            region_mask = (final_mask == lab)
            region_vals = img[region_mask]
            mean_sig = region_vals.mean()
            std_sig = region_vals.std()
            median_sig = np.median(region_vals)
            sig_ratio = median_sig / mean_sig if mean_sig > 0 else 1

            # (kept) dynamic boundary width logic using provided boundary_width
            if sig_ratio < 0.95 and std_sig > 0.3 * mean_sig:
                bw = max(2, int(boundary_width * 0.6))
            else:
                bw = min(int(boundary_width * 1.2), boundary_width + 2)

            eroded = binary_erosion(region_mask, disk(bw))
            boundary_zone = region_mask ^ eroded
            invag_zone = region_mask & ~boundary_zone

            reg_img = img.copy()
            reg_img[ ~region_mask ] = mean_sig
            sm = ndimage.gaussian_filter(reg_img, sigma=sigma)
            bright = frangi(sm, sigmas=range(1, 4), black_ridges=False)
            dark   = frangi(sm, sigmas=range(1, 4), black_ridges=True)
            ridge_resp = np.maximum(bright, dark) * invag_zone
            if ridge_resp[invag_zone].size > 0:
                thr = np.percentile(ridge_resp[invag_zone], 80)
                ridge_bin = ridge_resp > thr
            else:
                ridge_bin = np.zeros_like(ridge_resp, bool)

            ridge_bin = binary_closing(ridge_bin, disk(1))
            ridge_bin = remove_small_objects(ridge_bin, min_size=min_ridge_length)

            Ixx = ndimage.gaussian_filter(sm, sigma, order=[2, 0])
            Ixy = ndimage.gaussian_filter(sm, sigma, order=[1, 1])
            Iyy = ndimage.gaussian_filter(sm, sigma, order=[0, 2])
            tr = Ixx + Iyy
            det = Ixx * Iyy - Ixy ** 2
            lam1 = 0.5 * (tr + np.sqrt(tr ** 2 - 4 * det + 1e-10))
            lam2 = 0.5 * (tr - np.sqrt(tr ** 2 - 4 * det + 1e-10))
            hstr = np.abs(lam2) * (lam2 < -np.abs(lam1)) * (lam1 < np.abs(lam2))
            hstr *= invag_zone
            if hstr[invag_zone].size > 0:
                hthr = np.percentile(hstr[invag_zone], 75)
                hbin = hstr > hthr
            else:
                hbin = np.zeros_like(hstr, bool)

            invag = (ridge_bin | hbin)
            invag = remove_small_objects(invag, min_size=min_inv_size)
            invag = binary_opening(invag, disk(1))

            bcount = int(boundary_zone.sum())
            icount = int(invag.sum())
            total = bcount + icount
            ratio = icount / total if total > 0 else np.nan

            rgb_img[boundary_zone] = [0, 0, 255]
            rgb_img[invag] = [255, 165, 0]

            if region_mask.sum() > 0:
                ys, xs = np.where(region_mask)
                centroid_y = int(np.mean(ys))
                centroid_x = int(np.mean(xs))
                cv2.putText(
                    rgb_img, f"{ratio:.2f}", (centroid_x, centroid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
                )

            results.append({
                'image_id': image_id,
                'label': lab,
                'boundary_pixels': bcount,
                'invag_pixels': icount,
                'ratio': ratio
            })

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            print(f"Ridge-following invagination visualization saved to: {save_path}")

        return pd.DataFrame(results)

    # ================== VISUALIZATION ==================

    def annotate_cells_with_features(
        self,
        original_img: np.ndarray,
        label_mask: np.ndarray,
        df_features: pd.DataFrame,
        contour_color: Tuple[int, int, int] = (0, 255, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        linewidth: int = 1,
        show_image: bool = False,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """Overlay contours + text features."""
        norm_img = ((original_img - np.min(original_img)) /
                    (np.max(original_img) - np.min(original_img)) * 255).astype(np.uint8)
        rgb_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)

        for _, row in df_features.iterrows():
            label_id = int(row['label'])
            mask = (label_mask == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb_img, contours, -1, contour_color, thickness=linewidth)

            M = cv2.moments(mask)
            if M['m00'] <= 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            text = f"A:{int(row.get('area', 0))} P:{int(row.get('perimeter', 0))}"
            if 'circularity' in row and not pd.isna(row['circularity']):
                text += f" C:{row['circularity']:.2f}"
            if 'eccentricity' in row and not pd.isna(row['eccentricity']):
                text += f" E:{row['eccentricity']:.2f}"

            cv2.putText(rgb_img, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1, cv2.LINE_AA)

        if show_image:
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
            plt.title('Annotated Cells with Features')
            plt.axis('off')
            plt.show()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, rgb_img)
            print(f"Annotated image saved to: {save_path}")

        return rgb_img

    # ================== ARC CIRCULARITY ==================

    def get_contours(self, labels: np.ndarray) -> Dict[int, np.ndarray]:
        """Extract external contours for each label."""
        contours: Dict[int, np.ndarray] = {}
        for label_id in np.unique(labels):
            if label_id == 0:
                continue
            mask = (labels == label_id).astype(np.uint8)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cnts:
                cnt = max(cnts, key=lambda c: cv2.arcLength(c, True))
                contours[label_id] = cnt.reshape(-1, 2)
        return contours

    def touches_other(self, labels: np.ndarray, contour: np.ndarray, label_id: int) -> np.ndarray:
        """Flag contour points that touch other labels."""
        h, w = labels.shape
        flags = np.zeros(len(contour), dtype=bool)
        for i, (x, y) in enumerate(contour):
            xi, yi = int(x), int(y)
            xmin, xmax = max(xi - 1, 0), min(xi + 1, w - 1)
            ymin, ymax = max(yi - 1, 0), min(yi + 1, h - 1)
            patch = labels[ymin:ymax+1, xmin:xmax+1]
            if np.any((patch != label_id) & (patch != 0)):
                flags[i] = True
        return flags

    def split_arcs(self, contour: np.ndarray, flags: np.ndarray) -> List[np.ndarray]:
        """Split contour into continuous arcs of non-touching points."""
        arcs: List[np.ndarray] = []
        current: List[np.ndarray] = []
        for pt, f in zip(contour, flags):
            if not f:
                current.append(pt)
            else:
                if current:
                    arcs.append(np.array(current))
                    current = []
        if current:
            arcs.append(np.array(current))

        # wrap-around (kept identical)
        if len(arcs) > 1 and not flags[0] and not flags[-1]:
            arcs[0] = np.vstack([arcs[-1], arcs[0]])
            arcs.pop()
        return arcs

    def arc_length(self, arc: np.ndarray) -> float:
        """Arc length."""
        return float(np.linalg.norm(np.diff(arc, axis=0), axis=1).sum())

    def arc_circularity_index(self, arc: np.ndarray) -> float:
        """Arc Circularity Index (ACI) via circle fitting."""
        def residual(params):
            cx, cy, r = params
            d = np.sqrt((arc[:, 0] - cx) ** 2 + (arc[:, 1] - cy) ** 2)
            return d - r

        x0 = [arc[:, 0].mean(), arc[:, 1].mean(), self.arc_length(arc) / np.pi]
        res = least_squares(residual, x0)
        R = res.x[2]
        eps = np.std(res.fun)
        return max(0.0, 1.0 - eps / R)

    def analyze_arcs(self, labels: np.ndarray, min_length: int = 40, image_id: Optional[str] = None) -> pd.DataFrame:
        """Analyze mask: extract arcs ≥ min_length and compute ACI, or circularity for isolated cells."""
        contours = self.get_contours(labels)
        rows = []

        for label_id, contour in contours.items():
            flags = self.touches_other(labels, contour, label_id)
            if np.any(flags):
                arcs = self.split_arcs(contour, flags)
                for arc in arcs:
                    if len(arc) >= min_length:
                        aci = self.arc_circularity_index(arc)
                        rows.append({'image_id': image_id, 'label': label_id, 'ACI': aci})
            else:
                region_mask = (labels == label_id)
                props = regionprops(region_mask.astype(int))
                if props:
                    region = props[0]
                    area = region.area * self.pixel_area_um2
                    perimeter = region.perimeter * self.pixel_size_um
                    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
                    rows.append({'image_id': image_id, 'label': label_id, 'ACI': circularity})

        return pd.DataFrame(rows)

    def visualize_arcs_on_image(
        self,
        original_img: np.ndarray,
        label_mask: np.ndarray,
        min_arc_length: int = 40,
        arc_colors: Optional[List[Tuple[int, int, int]]] = None,
        linewidth: int = 2,
        show_image: bool = True,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """Overlay arcs with per-arc ACI labels."""
        norm_img = ((original_img - np.min(original_img)) /
                    (np.max(original_img) - np.min(original_img)) * 255).astype(np.uint8)
        rgb_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2RGB)

        if arc_colors is None:
            arc_colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (255, 165, 0), (128, 0, 128),
                (255, 192, 203), (0, 128, 0),
            ]

        contours = self.get_contours(label_mask)
        arc_count = 0

        for label_id, contour in contours.items():
            flags = self.touches_other(label_mask, contour, label_id)
            arcs = self.split_arcs(contour, flags)

            for arc in arcs:
                if len(arc) >= min_arc_length:
                    color = arc_colors[arc_count % len(arc_colors)]
                    for i in range(len(arc) - 1):
                        pt1 = tuple(map(int, arc[i]))
                        pt2 = tuple(map(int, arc[i + 1]))
                        cv2.line(rgb_img, pt1, pt2, color, linewidth)
                    arc_count += 1

                    aci = self.arc_circularity_index(arc)
                    mid_idx = len(arc) // 2
                    text_pos = tuple(map(int, arc[mid_idx]))
                    cv2.putText(rgb_img, f"ACI:{aci:.2f}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

                    arc_count += 1  # (kept: original double increment)

        if show_image:
            plt.figure(figsize=(12, 10))
            plt.imshow(rgb_img)
            plt.title(f'Arcs Visualization (min_length={min_arc_length}, {arc_count} arcs shown)')
            plt.axis('off')
            plt.show()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            print(f"Arc visualization saved to: {save_path}")

        print(f"Visualized {arc_count} arcs with length >= {min_arc_length} pixels")
        return rgb_img

    # ================== SIGNAL INTENSITY ==================

    @staticmethod
    def calculate_cell_intensity(image: np.ndarray, mask: np.ndarray, image_id: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate signal intensity for every cell in an image using masks.
        (Unchanged behavior: adds image_id only for single-channel images.)
        """
        cell_labels = np.unique(mask)
        cell_labels = cell_labels[cell_labels != 0]
        rows = []

        for lab in cell_labels:
            cell_mask = (mask == lab)
            row = {'label': int(lab)}

            if image.ndim == 3:
                for ch in range(image.shape[2]):
                    ch_pixels = image[:, :, ch][cell_mask]
                    row[f'ch{ch}_mean_intensity'] = float(np.mean(ch_pixels))
                    row[f'ch{ch}_total_intensity'] = float(np.sum(ch_pixels))
            else:
                vals = image[cell_mask]
                row['mean_intensity'] = float(np.mean(vals))
                row['total_intensity'] = float(np.sum(vals))
                row['image_id'] = image_id

            rows.append(row)

        return pd.DataFrame(rows)

    # ================== PIPELINE ==================

    def process_single_image(
        self,
        original_image: np.ndarray,
        preprocessed_image: np.ndarray,
        label_mask: np.ndarray,
        image_id: str,
        min_circularity: float = 0.4,
        min_area_um2: Optional[float] = None,
        min_arc_length: Optional[int] = None,
        visualization: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict:
        """Process a single image through the complete analysis pipeline."""
        print(f"Processing image {image_id}...")
        final_mask = self.get_final_mask(preprocessed_image, label_mask, min_circularity=min_circularity, min_area_um2=min_area_um2)

        final_overlaid_image, final_features_df = self.get_image_with_contours_and_features(preprocessed_image, final_mask)
        final_features_df = self.add_morphology_features(final_features_df, final_mask)
        final_features_df['image_id'] = image_id

        signal_intensities = self.calculate_cell_intensity(original_image, final_mask, image_id)
        aci_results = self.analyze_arcs(final_mask, min_arc_length, image_id)

        # Visualization + saving
        if output_dir:
            out_dir_img = os.path.join(output_dir, image_id)
            os.makedirs(out_dir_img, exist_ok=True)
        else:
            out_dir_img = None

        _ = self.visualize_arcs_on_image(
            preprocessed_image, final_mask, min_arc_length=min_arc_length,
            arc_colors=None, linewidth=2, show_image=visualization
        )

        if out_dir_img:
            ann_path = os.path.join(out_dir_img, "with_annotation.png")
            self.annotate_cells_with_features(
                original_image, final_mask, final_features_df,
                show_image=visualization, save_path=ann_path
            )

        if visualization:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1); plt.imshow(original_image, cmap='gray');     plt.title(f'{image_id} - Original Image');    plt.axis('off')
            plt.subplot(1, 3, 2); plt.imshow(preprocessed_image, cmap='gray');  plt.title(f'{image_id} - Preprocessed Image'); plt.axis('off')
            plt.subplot(1, 3, 3)
            difference = np.abs(original_image.astype(float) - preprocessed_image.astype(float))
            plt.imshow(difference, cmap='hot'); plt.title(f'{image_id} - Difference'); plt.axis('off'); plt.colorbar()
            plt.tight_layout(); plt.show()

            print(f"Original image stats - Min: {original_image.min()}, Max: {original_image.max()}, Mean: {original_image.mean():.2f}")
            print(f"Preprocessed image stats - Min: {preprocessed_image.min()}, Max: {preprocessed_image.max()}, Mean: {preprocessed_image.mean():.2f}")

        if out_dir_img:
            save_path_wrinkles = os.path.join(out_dir_img, "wrinkles.png")
        else:
            save_path_wrinkles = None

        results_wrinkles = self.analyze_image_wrinkle_2(
            original_image,
            final_mask,
            image_id=image_id,
            boundary_width=5,
            min_inv_size=25,
            min_ridge_length=25,
            sigma=1.0,
            save_path=save_path_wrinkles
        )

        if not results_wrinkles.empty:
            print(f"\nWrinkles analysis for {image_id}:")
            for _, row in results_wrinkles.iterrows():
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
            'features_df': final_features_df,
            'overlaid_image': final_overlaid_image,
            'aci_results': aci_results,
            'signal_intensities': signal_intensities,
            'wrinkles_results': results_wrinkles
        }

    def process_multiple_images(
        self,
        image_data_list: List[Dict],
        min_circularity: float = 0.4,
        min_area_um2: float = 10,
        min_arc_length: int = 40,
        visualization: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict:
        """Process multiple images and combine results."""
        all_results = []
        all_features = []
        all_aci_results = []
        wrinkles_results = []
        signal_intensities_results = []

        print(f"Found {len(image_data_list)} images with valid red masks for analysis.")

        for image_data in image_data_list:
            result = self.process_single_image(
                image_data['original_image'],
                image_data['preprocessed_image'],
                image_data['mask'],
                image_data['image_id'],
                min_circularity,
                min_area_um2,
                min_arc_length,
                visualization=visualization,
                output_dir=output_dir
            )

            all_results.append(result)
            all_features.append(result['features_df'])
            if not result['aci_results'].empty:
                all_aci_results.append(result['aci_results'])
            if not result['signal_intensities'].empty:
                signal_intensities_results.append(result['signal_intensities'])
            if not result['wrinkles_results'].empty:
                wrinkles_results.append(result['wrinkles_results'])

        combined_features_df = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()
        combined_aci_df = pd.concat(all_aci_results, ignore_index=True) if all_aci_results else pd.DataFrame()
        combined_signal_intensities_df = pd.concat(signal_intensities_results, ignore_index=True) if signal_intensities_results else pd.DataFrame()
        combined_wrinkles_df = pd.concat(wrinkles_results, ignore_index=True) if wrinkles_results else pd.DataFrame()

        self.all_features = combined_features_df
        self.all_aci_results = combined_aci_df
        self.all_wrinkles_results = combined_wrinkles_df
        self.all_signal_intensities = combined_signal_intensities_df

        return {
            'individual_results': all_results,
            'combined_features_df': combined_features_df,
            'combined_aci_df': combined_aci_df,
            'combined_wrinkles_df': combined_wrinkles_df,
            'combined_signal_intensities_df': combined_signal_intensities_df
        }

    # ================== PLOTTING ==================

    def plot_features(self, df_features: Optional[pd.DataFrame] = None, save_path: Optional[str] = None) -> None:
        """Plot feature distributions (unchanged behavior; uses seaborn)."""
        if df_features is None:
            df_features = self.all_features
        if df_features.empty:
            print("No features to plot. Process images first.")
            return

        features_to_plot = ['area', 'perimeter', 'circularity', 'eccentricity', 'solidity', 'aspect_ratio']
        available = [f for f in features_to_plot if f in df_features.columns]

        sns.set(style="whitegrid")
        fig, axs = plt.subplots(1, len(available), figsize=(15, 4))
        if len(available) == 1:
            axs = [axs]

        for i, feat in enumerate(available):
            sns.histplot(df_features[feat], bins=25, ax=axs[i], kde=True, color='skyblue')
            axs[i].set_title(f"Distribution of {feat}")

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.show()

        plt.figure(figsize=(12, 4))
        for i, feat in enumerate(available):
            plt.subplot(1, len(available), i + 1)
            sns.boxplot(data=df_features, y=feat, color="lightblue")
            plt.title(f"{feat.capitalize()} Boxplot")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path.replace('.png', '_boxplots.png'))
        plt.show()

    def plot_aci_analysis(self, aci_results: Optional[pd.DataFrame] = None, save_path: Optional[str] = None) -> Dict[str, float]:
        """Plot ACI results (scatter + mean±SD lines); return summary stats."""
        if aci_results is None:
            aci_results = self.all_aci_results
        if aci_results.empty:
            print("No ACI results to plot. Process images first.")
            return {}

        aci_values = aci_results['ACI'].values
        aci_mean = float(np.mean(aci_values))
        aci_std = float(np.std(aci_values))

        plt.figure(figsize=(10, 5))
        plt.scatter(range(len(aci_values)), aci_values, color='mediumseagreen', edgecolor='black', label='ACI')
        plt.axhline(aci_mean, color='orange', linestyle='--', label=f'Mean = {aci_mean:.3f}')
        plt.axhline(aci_mean + aci_std, color='red', linestyle=':', label=f'Mean + 1 SD = {aci_mean + aci_std:.3f}')
        plt.axhline(aci_mean - aci_std, color='blue', linestyle=':', label=f'Mean - 1 SD = {aci_mean - aci_std:.3f}')
        plt.xlabel('Arc Index'); plt.ylabel('Arc Circularity Index (ACI)'); plt.title('Scatter Plot of Arc Circularity Index (ACI)')
        plt.legend(); plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.show()

        print(f"ACI Mean: {aci_mean:.4f}")
        print(f"ACI Std: {aci_std:.4f}")
        print(f"ACI Min: {aci_values.min():.4f}")
        print(f"ACI Max: {aci_values.max():.4f}")

        return {'mean': aci_mean, 'std': aci_std, 'min': float(aci_values.min()), 'max': float(aci_values.max())}

    def save_results(self, output_dir: str = "./results/") -> None:
        """Save combined features and ACI tables if present."""
        os.makedirs(output_dir, exist_ok=True)
        if not self.all_features.empty:
            self.all_features.to_csv(f"{output_dir}combined_features.csv", index=False)
            print(f"Combined features saved to {output_dir}combined_features.csv")
        if not self.all_aci_results.empty:
            self.all_aci_results.to_csv(f"{output_dir}aci_results.csv", index=False)
            print(f"ACI results saved to {output_dir}aci_results.csv")


# ================== UTILITIES ==================

def prepare_image_data_for_analysis(batch_results: Dict) -> List[Dict]:
    """Collect red-channel image + mask triplets for analysis."""
    image_data_list: List[Dict] = []
    for image_id, data in batch_results.items():
        if 'red_preprocessed_image' in data and 'red_mask' in data:
            image_data_list.append({
                'original_image': data['red_original_image'],
                'preprocessed_image': data['red_preprocessed_image'],
                'mask': data['red_mask'],
                'image_id': image_id
            })
    return image_data_list


def run_morphological_analysis(
    image_data_list: List[Dict],
    output_dir: str,
    visualization: bool = False,
    pixel_size_um: Optional[float] = None
):
    """Run morphological analysis on prepared image data."""
    analyzer = ImageAnalyzer(pixel_size_um=pixel_size_um if pixel_size_um is not None else 0.124)
    morpho_results = analyzer.process_multiple_images(
        image_data_list=image_data_list,
        min_circularity=0.4,
        min_area_um2=20,
        min_arc_length=40,
        visualization=visualization,
        output_dir=output_dir
    )

    combined_morpho_df = save_analysis_results(morpho_results, output_dir)
    return morpho_results


def save_analysis_results(morpho_results: Dict, output_dir: str) -> pd.DataFrame:
    """Save paths + combined CSV (kept: individual CSV writes commented out)."""
    combined_features = morpho_results['combined_features_df']
    all_aci_results = morpho_results['combined_aci_df']
    wrinkles_results = morpho_results['combined_wrinkles_df']
    signal_intensities = morpho_results['combined_signal_intensities_df']

    print("\nSaving analysis results...")
    os.makedirs(output_dir, exist_ok=True)

    features_path = os.path.join(output_dir, "combined_features.csv")
    print(f"Combined features saved to: {features_path}")

    aci_path = os.path.join(output_dir, "all_aci_results.csv")
    print(f"ACI results saved to: {aci_path}")

    wrinkles_path = os.path.join(output_dir, "combined_wrinkles.csv")
    print(f"Wrinkles results saved to: {wrinkles_path}")

    signal_intensities_path = os.path.join(output_dir, "combined_signal_intensities.csv")
    print(f"Signal intensities results saved to: {signal_intensities_path}")

    combined_path = os.path.join(output_dir, "combined_df.csv")
    combined_df = pd.merge(combined_features, all_aci_results, on=['image_id', 'label'], how='left')
    combined_df = pd.merge(combined_df, wrinkles_results, on=['image_id', 'label'], how='left')
    combined_df = pd.merge(combined_df, signal_intensities, on=['image_id', 'label'], how='left')

    combined_df.to_csv(combined_path, index=False)
    print(f"Combined features, ACI & wrinkles results saved to: {combined_path}")
    return combined_df


def print_analysis_summary(morpho_results: Dict) -> None:
    """Print a short numerical summary."""
    combined_features = morpho_results['combined_features_df']
    all_aci_results = morpho_results['combined_aci_df']

    print(f"Total cells analyzed: {len(combined_features)}")
    print(f"Total arcs analyzed:  {len(all_aci_results)}")
    print(f"Images processed:     {combined_features['image_id'].nunique()}")

    print("\nCells per image:")
    print(combined_features.groupby('image_id').size())
