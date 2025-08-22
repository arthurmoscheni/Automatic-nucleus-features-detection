from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import os
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import least_squares
from skimage.filters import threshold_otsu, frangi
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import (
    binary_erosion, binary_closing, binary_opening, disk, remove_small_objects
)
from skimage.segmentation import clear_border


class ImageAnalyzer:
    """
    Core algorithms for morphology + arc/ACI + wrinkle metrics.
    No plotting or file I/O here.
    """

    def __init__(
        self,
        pixel_size_um: float = None,
        threshold_method: str = 'original',   # kept for backward-compat
        threshold_value: Optional[float] = None
    ):
        self.pixel_size_um = pixel_size_um
        self.pixel_area_um2 = pixel_size_um ** 2
        self.all_features = pd.DataFrame()
        self.all_aci_results = pd.DataFrame()
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value

    # --------------------- preprocessing ---------------------

    def remove_edge_touching_nuclei(self, label_mask: np.ndarray) -> np.ndarray:
        cleaned = clear_border(label_mask)
        return label(cleaned, connectivity=1).astype(label_mask.dtype)

    def remove_touching_cells(
        self,
        label_mask: np.ndarray,
        features_df: pd.DataFrame,
        min_circularity: Optional[float] = None,
        min_area_um2: Optional[float] = None
    ) -> np.ndarray:
        low_circ = 0.4 if min_circularity is None else float(min_circularity)
        low_area = 10.0 if min_area_um2 is None else float(min_area_um2)

        low_circ_labels = set(features_df.loc[features_df['circularity'] < low_circ, 'label'].astype(label_mask.dtype))
        small_area_labels = set(features_df.loc[features_df['area'] < low_area, 'label'].astype(label_mask.dtype))
        remove_labels = low_circ_labels | small_area_labels

        cleaned_mask = np.zeros_like(label_mask)
        cur = 1
        for r in regionprops(label_mask):
            if r.label not in remove_labels:
                coords = tuple(zip(*r.coords))
                cleaned_mask[coords] = cur
                cur += 1

        final_labels = np.unique(cleaned_mask)
        final_labels = final_labels[final_labels != 0]

        # touching-perimeter %
        touching_pct: Dict[int, float] = {}
        for lid in final_labels:
            if lid not in cleaned_mask:  # parity with your original guard
                continue
            region_mask = (cleaned_mask == lid).astype(np.uint8)
            cnts, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            contour = max(cnts, key=lambda c: cv2.arcLength(c, True)).reshape(-1, 2)
            flags = self.touches_other(cleaned_mask, contour, lid)
            touching_pct[lid] = np.sum(flags) / len(flags) if len(flags) > 0 else 0.0

        crowd = [lid for lid, pct in touching_pct.items() if pct > 0.3]
        if len(crowd) > 1:
            circ_tbl = features_df[features_df['label'].isin(crowd)][['label', 'circularity']]
            best = circ_tbl.loc[circ_tbl['circularity'].idxmax(), 'label']
            for lid in crowd:
                if lid != best:
                    cleaned_mask[cleaned_mask == lid] = 0

        return cleaned_mask

    # --------------------- features ---------------------

    def get_image_with_contours_and_features(
        self,
        original_img: np.ndarray,
        label_mask: np.ndarray,
        contour_color: Tuple[int, int, int] = (0, 255, 0),
        linewidth: int = 1
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Returns (overlay_rgb, features_df) — logic matches your original."""
        original_norm = ((original_img - np.min(original_img)) /
                         (np.max(original_img) - np.min(original_img)) * 255).astype(np.uint8)
        rgb = cv2.cvtColor(original_norm, cv2.COLOR_GRAY2BGR)

        rows = []
        for lab in np.unique(label_mask):
            if lab == 0:
                continue
            mask = (label_mask == lab).astype(np.uint8)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, cnts, -1, contour_color, thickness=linewidth)
            area = int(np.sum(mask)) * self.pixel_area_um2
            perim = sum(cv2.arcLength(c, True) for c in cnts) * self.pixel_size_um
            rows.append({'label': lab, 'area': area, 'perimeter': perim})
        return rgb, pd.DataFrame(rows)

    def add_morphology_features(self, features_df: pd.DataFrame, label_mask: np.ndarray) -> pd.DataFrame:
        props = regionprops_table(
            label_mask,
            properties=[
                'label', 'eccentricity', 'solidity', 'orientation',
                'major_axis_length', 'minor_axis_length', 'centroid', 'area', 'perimeter'
            ],
            spacing=(self.pixel_size_um, self.pixel_size_um)
        )
        df = pd.DataFrame(props)
        # keep parity
        df['major_axis_length'] *= self.pixel_size_um
        df['minor_axis_length'] *= self.pixel_size_um
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
        cleaned = self.remove_edge_touching_nuclei(label_mask)
        _overlay, feats = self.get_image_with_contours_and_features(original_image, cleaned)
        feats = self.add_morphology_features(feats, cleaned)
        return self.remove_touching_cells(cleaned, feats, min_circularity, min_area_um2)

    # --------------------- wrinkles (compute only) ---------------------

    def analyze_image_wrinkle_2(
        self,
        original_image: np.ndarray,
        label_mask: np.ndarray,
        image_id: Optional[str] = None,
        boundary_width: int = 5,
        min_inv_size: int = 25,
        sigma: float = 1.0,
        min_ridge_length: int = 25,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Same logic as your function, but returns the overlay image instead of saving it.
        """
        results = []
        final_mask = label_mask.copy()
        labels = np.unique(final_mask)
        labels = labels[labels != 0]

        norm = ((original_image - np.min(original_image)) /
                (np.max(original_image) - np.min(original_image)) * 255).astype(np.uint8)
        rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
        img = original_image.astype(float)

        for lab in labels:
            region_mask = (final_mask == lab)
            vals = img[region_mask]
            mean_sig = vals.mean()
            std_sig = vals.std()
            med_sig = np.median(vals)
            sig_ratio = med_sig / mean_sig if mean_sig > 0 else 1

            if sig_ratio < 0.95 and std_sig > 0.3 * mean_sig:
                bw = max(2, int(boundary_width * 0.6))
            else:
                bw = min(int(boundary_width * 1.2), boundary_width + 2)

            eroded = binary_erosion(region_mask, disk(bw))
            boundary_zone = region_mask ^ eroded
            invag_zone = region_mask & ~boundary_zone

            reg_img = img.copy()
            reg_img[~region_mask] = mean_sig
            sm = ndimage.gaussian_filter(reg_img, sigma=sigma)
            bright = frangi(sm, sigmas=range(1, 4), black_ridges=False)
            dark = frangi(sm, sigmas=range(1, 4), black_ridges=True)
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

            rgb[boundary_zone] = [0, 0, 255]
            rgb[invag] = [255, 165, 0]

            results.append({
                'image_id': image_id,
                'label': lab,
                'boundary_pixels': bcount,
                'invag_pixels': icount,
                'ratio': ratio
            })

        return pd.DataFrame(results), rgb

    # --------------------- arcs / ACI ---------------------

    def get_contours(self, labels: np.ndarray) -> Dict[int, np.ndarray]:
        contours: Dict[int, np.ndarray] = {}
        for lid in np.unique(labels):
            if lid == 0:
                continue
            mask = (labels == lid).astype(np.uint8)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cnts:
                cnt = max(cnts, key=lambda c: cv2.arcLength(c, True))
                contours[lid] = cnt.reshape(-1, 2)
        return contours

    def touches_other(self, labels: np.ndarray, contour: np.ndarray, label_id: int) -> np.ndarray:
        h, w = labels.shape
        flags = np.zeros(len(contour), dtype=bool)
        for i, (x, y) in enumerate(contour):
            xi, yi = int(x), int(y)
            xmin, xmax = max(xi-1, 0), min(xi+1, w-1)
            ymin, ymax = max(yi-1, 0), min(yi+1, h-1)
            patch = labels[ymin:ymax+1, xmin:xmax+1]
            if np.any((patch != label_id) & (patch != 0)):
                flags[i] = True
        return flags

    def split_arcs(self, contour: np.ndarray, flags: np.ndarray) -> List[np.ndarray]:
        arcs: List[np.ndarray] = []
        cur: List[np.ndarray] = []
        for pt, f in zip(contour, flags):
            if not f:
                cur.append(pt)
            else:
                if cur:
                    arcs.append(np.array(cur))
                    cur = []
        if cur:
            arcs.append(np.array(cur))
        if len(arcs) > 1 and not flags[0] and not flags[-1]:
            arcs[0] = np.vstack([arcs[-1], arcs[0]])
            arcs.pop()
        return arcs

    def arc_length(self, arc: np.ndarray) -> float:
        return float(np.linalg.norm(np.diff(arc, axis=0), axis=1).sum())

    def arc_circularity_index(self, arc: np.ndarray) -> float:
        def residual(params):
            cx, cy, r = params
            d = np.sqrt((arc[:, 0]-cx)**2 + (arc[:, 1]-cy)**2)
            return d - r
        x0 = [arc[:, 0].mean(), arc[:, 1].mean(), self.arc_length(arc)/np.pi]
        res = least_squares(residual, x0)
        R = res.x[2]
        eps = np.std(res.fun)
        return max(0.0, 1.0 - eps/R)

    def analyze_arcs(self, labels: np.ndarray, min_length: int = 40, image_id: Optional[str] = None) -> pd.DataFrame:
        contours = self.get_contours(labels)
        rows = []
        for lid, contour in contours.items():
            flags = self.touches_other(labels, contour, lid)
            if np.any(flags):
                for arc in self.split_arcs(contour, flags):
                    if len(arc) >= min_length:
                        rows.append({'image_id': image_id, 'label': lid, 'ACI': self.arc_circularity_index(arc)})
            else:
                region_mask = (labels == lid)
                props = regionprops(region_mask.astype(int))
                if props:
                    r = props[0]
                    area = r.area * self.pixel_area_um2
                    perim = r.perimeter * self.pixel_size_um
                    circ = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0.0
                    rows.append({'image_id': image_id, 'label': lid, 'ACI': circ})
        return pd.DataFrame(rows)

    # --------------------- intensities ---------------------

    @staticmethod
    def calculate_cell_intensity(image: np.ndarray, mask: np.ndarray, image_id: Optional[str] = None) -> pd.DataFrame:
        labels = np.unique(mask)
        labels = labels[labels != 0]
        rows = []
        for lab in labels:
            cell = (mask == lab)
            row = {'label': int(lab)}
            if image.ndim == 3:
                for ch in range(image.shape[2]):
                    pix = image[:, :, ch][cell]
                    row[f'ch{ch}_mean_intensity'] = float(np.mean(pix))
                    row[f'ch{ch}_total_intensity'] = float(np.sum(pix))
            else:
                vals = image[cell]
                row['mean_intensity'] = float(np.mean(vals))
                row['total_intensity'] = float(np.sum(vals))
                row['image_id'] = image_id
            rows.append(row)
        return pd.DataFrame(rows)
