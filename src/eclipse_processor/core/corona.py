"""
Corona detection and maximum radial extent measurement for solar eclipse photography.
Measures sky background level, noise, and inner coronal dynamic range to perform
a contrast pass segmenting visible and faint coronal streamers, and calculates
the maximum coronal radial extension to determine ideal square crop dimensions (+5% margin, capped to image height).
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class CoronaResult:
    r_corona_max: float
    ratio_to_sun: float
    ideal_crop_size: int
    background_median: float
    background_sigma: float
    threshold_value: float
    mask_small: np.ndarray
    downscale_factor: float


class EclipseCoronaDetector:
    def __init__(self,
                 contrast_pct: float = 0.035,
                 k_sigma: float = 8.0,
                 downscale_factor: float = 0.25,
                 margin_ratio: float = 0.05):
        """
        :param contrast_pct: Fraction of inner corona dynamic range required for coronal streamer detection (default 0.035 = 3.5%).
        :param k_sigma: Multiplier above sky noise floor for coronal boundary threshold (default 8.0).
        :param downscale_factor: Scale factor used for background estimation and mask segmentation.
        :param margin_ratio: Extra margin added to corona size for crop (0.05 = +5% margin).
        """
        self.contrast_pct = contrast_pct
        self.k_sigma = k_sigma
        self.downscale_factor = downscale_factor
        self.margin_ratio = margin_ratio

    def detect(self, img: np.ndarray, cx: float, cy: float, r: float) -> CoronaResult:
        """
        Segments the coronal envelope capturing faint streamers and computes maximum radial extension.
        :param img: Full resolution input image (uint16 or uint8).
        :param cx: Sun/Moon center X in full resolution coordinates.
        :param cy: Sun/Moon center Y in full resolution coordinates.
        :param r: Sun/Moon nominal radius in full resolution pixels.
        :return: CoronaResult with max radial extent, crop size, and mask.
        """
        h, w = img.shape[:2]
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        sw = int(round(w * self.downscale_factor))
        sh = int(round(h * self.downscale_factor))
        small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
        small_f = small.astype(np.float32)

        # 1. Measure background and noise from the 4 outer peripheral corners
        patch = max(int(round(40 * self.downscale_factor)), 15)
        corners = np.concatenate([
            small_f[:patch, :patch].ravel(),
            small_f[:patch, -patch:].ravel(),
            small_f[-patch:, :patch].ravel(),
            small_f[-patch:, -patch:].ravel()
        ])
        med_bg = float(np.median(corners))
        mad = float(np.median(np.abs(corners - med_bg)))
        sigma = max(1.4826 * mad, 5.0)

        # 2. Measure inner corona intensity along a ring at 1.15 * r to define dynamic range
        scx = cx * self.downscale_factor
        scy = cy * self.downscale_factor
        sr = r * self.downscale_factor

        n_samples = 36
        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        xs_inner = np.clip(np.round(scx + 1.15 * sr * np.cos(angles)).astype(int), 0, sw - 1)
        ys_inner = np.clip(np.round(scy + 1.15 * sr * np.sin(angles)).astype(int), 0, sh - 1)
        inner_corona_val = float(np.median(small_f[ys_inner, xs_inner]))
        dyn_range = max(inner_corona_val - med_bg, 100.0)

        # 3. Dynamic contrast threshold: tuned to preserve delicate faint streamers (~30% further out)
        thresh_val = med_bg + max(self.k_sigma * sigma, self.contrast_pct * dyn_range)

        # 4. Gaussian smoothing & binary segmentation
        blurred = cv2.GaussianBlur(small_f, (7, 7), 2.0)
        mask = (blurred > thresh_val).astype(np.uint8)

        # 5. Always include the central celestial disc so the corona connects to it
        cv2.circle(mask, (int(round(scx)), int(round(scy))), int(round(sr)), 1, -1)

        # 6. Morphological closing to bridge thin streamer fissures
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        # 7. Connected component analysis: retain the continuous component containing the Moon
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_closed)
        center_x_idx = int(round(np.clip(scx, 0, sw - 1)))
        center_y_idx = int(round(np.clip(scy, 0, sh - 1)))
        center_label = labels[center_y_idx, center_x_idx]

        if center_label > 0:
            corona_mask = (labels == center_label).astype(np.uint8)
        else:
            corona_mask = mask_closed

        # 8. Compute radial distance for all pixels in coronal envelope
        ys, xs = np.where(corona_mask > 0)
        if len(xs) > 0:
            full_xs = xs / self.downscale_factor
            full_ys = ys / self.downscale_factor
            dists = np.sqrt((full_xs - cx)**2 + (full_ys - cy)**2)
            # Use 99.95 percentile to reject isolated single-pixel star spikes on outer boundary
            r_corona_max = float(np.percentile(dists, 99.95))
        else:
            r_corona_max = r * (1.0 + self.margin_ratio)

        ratio_to_sun = float(r_corona_max / r)

        # S_crop = 2 * (1 + margin_ratio) * R_corona_max, strictly capped to image height h
        ideal_crop_size = int(round(2.0 * (1.0 + self.margin_ratio) * r_corona_max))
        ideal_crop_size = min(ideal_crop_size, h)
        # Ensure crop size is even
        if ideal_crop_size % 2 != 0:
            ideal_crop_size -= 1

        return CoronaResult(
            r_corona_max=r_corona_max,
            ratio_to_sun=ratio_to_sun,
            ideal_crop_size=ideal_crop_size,
            background_median=med_bg,
            background_sigma=sigma,
            threshold_value=thresh_val,
            mask_small=corona_mask,
            downscale_factor=self.downscale_factor
        )

    def create_visual_overlay(self,
                              img: np.ndarray,
                              cx: float,
                              cy: float,
                              r: float,
                              res: CoronaResult,
                              filename: str = "") -> np.ndarray:
        """
        Creates an annotated visual inspection image showing:
        - Cyan contour for coronal envelope
        - Green circle for celestial limb + Red center crosshair
        - Magenta circle for maximum radial coronal extent
        - Yellow bounding box for suggested square crop (+5% margin, capped to H)
        """
        h, w = img.shape[:2]
        sw = res.mask_small.shape[1]
        sh = res.mask_small.shape[0]
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)

        # Log stretch for visibility across large dynamic range
        sf = small.astype(np.float32)
        slog = np.log1p(sf)
        slog = ((slog - slog.min()) / (slog.max() - slog.min()) * 255).astype(np.uint8)
        vis = cv2.cvtColor(slog, cv2.COLOR_RGB2BGR) if small.ndim == 3 else cv2.cvtColor(slog, cv2.COLOR_GRAY2BGR)

        # 1. Coronal boundary contour in cyan
        contours, _ = cv2.findContours(res.mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 0), 1, cv2.LINE_AA)

        # 2. Celestial limb in green & center marker in red
        scale = res.downscale_factor
        scx, scy, sr = int(round(cx * scale)), int(round(cy * scale)), int(round(r * scale))
        cv2.circle(vis, (scx, scy), sr, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(vis, (scx, scy), (0, 0, 255), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)

        # 3. Maximum coronal extent in magenta
        s_rcorona = int(round(res.r_corona_max * scale))
        cv2.circle(vis, (scx, scy), s_rcorona, (255, 0, 255), 2, cv2.LINE_AA)

        # 4. Square crop bounding box in yellow (+5% margin, capped to image height)
        crop_half = int(round((res.ideal_crop_size / 2.0) * scale))
        bx1 = max(0, scx - crop_half)
        by1 = max(0, scy - crop_half)
        bx2 = min(sw - 1, scx + crop_half)
        by2 = min(sh - 1, scy + crop_half)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 255), 2, cv2.LINE_AA)

        # 5. Informative text banner
        title = f"{filename}: Sun R={r:.1f}px | Corona Max R={res.r_corona_max:.1f}px ({res.ratio_to_sun:.2f}x)"
        sub = f"Square Crop (+5%, capped <= {h}px): {res.ideal_crop_size}x{res.ideal_crop_size} px | Bg: {res.background_median:.0f} (sig={res.background_sigma:.1f})"
        cv2.putText(vis, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, sub, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2, cv2.LINE_AA)

        return vis
