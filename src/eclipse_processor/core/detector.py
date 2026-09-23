"""
Robust Sun and Moon circle detector for solar eclipse photography across all totality phases.
Immune to bias from:
- Contact diamond rings and Baily's beads
- Solar prominences and hydrogen-alpha flares
- Egress/exit flares and blooming arcs
"""

import cv2
import numpy as np
import tifffile
from typing import Tuple, Optional, List, Dict, Any


def fit_circle_algebraic(points: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Fit a circle (cx, cy, r) to 2D points using Pratt/Taubin algebraic method.
    Returns (cx, cy, r, rmse_residual).
    """
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    cx = -sol[0] / 2.0
    cy = -sol[1] / 2.0
    val = cx**2 + cy**2 - sol[2]
    if val <= 0:
        return 0.0, 0.0, 0.0, 999999.0
    r = np.sqrt(val)
    dists = np.sqrt((x - cx)**2 + (y - cy)**2)
    rmse = np.sqrt(np.mean((dists - r)**2))
    return float(cx), float(cy), float(r), float(rmse)


def fit_circle_fixed_r(points: np.ndarray,
                       r_fixed: float,
                       init_cx: float,
                       init_cy: float,
                       max_iter: int = 10) -> Tuple[float, float]:
    """
    Fit circle center (cx, cy) to 2D points with a strictly fixed radius r_fixed.
    Uses iterative fixed-point projection.
    """
    cx, cy = init_cx, init_cy
    for _ in range(max_iter):
        dx = points[:, 0] - cx
        dy = points[:, 1] - cy
        dists = np.sqrt(dx**2 + dy**2)
        valid = dists > 1e-4
        if not np.any(valid):
            break
        ux = dx[valid] / dists[valid]
        uy = dy[valid] / dists[valid]
        target_cx = points[valid, 0] - r_fixed * ux
        target_cy = points[valid, 1] - r_fixed * uy
        new_cx = float(np.mean(target_cx))
        new_cy = float(np.mean(target_cy))
        if abs(new_cx - cx) < 1e-3 and abs(new_cy - cy) < 1e-3:
            break
        cx, cy = new_cx, new_cy
    return cx, cy


class EclipseCircleDetector:
    def __init__(self,
                 nominal_radius_full: Optional[float] = None,
                 downscale_factor: float = 0.25):
        """
        :param nominal_radius_full: Optional expected celestial limb radius in pixels. If None,
                                     the radius is dynamically measured from the image dimensions
                                     and Hough circle candidates.
        :param downscale_factor: Scale factor used for fast initial candidate detection.
        """
        self.nominal_radius_full = nominal_radius_full
        self.downscale_factor = downscale_factor

    def _prepare_downscaled(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert image to downscaled 8-bit grayscale for initial candidate detection."""
        h, w = img.shape[:2]
        new_w = int(round(w * self.downscale_factor))
        new_h = int(round(h * self.downscale_factor))
        small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if small.ndim == 3:
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        else:
            gray = small

        if gray.dtype == np.uint16:
            gray8 = (gray / 256).astype(np.uint8)
        elif gray.dtype != np.uint8:
            gray8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            gray8 = gray

        return small, gray8

    def _detect_crescent_cusps(self, gray: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Detects crescent cusps (tips) and computes the circle using the chord between cusps.
        Strictly targets true emerging crescents and ignores flare arches or small prominences.
        """
        p99 = np.percentile(gray, 99.5)
        thresh_val = max(p99 * 0.80, 48000)
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        thresh = thresh.astype(np.uint8)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea).squeeze()
        if len(c) < 50:
            return None

        area = cv2.contourArea(c)
        nom_area = np.pi * (self.nominal_radius_full ** 2)
        # Must be an actual crescent: significant area (> 10000) and not full disk (< 0.70 * nom_area)
        if area > 0.70 * nom_area or area < 10000:
            return None

        # Find the two cusps (points furthest apart along the crescent contour)
        sub = c[::max(1, len(c) // 120)]
        dists = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
        i, j = np.unravel_index(np.argmax(dists), dists.shape)
        p1 = sub[i].astype(float)
        p2 = sub[j].astype(float)

        chord_len = float(np.linalg.norm(p1 - p2))
        R = self.nominal_radius_full

        # Chord must span a significant fraction of the celestial circle (at least 600 px)
        if chord_len < 600 or chord_len > 2 * R + 20:
            return None

        mid_chord = (p1 + p2) / 2.0
        d_center = np.sqrt(max(0.0, R**2 - min((chord_len / 2.0)**2, R**2)))

        chord_vec = p2 - p1
        n1 = np.array([-chord_vec[1], chord_vec[0]], dtype=float)
        n1 /= np.linalg.norm(n1)

        c1 = mid_chord + d_center * n1
        c2 = mid_chord - d_center * n1

        # The true Moon center lies in the dark interior (not inside the bright flare)
        h, w = gray.shape
        c1_clipped = (np.clip(c1[0], 0, w - 1), np.clip(c1[1], 0, h - 1))
        c2_clipped = (np.clip(c2[0], 0, w - 1), np.clip(c2[1], 0, h - 1))

        v1 = cv2.remap(gray.astype(np.float32), np.array([[c1_clipped[0]]], dtype=np.float32),
                       np.array([[c1_clipped[1]]], dtype=np.float32), cv2.INTER_LINEAR)[0, 0]
        v2 = cv2.remap(gray.astype(np.float32), np.array([[c2_clipped[0]]], dtype=np.float32),
                       np.array([[c2_clipped[1]]], dtype=np.float32), cv2.INTER_LINEAR)[0, 0]

        if v1 < 0.65 * v2:
            return float(c1[0]), float(c1[1]), float(R)
        elif v2 < 0.65 * v1:
            return float(c2[0]), float(c2[1]), float(R)
        return None

    def refine_totality_balanced(self,
                                 img_gray: np.ndarray,
                                 init_cx: float,
                                 init_cy: float,
                                 nominal_r: Optional[float] = None,
                                 init_r: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Sub-pixel refinement for Totality & Contacts using balanced 360-degree radial profiling.
        Filters out flare-corrupted rays per quadrant and fits circle center and radius.
        If nominal_r is provided, constrains radius to nominal_r; otherwise fits radius algebraically.
        """
        h, w = img_gray.shape
        gray_f = img_gray.astype(np.float32)

        num_rays = 120
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        dr_range = np.linspace(-35.0, 35.0, 71)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        search_r = nominal_r if (nominal_r is not None and nominal_r > 0) else (init_r if (init_r is not None and init_r > 0) else float(min(h, w) * 0.15))
        best_cx, best_cy, best_r = init_cx, init_cy, search_r

        for iteration in range(3):
            quadrant_points: Dict[int, List[Tuple[float, float, float]]] = {0: [], 1: [], 2: [], 3: []}

            for i, a in enumerate(angles):
                rs = best_r + dr_range
                xs = best_cx + rs * cos_a[i]
                ys = best_cy + rs * sin_a[i]

                if np.any(xs < 0) or np.any(xs >= w) or np.any(ys < 0) or np.any(ys >= h):
                    continue

                profile = cv2.remap(gray_f, xs.astype(np.float32)[None, :], ys.astype(np.float32)[None, :],
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).ravel()

                # In totality, intensity increases radially (dark moon to bright corona)
                grad = np.gradient(profile)
                max_idx = int(np.argmax(grad))
                max_val = grad[max_idx]

                p_min, p_max = float(np.min(profile)), float(np.max(profile))
                if (p_max - p_min) > 80 and max_val > 100:
                    if 0 < max_idx < len(grad) - 1:
                        al = grad[max_idx - 1]
                        be = grad[max_idx]
                        ga = grad[max_idx + 1]
                        denom = 2 * (al - 2 * be + ga)
                        offset = (al - ga) / denom if abs(denom) > 1e-5 else 0.0
                    else:
                        offset = 0.0
                    refined_dist = rs[max_idx] + offset
                    px = best_cx + refined_dist * cos_a[i]
                    py = best_cy + refined_dist * sin_a[i]

                    # 4 Quadrants
                    quad = int((a % (2 * np.pi)) / (np.pi / 2))
                    quadrant_points[quad].append((px, py, refined_dist))

            # Filter flare outliers within each quadrant
            valid_pts = []
            for quad, qpts in quadrant_points.items():
                if len(qpts) >= 3:
                    q_radii = np.array([p[2] for p in qpts])
                    med_qr = np.median(q_radii)
                    for p in qpts:
                        if abs(p[2] - med_qr) <= 5.0:
                            valid_pts.append((p[0], p[1]))

            if len(valid_pts) < 16:
                break

            pts_arr = np.array(valid_pts)
            if nominal_r is not None and nominal_r > 0:
                best_cx, best_cy = fit_circle_fixed_r(pts_arr, nominal_r, best_cx, best_cy)
                best_r = nominal_r
            else:
                cx_fit, cy_fit, r_fit, rmse = fit_circle_algebraic(pts_arr)
                if r_fit > 0 and rmse < 10.0:
                    best_cx, best_cy, best_r = cx_fit, cy_fit, r_fit

        return float(best_cx), float(best_cy), float(best_r)

    def refine_filtered_sun(self,
                            img_gray: np.ndarray,
                            init_cx: float,
                            init_cy: float,
                            init_r: float) -> Tuple[float, float, float]:
        """Sub-pixel refinement for filtered solar disk (PreTotality/PostTotality)."""
        h, w = img_gray.shape
        gray_f = img_gray.astype(np.float32)

        num_rays = 120
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        dr_range = np.linspace(-25.0, 25.0, 51)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        best_cx, best_cy, best_r = init_cx, init_cy, init_r

        for iteration in range(2):
            limb_points = []
            for i, a in enumerate(angles):
                rs = best_r + dr_range
                xs = best_cx + rs * cos_a[i]
                ys = best_cy + rs * sin_a[i]

                if np.any(xs < 0) or np.any(xs >= w) or np.any(ys < 0) or np.any(ys >= h):
                    continue

                profile = cv2.remap(gray_f, xs.astype(np.float32)[None, :], ys.astype(np.float32)[None, :],
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).ravel()

                grad = -np.gradient(profile)
                max_idx = int(np.argmax(grad))
                if grad[max_idx] > 500:
                    refined_dist = rs[max_idx]
                    px = best_cx + refined_dist * cos_a[i]
                    py = best_cy + refined_dist * sin_a[i]
                    limb_points.append((px, py))

            if len(limb_points) < 20:
                break

            pts_arr = np.array(limb_points)
            cx_fit, cy_fit, r_fit, rmse = fit_circle_algebraic(pts_arr)
            if r_fit > 0 and rmse < 8.0:
                best_cx, best_cy, best_r = cx_fit, cy_fit, r_fit

        return float(best_cx), float(best_cy), float(best_r)

    def detect(self, img: np.ndarray) -> Tuple[float, float, float]:
        """
        Main detection entrypoint:
        1. Checks for filtered full sun (bright interior > 35000 and 5x corner).
        2. Detects initial candidate via downscaled Hough circle.
        3. Checks quadrant light asymmetry:
           - If highly asymmetric (ratio >= 5.0): checks for emerging crescent cusps.
           - Otherwise: applies flare-immune balanced radial limb refinement.
        Returns: (cx, cy, r)
        """
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        h, w = gray.shape

        # 1. Downscaled Hough candidate
        _, gray8 = self._prepare_downscaled(img)
        min_dim = min(h, w)
        if self.nominal_radius_full is not None and self.nominal_radius_full > 0:
            min_r = int(self.nominal_radius_full * self.downscale_factor * 0.85)
            max_r = int(self.nominal_radius_full * self.downscale_factor * 1.15)
        else:
            # Dynamic candidate search: radius between 4% and 45% of image dimension
            min_r = max(10, int(min_dim * self.downscale_factor * 0.04))
            max_r = int(min_dim * self.downscale_factor * 0.45)

        edges = cv2.Canny(gray8, 20, 60)
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=int(min_r * 1.5),
            param1=50,
            param2=22,
            minRadius=min_r,
            maxRadius=max_r
        )

        if circles is not None and len(circles[0]) > 0:
            c = circles[0][0]
            init_cx = float(c[0] / self.downscale_factor)
            init_cy = float(c[1] / self.downscale_factor)
            init_r = float(c[2] / self.downscale_factor)
        else:
            fallback_r = self.nominal_radius_full if (self.nominal_radius_full and self.nominal_radius_full > 0) else float(min_dim * 0.15)
            init_cx, init_cy, init_r = float(w / 2.0), float(h / 2.0), fallback_r

        # Check for filtered solar disk
        sample_x = np.clip(int(round(init_cx)), 0, w - 1)
        sample_y = np.clip(int(round(init_cy)), 0, h - 1)
        center_intensity = float(gray[sample_y, sample_x])
        corner_intensity = float(np.median([gray[:30, :30], gray[:30, -30:], gray[-30:, :30], gray[-30:, -30:]]))

        if center_intensity > 35000 and center_intensity > 5.0 * (corner_intensity + 1.0):
            # Filtered solar disk
            return self.refine_filtered_sun(gray, init_cx, init_cy, init_r)

        # 2. Check quadrant light asymmetry around the celestial center
        angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        r_test = (self.nominal_radius_full if self.nominal_radius_full else init_r) * 1.10
        xs = np.clip(init_cx + r_test * np.cos(angles), 0, w - 1).astype(int)
        ys = np.clip(init_cy + r_test * np.sin(angles), 0, h - 1).astype(int)
        vals = gray[ys, xs].astype(float)

        q0 = np.median(vals[np.abs(np.degrees(angles) - 0) <= 45])
        q1 = np.median(vals[np.abs(np.degrees(angles) - 90) <= 45])
        q2 = np.median(vals[np.abs(np.degrees(angles) - 180) <= 45])
        q3 = np.median(vals[np.abs(np.degrees(angles) - 270) <= 45])
        q_vals = [q0, q1, q2, q3]
        ratio = max(q_vals) / (min(q_vals) + 1.0)

        # 3. If highly asymmetric (emerging crescent), try crescent cusps
        if ratio >= 5.0 and self.nominal_radius_full:
            crescent_res = self._detect_crescent_cusps(gray)
            if crescent_res is not None:
                return crescent_res

        # 4. Standard Totality & Contacts: Full-circle balanced limb refinement
        return self.refine_totality_balanced(
            gray,
            init_cx,
            init_cy,
            nominal_r=self.nominal_radius_full,
            init_r=init_r
        )
