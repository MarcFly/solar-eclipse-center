"""
Multi-Point Prominence Rotation Alignment & Solar Registration.
Detects stable solar prominence anchors (Top ~12h, Right ~3h, and Left ~9h)
to track camera field rotation and lunar parallax, enabling perfect feature-locked
stacking for solar eclipse composite photography.
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import tifffile
from eclipse_processor.core.detector import EclipseCircleDetector


@dataclass
class AnchorPoint:
    name: str
    point: np.ndarray # (x, y) absolute image coordinates
    rel_center: np.ndarray # (dx, dy) relative to circle center
    score: float # NCC confidence score (0 to 1)


@dataclass
class RotationResult:
    filename: str
    rotation_deg: float # Estimated field rotation relative to reference (degrees)
    scale: float # Relative scale ratio
    confidence: float # Overall confidence (0 to 1)
    anchors: Dict[str, AnchorPoint]


class EclipseRotationAligner:
    """
    Multi-point prominence aligner using three solar limb anchors:
    1. Top prominence (~11:30 - 12:00 o'clock, ~265 deg)
    2. Right prominence (~3:00 o'clock, ~0 deg)
    3. Left prominence (~9:00 o'clock, ~180 deg)

    The Top and Right prominences provide an ultra-stable rigid baseline to isolate
    global camera rotation without bias from dynamic eruptive flares (such as the
    bursting left prominence).
    """

    def __init__(self,
                 template_size: int = 32,
                 search_radius: int = 50,
                 min_confidence: float = 0.35):
        self.template_size = template_size
        self.search_radius = search_radius
        self.min_confidence = min_confidence

        self.ref_filename: Optional[str] = None
        self.ref_cx: float = 0.0
        self.ref_cy: float = 0.0
        self.ref_r: float = 0.0

        # Reference template patches (normalized zero-mean unit-variance)
        self.ref_templates: Dict[str, np.ndarray] = {}
        # Reference relative anchor offsets from center (dx, dy)
        self.ref_rel_anchors: Dict[str, np.ndarray] = {}
        # Reference baseline vector (Top -> Right)
        self.ref_angle_tr: float = 0.0
        self.ref_dist_tr: float = 0.0

    @staticmethod
    def extract_halpha(img: np.ndarray) -> np.ndarray:
        """
        Extracts H-alpha chromospheric signal (Red excess over Green/Blue)
        with logarithmic dynamic range compression.
        """
        if img.ndim == 3:
            r = img[:, :, 0].astype(np.float32)
            g = img[:, :, 1].astype(np.float32)
            b = img[:, :, 2].astype(np.float32)
            excess = np.maximum(0.0, r - 0.5 * (g + b))
            return np.log1p(excess)
        return np.log1p(img.astype(np.float32))

    def initialize_reference(self,
                             ref_img: np.ndarray,
                             cx: float,
                             cy: float,
                             r: float,
                             filename: str = "reference") -> Dict[str, AnchorPoint]:
        """
        Initializes the master reference frame, locating the 3 prominence anchors
        and building normalized template fingerprints.
        """
        self.ref_filename = filename
        self.ref_cx = cx
        self.ref_cy = cy
        self.ref_r = r

        ha = self.extract_halpha(ref_img)
        sz = self.template_size

        # Find exact prominence peaks around expected limb sectors:
        # Top: ~12h (x ~ cx, y ~ cy - 1.05*r)
        # Right: ~3h (x ~ cx + 1.05*r, y ~ cy)
        # Left: ~9h (x ~ cx - 1.05*r, y ~ cy)
        pt_top = self._find_peak_in_sector(ha, cx, cy, r, angle_deg=265.0, search_angle_span=15.0)
        pt_rt = self._find_peak_in_sector(ha, cx, cy, r, angle_deg=0.0, search_angle_span=15.0)
        pt_lt = self._find_peak_in_sector(ha, cx, cy, r, angle_deg=180.0, search_angle_span=15.0)

        self.ref_rel_anchors = {
            "top": pt_top - np.array([cx, cy]),
            "right": pt_rt - np.array([cx, cy]),
            "left": pt_lt - np.array([cx, cy]),
        }

        # Build normalized templates
        self.ref_templates = {
            "top": self._extract_normalized_patch(ha, pt_top[0], pt_top[1], sz),
            "right": self._extract_normalized_patch(ha, pt_rt[0], pt_rt[1], sz),
            "left": self._extract_normalized_patch(ha, pt_lt[0], pt_lt[1], sz),
        }

        # Reference baseline vector (Top -> Right)
        v_tr = pt_rt - pt_top
        self.ref_dist_tr = float(np.linalg.norm(v_tr))
        self.ref_angle_tr = float(np.rad2deg(np.arctan2(v_tr[1], v_tr[0])))

        anchors = {
            "top": AnchorPoint("top", pt_top, self.ref_rel_anchors["top"], 1.0),
            "right": AnchorPoint("right", pt_rt, self.ref_rel_anchors["right"], 1.0),
            "left": AnchorPoint("left", pt_lt, self.ref_rel_anchors["left"], 1.0),
        }
        return anchors

    def _find_peak_in_sector(self,
                             ha: np.ndarray,
                             cx: float,
                             cy: float,
                             r: float,
                             angle_deg: float,
                             search_angle_span: float = 15.0) -> np.ndarray:
        """
        Finds the prominent limb peak within an angular sector and radial band r in [0.98*r, 1.15*r].
        """
        angles = np.linspace(np.deg2rad(angle_deg - search_angle_span),
                             np.deg2rad(angle_deg + search_angle_span), 60)
        radii = np.linspace(r * 0.99, r * 1.12, 30)
        grid_th, grid_r = np.meshgrid(angles, radii)
        xs = (cx + grid_r * np.cos(grid_th)).astype(np.float32)
        ys = (cy + grid_r * np.sin(grid_th)).astype(np.float32)

        vals = cv2.remap(ha, xs, ys, interpolation=cv2.INTER_LINEAR)
        # Find global peak
        _, max_v, _, max_loc = cv2.minMaxLoc(vals)
        opt_th = angles[max_loc[0]]
        opt_r = radii[max_loc[1]]

        px = cx + opt_r * np.cos(opt_th)
        py = cy + opt_r * np.sin(opt_th)
        return np.array([px, py], dtype=np.float32)

    def _extract_normalized_patch(self,
                                  ha: np.ndarray,
                                  x: float,
                                  y: float,
                                  sz: int) -> np.ndarray:
        x1, y1 = int(round(x - sz)), int(round(y - sz))
        x2, y2 = int(round(x + sz)), int(round(y + sz))
        patch = ha[y1:y2, x1:x2]
        std = float(np.std(patch))
        return (patch - np.mean(patch)) / (std if std > 1e-5 else 1.0)

    @staticmethod
    def _subpixel_peak(res: np.ndarray, max_loc: Tuple[int, int]) -> Tuple[float, float]:
        x, y = max_loc
        h, w = res.shape
        if 1 <= x < w - 1 and 1 <= y < h - 1:
            patch = res[y-1:y+2, x-1:x+2]
            denom_x = 2.0 * (2.0 * patch[1, 1] - patch[1, 0] - patch[1, 2])
            denom_y = 2.0 * (2.0 * patch[1, 1] - patch[0, 1] - patch[2, 1])
            dx = (patch[1, 2] - patch[1, 0]) / denom_x if abs(denom_x) > 1e-6 else 0.0
            dy = (patch[2, 1] - patch[0, 1]) / denom_y if abs(denom_y) > 1e-6 else 0.0
            return float(x + np.clip(dx, -0.5, 0.5)), float(y + np.clip(dy, -0.5, 0.5))
        return float(x), float(y)

    def _match_anchor(self,
                      ha: np.ndarray,
                      pred_x: float,
                      pred_y: float,
                      tpl: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        sz = tpl.shape[0] // 2
        rad = self.search_radius
        x1 = max(0, int(round(pred_x - rad - sz)))
        y1 = max(0, int(round(pred_y - rad - sz)))
        x2 = min(ha.shape[1], int(round(pred_x + rad + sz)))
        y2 = min(ha.shape[0], int(round(pred_y + rad + sz)))

        sa = ha[y1:y2, x1:x2]
        std = float(np.std(sa))
        if std < 1e-5 or sa.shape[0] < tpl.shape[0] or sa.shape[1] < tpl.shape[1]:
            return None, 0.0

        sa_norm = (sa - np.mean(sa)) / std
        res = cv2.matchTemplate(sa_norm.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_l = cv2.minMaxLoc(res)

        if max_v < self.min_confidence:
            return None, float(max_v)

        sub_x, sub_y = self._subpixel_peak(res, max_l)
        px = x1 + sub_x + sz
        py = y1 + sub_y + sz
        return np.array([px, py], dtype=np.float32), float(max_v)

    def estimate_frame(self,
                       img: np.ndarray,
                       cx: float,
                       cy: float,
                       r: float,
                       filename: str = "") -> RotationResult:
        """
        Estimates rotation angle and solar scale ratio for a single frame
        relative to the reference frame.
        """
        ha = self.extract_halpha(img)
        s_rad = r / self.ref_r if self.ref_r > 0 else 1.0

        pred_top = np.array([cx, cy]) + self.ref_rel_anchors["top"] * s_rad
        pred_rt = np.array([cx, cy]) + self.ref_rel_anchors["right"] * s_rad
        pred_lt = np.array([cx, cy]) + self.ref_rel_anchors["left"] * s_rad

        pt_top, sc_top = self._match_anchor(ha, pred_top[0], pred_top[1], self.ref_templates["top"])
        pt_rt, sc_rt = self._match_anchor(ha, pred_rt[0], pred_rt[1], self.ref_templates["right"])
        pt_lt, sc_lt = self._match_anchor(ha, pred_lt[0], pred_lt[1], self.ref_templates["left"])

        anchors: Dict[str, AnchorPoint] = {}
        if pt_top is not None:
            anchors["top"] = AnchorPoint("top", pt_top, pt_top - np.array([cx, cy]), sc_top)
        if pt_rt is not None:
            anchors["right"] = AnchorPoint("right", pt_rt, pt_rt - np.array([cx, cy]), sc_rt)
        if pt_lt is not None:
            anchors["left"] = AnchorPoint("left", pt_lt, pt_lt - np.array([cx, cy]), sc_lt)

        # Primary Solution: Top and Right anchors (ultra-stable baseline)
        if "top" in anchors and "right" in anchors:
            v_tr = anchors["right"].point - anchors["top"].point
            dist_tr = float(np.linalg.norm(v_tr))
            ang_tr = float(np.rad2deg(np.arctan2(v_tr[1], v_tr[0])))

            # Correction angle to align target frame to reference frame orientation
            d_rot = self.ref_angle_tr - ang_tr
            scale = dist_tr / self.ref_dist_tr
            conf = min(sc_top, sc_rt)

            return RotationResult(
                filename=filename,
                rotation_deg=d_rot,
                scale=scale,
                confidence=conf,
                anchors=anchors
            )

        # Fallback if Top anchor is occluded/faint: use Left and Right
        if "left" in anchors and "right" in anchors:
            v_lr = anchors["right"].point - anchors["left"].point
            ref_v_lr = self.ref_rel_anchors["right"] - self.ref_rel_anchors["left"]
            ang_lr = float(np.rad2deg(np.arctan2(v_lr[1], v_lr[0])))
            ref_ang_lr = float(np.rad2deg(np.arctan2(ref_v_lr[1], ref_v_lr[0])))
            d_rot = ref_ang_lr - ang_lr
            scale = float(np.linalg.norm(v_lr) / np.linalg.norm(ref_v_lr))
            conf = min(sc_lt, sc_rt) * 0.8
            return RotationResult(filename, d_rot, scale, conf, anchors)

        return RotationResult(filename, 0.0, 1.0, 0.0, anchors)

    def process_sequence(self,
                         file_list: List[str],
                         detector: Optional[EclipseCircleDetector] = None) -> List[RotationResult]:
        """
        Processes a sequence of images:
        1. Reference initialization on the first well-exposed frame.
        2. Anchor tracking across all frames.
        3. Trajectory smoothing (robust linear/low-order polynomial fit) to guarantee
           smooth, physical field rotation across bracketed exposure bursts.
        """
        if detector is None:
            detector = EclipseCircleDetector()

        results: List[RotationResult] = []

        # Find reference frame (first frame with moderate exposure)
        ref_idx = 0
        ref_img = None
        for i, fpath in enumerate(file_list):
            img = tifffile.imread(fpath)
            cx, cy, r = detector.detect(img)
            # moderate exposure: max brightness > 10000, not too many saturated pixels
            if (img > 60000).sum() < 300000 and img.max() > 10000:
                ref_idx = i
                ref_img = img
                self.initialize_reference(img, cx, cy, r, filename=os.path.basename(fpath))
                break

        if ref_img is None:
            ref_img = tifffile.imread(file_list[0])
            cx, cy, r = detector.detect(ref_img)
            self.initialize_reference(ref_img, cx, cy, r, filename=os.path.basename(file_list[0]))

        # Track anchors across all frames
        for fpath in file_list:
            fname = os.path.basename(fpath)
            img = tifffile.imread(fpath)
            cx, cy, r = detector.detect(img)
            res = self.estimate_frame(img, cx, cy, r, filename=fname)
            results.append(res)

        # Smooth rotation trajectory across indices
        valid_indices = [i for i, r in enumerate(results) if r.confidence >= self.min_confidence]

        if len(valid_indices) >= 3:
            indices_arr = np.array(valid_indices, dtype=np.float32)
            angles_arr = np.array([results[i].rotation_deg for i in valid_indices], dtype=np.float32)

            # Robust linear fit to model continuous field rotation
            poly = np.polyfit(indices_arr, angles_arr, deg=1)
            smoothed_angles = np.polyval(poly, np.arange(len(results)))

            for i, res in enumerate(results):
                res.rotation_deg = float(smoothed_angles[i])
        elif len(valid_indices) > 0:
            median_angle = float(np.median([results[i].rotation_deg for i in valid_indices]))
            for res in results:
                if res.confidence < self.min_confidence:
                    res.rotation_deg = median_angle

        return results

    @staticmethod
    def render_anchor_overlay(img: np.ndarray,
                              result: RotationResult,
                              cx: float,
                              cy: float,
                              r: float,
                              out_size: int = 1200) -> np.ndarray:
        """
        Renders a diagnostic visualization showing:
        - Detected prominence anchors (Top, Right, Left)
        - Baseline vectors connecting anchors
        - Lunar center and solar radius circle
        - Estimated rotation angle readout
        """
        h, w = img.shape[:2]
        scale = out_size / max(h, w)
        small = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # 8-bit log visualization
        if small.ndim == 3:
            lum = 0.299 * small[:, :, 0] + 0.587 * small[:, :, 1] + 0.114 * small[:, :, 2]
        else:
            lum = small.astype(np.float32)
        log_v = np.log1p(lum.astype(np.float32))
        norm_v = ((log_v - log_v.min()) / (log_v.max() - log_v.min() + 1e-6) * 255.0).astype(np.uint8)
        vis = cv2.cvtColor(norm_v, cv2.COLOR_GRAY2BGR)

        # Scale coordinates
        scx, scy = int(round(cx * scale)), int(round(cy * scale))
        sr = int(round(r * scale))

        # Circle
        cv2.circle(vis, (scx, scy), sr, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(vis, (scx, scy), (0, 255, 255), cv2.MARKER_CROSS, 16, 1)

        colors = {
            "top": (0, 255, 0),     # Green: Top anchor (~12h)
            "right": (0, 165, 255), # Orange: Right anchor (~3h)
            "left": (255, 0, 128)   # Magenta: Left anchor (~9h)
        }

        pts_dict = {}
        for name, anchor in result.anchors.items():
            px = int(round(anchor.point[0] * scale))
            py = int(round(anchor.point[1] * scale))
            pts_dict[name] = (px, py)
            col = colors.get(name, (255, 255, 255))
            cv2.circle(vis, (px, py), 6, col, 2, cv2.LINE_AA)
            cv2.putText(vis, f"{name.upper()} ({anchor.score:.2f})", (px + 8, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

        # Draw baseline between Top and Right
        if "top" in pts_dict and "right" in pts_dict:
            cv2.line(vis, pts_dict["top"], pts_dict["right"], (0, 255, 0), 1, cv2.LINE_AA)

        # Header readout
        cv2.putText(vis, f"File: {result.filename}", (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"Field Rotation: {result.rotation_deg:+6.3f} deg", (16, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Confidence: {result.confidence:.3f} | Scale: {result.scale:.4f}", (16, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        return vis
