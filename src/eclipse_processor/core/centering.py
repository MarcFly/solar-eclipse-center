"""
Sub-pixel circle centering for solar eclipse photography.
Translates detected solar/lunar circle center (cx, cy) to the geometric center
of the canvas (W/2, H/2) via sub-pixel affine translation while strictly preserving
native 16-bit uint16 depth and photometric values.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class CenteringResult:
    orig_cx: float
    orig_cy: float
    target_cx: float
    target_cy: float
    dx: float
    dy: float
    r: float
    img_centered: np.ndarray


class EclipseCenterer:
    def __init__(self,
                 interpolation: int = cv2.INTER_LANCZOS4,
                 fill_background: bool = True):
        """
        :param interpolation: Interpolation algorithm (default: cv2.INTER_LANCZOS4).
        :param fill_background: If True, uses the median corner sky background level to fill newly exposed borders; if False, fills with 0.
        """
        self.interpolation = interpolation
        self.fill_background = fill_background

    def center(self, img: np.ndarray, cx: float, cy: float, r: float) -> CenteringResult:
        """
        Translates the image so that (cx, cy) is positioned at the geometric canvas center (W/2, H/2).
        :param img: Full resolution 16-bit or 8-bit image array (H, W, C) or (H, W).
        :param cx: Detected circle center X in full resolution pixels.
        :param cy: Detected circle center Y in full resolution pixels.
        :param r: Detected circle radius in full resolution pixels.
        :return: CenteringResult containing centered image and translation metrics.
        """
        h, w = img.shape[:2]
        target_cx = w / 2.0
        target_cy = h / 2.0

        dx = target_cx - cx
        dy = target_cy - cy

        # Affine translation matrix
        M = np.float32([
            [1.0, 0.0, dx],
            [0.0, 1.0, dy]
        ])

        # Determine border fill value
        if self.fill_background:
            patch = min(40, h // 20, w // 20)
            if img.ndim == 3:
                corners = np.concatenate([
                    img[:patch, :patch, :].reshape(-1, img.shape[2]),
                    img[:patch, -patch:, :].reshape(-1, img.shape[2]),
                    img[-patch:, :patch, :].reshape(-1, img.shape[2]),
                    img[-patch:, -patch:, :].reshape(-1, img.shape[2])
                ])
                border_val = [float(np.median(corners[:, c])) for c in range(img.shape[2])]
            else:
                corners = np.concatenate([
                    img[:patch, :patch].ravel(),
                    img[:patch, -patch:].ravel(),
                    img[-patch:, :patch].ravel(),
                    img[-patch:, -patch:].ravel()
                ])
                border_val = float(np.median(corners))
        else:
            border_val = 0

        warped = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=self.interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val
        )

        return CenteringResult(
            orig_cx=cx,
            orig_cy=cy,
            target_cx=target_cx,
            target_cy=target_cy,
            dx=dx,
            dy=dy,
            r=r,
            img_centered=warped
        )

    def create_visual_overlay(self,
                              res: CenteringResult,
                              filename: str = "",
                              downscale_factor: float = 0.25) -> np.ndarray:
        """
        Creates a visual verification overlay showing:
        - Red crosshair right at the canvas center (W/2, H/2)
        - Full-canvas horizontal and vertical crosshair guide lines passing through (W/2, H/2)
        - Green circle of radius R around (W/2, H/2)
        - Concentric alignment rings at 1.5x and 2.0x radius in faint yellow
        - Informative banner with (cx, cy) -> (dx, dy) -> (target_cx, target_cy)
        """
        img = res.img_centered
        h, w = img.shape[:2]
        sw = int(round(w * downscale_factor))
        sh = int(round(h * downscale_factor))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)

        # Log stretch for clear visual inspection of corona and limb
        sf = small.astype(np.float32)
        slog = np.log1p(sf)
        slog = ((slog - slog.min()) / (slog.max() - slog.min()) * 255).astype(np.uint8)
        vis = cv2.cvtColor(slog, cv2.COLOR_RGB2BGR) if small.ndim == 3 else cv2.cvtColor(slog, cv2.COLOR_GRAY2BGR)

        tcx = int(round(res.target_cx * downscale_factor))
        tcy = int(round(res.target_cy * downscale_factor))
        sr = int(round(res.r * downscale_factor))

        # 1. Full-canvas coordinate axis lines intersecting at center (thin yellow line)
        cv2.line(vis, (0, tcy), (sw - 1, tcy), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(vis, (tcx, 0), (tcx, sh - 1), (0, 255, 255), 1, cv2.LINE_AA)

        # 2. Celestial limb circle centered at (W/2, H/2) in green
        cv2.circle(vis, (tcx, tcy), sr, (0, 255, 0), 2, cv2.LINE_AA)

        # 3. Canvas center crosshair in red
        cv2.drawMarker(vis, (tcx, tcy), (0, 0, 255), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)

        # 4. Concentric alignment rings at 1.5x and 2.0x radius in faint cyan
        cv2.circle(vis, (tcx, tcy), int(round(sr * 1.5)), (255, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(vis, (tcx, tcy), int(round(sr * 2.0)), (255, 255, 0), 1, cv2.LINE_AA)

        # 5. Informative text banner
        title = f"{filename}: Centered at Canvas Center ({res.target_cx:.1f}, {res.target_cy:.1f})"
        sub = f"Original: ({res.orig_cx:.1f}, {res.orig_cy:.1f}) | Shift: dx={res.dx:+.1f} px, dy={res.dy:+.1f} px | Limb R={res.r:.1f} px"
        cv2.putText(vis, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, sub, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2, cv2.LINE_AA)

        return vis

