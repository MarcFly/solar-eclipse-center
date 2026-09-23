"""
Affine transformation for solar eclipse photography.
Performs single-pass combined sub-pixel centering, uniform solar radius scaling (R -> R_target),
and square cropping (S_crop x S_crop) using high-order Lanczos interpolation while strictly
preserving native 16-bit uint16 depth and photometric accuracy.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class TransformationResult:
    orig_cx: float
    orig_cy: float
    orig_r: float
    target_r: float
    scale: float
    crop_size: int
    target_cx: float
    target_cy: float
    rotation_deg: float
    img_transformed: np.ndarray


class EclipseAffineTransformer:
    def __init__(self,
                 interpolation: int = cv2.INTER_LANCZOS4,
                 fill_background: bool = True):
        """
        :param interpolation: Interpolation algorithm (default: cv2.INTER_LANCZOS4 for sub-pixel fidelity).
        :param fill_background: If True, uses the median sky background level from image corners to fill borders; if False, fills with 0.
        """
        self.interpolation = interpolation
        self.fill_background = fill_background

    def transform(self,
                  img: np.ndarray,
                  cx: float,
                  cy: float,
                  r: float,
                  target_r: Optional[float] = None,
                  crop_size: Optional[int] = None,
                  rotation_deg: float = 0.0,
                  dx: float = 0.0,
                  dy: float = 0.0) -> TransformationResult:
        """
        Applies a single-pass composite affine transformation that simultaneously:
        1. Translates the solar/lunar center (cx, cy) to the square crop center (crop_size/2, crop_size/2).
        2. Resizes the solar radius by scale s = target_r / r (if target_r is None, s = 1.0).
        3. Rotates by angle rotation_deg around the solar center.
        4. Applies fine sub-pixel solar offset (dx, dy).
        5. Crops the output canvas to (crop_size, crop_size).

        All transformations are composed into a single affine matrix, guaranteeing
        zero generational loss and preserving native 16-bit uint16 depth.

        :param img: Full resolution input image (uint16 or uint8).
        :param cx: Detected circle center X in source image pixels.
        :param cy: Detected circle center Y in source image pixels.
        :param r: Detected circle radius in source image pixels.
        :param target_r: Desired standardized solar/lunar radius in output pixels (default: None, preserves detected radius r).
        :param crop_size: Dimension of the output square crop in pixels (default: None, uses min(h, w)).
        :param rotation_deg: Rotation angle in degrees (clockwise positive).
        :param dx: Fine sub-pixel horizontal solar offset (default: 0.0).
        :param dy: Fine sub-pixel vertical solar offset (default: 0.0).
        :return: TransformationResult containing the transformed 16-bit array and metadata.
        """
        h, w = img.shape[:2]
        effective_target_r = target_r if (target_r is not None and target_r > 0) else r
        effective_crop_size = crop_size if (crop_size is not None and crop_size > 0) else min(h, w)
        s = float(effective_target_r / r) if r > 0 else 1.0
        target_cx = effective_crop_size / 2.0
        target_cy = effective_crop_size / 2.0

        # Composite affine matrix:
        # [x']   [s*cos(th) -s*sin(th)] [x - cx]   [target_cx + dx]
        # [y'] = [s*sin(th)  s*cos(th)] [y - cy] + [target_cy + dy]
        # Let alpha = s * cos(th), beta = s * sin(th):
        # x' = alpha * x - beta * y + (target_cx + dx - alpha * cx + beta * cy)
        # y' = beta * x + alpha * y + (target_cy + dy - beta * cx - alpha * cy)
        rad = np.deg2rad(rotation_deg)
        alpha = s * np.cos(rad)
        beta = s * np.sin(rad)

        tx = target_cx + dx - alpha * cx + beta * cy
        ty = target_cy + dy - beta * cx - alpha * cy

        M = np.float32([
            [alpha, -beta, tx],
            [beta,   alpha, ty]
        ])

        # Border fill value
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

        # Single-pass 16-bit affine warp
        transformed = cv2.warpAffine(
            img,
            M,
            (effective_crop_size, effective_crop_size),
            flags=self.interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val
        )

        return TransformationResult(
            orig_cx=cx,
            orig_cy=cy,
            orig_r=r,
            target_r=effective_target_r,
            scale=s,
            crop_size=effective_crop_size,
            target_cx=target_cx,
            target_cy=target_cy,
            rotation_deg=rotation_deg,
            img_transformed=transformed
        )

    def create_visual_overlay(self,
                              res: TransformationResult,
                              filename: str = "",
                              downscale_factor: float = 0.25) -> np.ndarray:
        """
        Creates an annotated visual verification overlay for the transformed square crop:
        - Red crosshair right at the output center (crop_size/2, crop_size/2)
        - Full-canvas horizontal and vertical axis guide lines
        - Green circle of radius target_r around output center
        - Concentric alignment rings at 1.5x and 2.0x target radius
        - Information banner reporting source center, scale factor, target radius, and crop dimensions
        """
        img = res.img_transformed
        h, w = img.shape[:2]
        sw = int(round(w * downscale_factor))
        sh = int(round(h * downscale_factor))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)

        # Log stretch for clear visual inspection
        sf = small.astype(np.float32)
        slog = np.log1p(sf)
        slog = ((slog - slog.min()) / (slog.max() - slog.min()) * 255).astype(np.uint8)
        vis = cv2.cvtColor(slog, cv2.COLOR_RGB2BGR) if small.ndim == 3 else cv2.cvtColor(slog, cv2.COLOR_GRAY2BGR)

        tcx = int(round(res.target_cx * downscale_factor))
        tcy = int(round(res.target_cy * downscale_factor))
        sr = int(round(res.target_r * downscale_factor))

        # 1. Full-canvas coordinate axis lines intersecting at center
        cv2.line(vis, (0, tcy), (sw - 1, tcy), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(vis, (tcx, 0), (tcx, sh - 1), (0, 255, 255), 1, cv2.LINE_AA)

        # 2. Celestial limb circle centered at (crop_size/2, crop_size/2) in green
        cv2.circle(vis, (tcx, tcy), sr, (0, 255, 0), 2, cv2.LINE_AA)

        # 3. Canvas center crosshair in red
        cv2.drawMarker(vis, (tcx, tcy), (0, 0, 255), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)

        # 4. Concentric alignment rings at 1.5x and 2.0x radius in faint yellow
        cv2.circle(vis, (tcx, tcy), int(round(sr * 1.5)), (255, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(vis, (tcx, tcy), int(round(sr * 2.0)), (255, 255, 0), 1, cv2.LINE_AA)

        # 5. Informative text banner
        title = f"{filename}: Standardized Square Crop {res.crop_size}x{res.crop_size} px"
        sub = f"Sun R: {res.orig_r:.1f}px -> {res.target_r:.1f}px (scale={res.scale:.4f}) | Center: ({res.target_cx:.1f}, {res.target_cy:.1f})"
        cv2.putText(vis, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, sub, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2, cv2.LINE_AA)

        return vis

