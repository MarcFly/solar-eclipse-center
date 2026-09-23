import os
import tifffile
import numpy as np
import cv2
from detector import EclipseCircleDetector

det = EclipseCircleDetector()
for idx in [75, 90, 105]:
    f = sorted(os.listdir("../EclipseTry2/3_Totality"))[idx]
    img = tifffile.imread(f"../EclipseTry2/3_Totality/{f}")
    cx, cy, r = det.detect(img)
    print(f"\n{f}: Moon center=({cx:.1f}, {cy:.1f}), r={r:.1f}")
    # Inspect right prominence region: cx + r, cy
    bx, by = int(round(cx + r)), int(round(cy))
    crop_rt = img[by-50:by+50, bx-50:bx+50]
    prom_rt = np.maximum(0, crop_rt[:, :, 0].astype(np.float32) - 0.5*(crop_rt[:, :, 1].astype(np.float32) + crop_rt[:, :, 2].astype(np.float32)))
    print(f"  Right crop prom max={prom_rt.max():.0f}, mean={prom_rt.mean():.0f}")
    
    # Inspect top prominence region: cx, cy - r
    bx_t, by_t = int(round(cx)), int(round(cy - r))
    crop_t = img[by_t-50:by_t+50, bx_t-50:bx_t+50]
    prom_t = np.maximum(0, crop_t[:, :, 0].astype(np.float32) - 0.5*(crop_t[:, :, 1].astype(np.float32) + crop_t[:, :, 2].astype(np.float32)))
    print(f"  Top crop prom max={prom_t.max():.0f}, mean={prom_t.mean():.0f}")

