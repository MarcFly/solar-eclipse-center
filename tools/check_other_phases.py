import os
import glob
import cv2
import numpy as np
import tifffile
from detector import EclipseCircleDetector

def check_other_phases():
    det = EclipseCircleDetector()
    samples = [
        ("../EclipseTry2/1_Totality_Contact1/_DSF5100.tif", "Contact 1"),
        ("../EclipseTry2/2_Totality_Contact2/_DSF5139.tif", "Contact 2"),
        ("../EclipseTry2/3_Totality/_DSF5184.tif", "Totality Start"),
        ("../EclipseTry2/3_Totality/_DSF5335.tif", "Totality End"),
        ("../EclipseTry2/4_TotalityExit/_DSF5398.tif", "Totality Exit"),
    ]
    for fpath, label in samples:
        if not os.path.exists(fpath):
            continue
        img = tifffile.imread(fpath)
        cx, cy, r = det.detect(img)
        print(f"\n{label} ({os.path.basename(fpath)}): cx={cx:.1f}, cy={cy:.1f}, r={r:.1f}")
        # Look for prominences
        # Red excess
        # Top prominence: (cx, cy - r)
        # Right prominence: (cx + r, cy)
        def get_max_prom(bx, by):
            bx, by = int(round(bx)), int(round(by))
            crop = img[max(0, by-40):min(img.shape[0], by+40), max(0, bx-40):min(img.shape[1], bx+40)]
            if crop.size == 0: return 0
            p = np.maximum(0, crop[:, :, 0].astype(np.float32) - 0.5 * (crop[:, :, 1].astype(np.float32) + crop[:, :, 2].astype(np.float32)))
            return np.max(p)
            
        v_top = get_max_prom(cx, cy - r * 1.05)
        v_rt = get_max_prom(cx + r * 1.05, cy)
        v_lt = get_max_prom(cx - r * 1.05, cy)
        print(f"  Max Prominence: Left(9h)={v_lt:.0f}, Top(12h)={v_top:.0f}, Right(3h)={v_rt:.0f}")

if __name__ == "__main__":
    check_other_phases()

