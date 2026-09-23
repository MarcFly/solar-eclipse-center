import os
import glob
import cv2
import numpy as np
import tifffile
from detector import EclipseCircleDetector

def measure_prominence_geometry():
    det = EclipseCircleDetector()
    files = sorted(glob.glob("../EclipseTry2/3_Totality/*.tif"))
    
    # Let's inspect 10 frames with good prominence visibility
    # (e.g. short to medium exposures)
    # Let's test on frames where prominences are clearly visible
    print(f"Total files: {len(files)}")
    
    for idx in [0, 15, 40, 50, 60, 75, 80, 90, 110, 120, 135, 151]:
        f = files[idx]
        fname = os.path.basename(f)
        img = tifffile.imread(f)
        cx, cy, r = det.detect(img)
        
        # We can extract the limb image around the moon
        # In full image coordinates:
        # Left prominence: x in [cx - 1.15*r, cx - 0.95*r], y in [cy - 0.15*r, cy + 0.15*r]
        # Top prominence:  x in [cx - 0.15*r, cx + 0.15*r], y in [cy - 1.15*r, cy - 0.95*r]
        # Right prominence:x in [cx + 0.95*r, cx + 1.15*r], y in [cy - 0.15*r, cy + 0.15*r]
        
        def find_prom_centroid(bx1, by1, bx2, by2):
            bx1, by1, bx2, by2 = int(round(bx1)), int(round(by1)), int(round(bx2)), int(round(by2))
            crop = img[by1:by2, bx1:bx2]
            if crop.size == 0:
                return None, 0
            # Red excess
            prom = np.maximum(0, crop[:, :, 0].astype(np.float32) - 0.5 * (crop[:, :, 1].astype(np.float32) + crop[:, :, 2].astype(np.float32)))
            max_v = np.max(prom)
            if max_v < 1000:
                return None, max_v
            mask = prom >= 0.6 * max_v
            ys, xs = np.where(mask)
            w = prom[ys, xs]
            px = bx1 + np.average(xs, weights=w)
            py = by1 + np.average(ys, weights=w)
            return (px, py), max_v

        # Left:
        p_left, v_left = find_prom_centroid(cx - 1.15*r, cy - 0.15*r, cx - 0.95*r, cy + 0.15*r)
        # Top:
        p_top, v_top = find_prom_centroid(cx - 0.15*r, cy - 1.15*r, cx + 0.15*r, cy - 0.95*r)
        # Right:
        p_right, v_right = find_prom_centroid(cx + 0.95*r, cy - 0.15*r, cx + 1.15*r, cy + 0.15*r)
        
        top_str = f"({p_top[0]:.1f}, {p_top[1]:.1f})" if p_top else "None"
        rt_str = f"({p_right[0]:.1f}, {p_right[1]:.1f})" if p_right else "None"
        lt_str = f"({p_left[0]:.1f}, {p_left[1]:.1f})" if p_left else "None"
        
        # Calculate distances between prominences if found
        dist_tr = np.linalg.norm(np.array(p_top) - np.array(p_right)) if (p_top and p_right) else 0.0
        angle_tr = np.rad2deg(np.arctan2(p_right[1] - p_top[1], p_right[0] - p_top[0])) if (p_top and p_right) else 0.0
        
        print(f"[{idx:3d}] {fname} | Top: {top_str} | Right: {rt_str} | Dist(T-R): {dist_tr:6.1f} | Angle(T->R): {angle_tr:6.2f}°")

if __name__ == "__main__":
    measure_prominence_geometry()

