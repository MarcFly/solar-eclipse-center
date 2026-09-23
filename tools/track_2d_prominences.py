import os
import glob
import cv2
import numpy as np
import tifffile
from detector import EclipseCircleDetector

def track_2d_prominences():
    det = EclipseCircleDetector()
    files = sorted(glob.glob("../EclipseTry2/3_Totality/*.tif"))
    
    # Let's inspect 10 frames
    indices = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 151]
    
    # We will locate the 2D subpixel centroids of:
    # 1. Top Prominence:   y ~ cy - r, x ~ cx
    # 2. Right Prominence: x ~ cx + r, y ~ cy
    # 3. Left Prominence:  x ~ cx - r, y ~ cy
    
    records = []
    
    for idx in indices:
        f = files[idx]
        fname = os.path.basename(f)
        img = tifffile.imread(f)
        cx, cy, r = det.detect(img)
        
        def find_subpixel_prom(bx, by, w=50):
            bx, by = int(round(bx)), int(round(by))
            crop = img[max(0, by-w):min(img.shape[0], by+w), max(0, bx-w):min(img.shape[1], bx+w)]
            if crop.size == 0:
                return None
            prom = np.maximum(0, crop[:, :, 0].astype(np.float32) - 0.5 * (crop[:, :, 1].astype(np.float32) + crop[:, :, 2].astype(np.float32)))
            max_v = np.max(prom)
            if max_v < 1000:
                return None
            # Centroid of top 25% intensities
            mask = prom >= 0.75 * max_v
            ys, xs = np.where(mask)
            weights = prom[ys, xs]
            x_cent = (bx - w) + np.average(xs, weights=weights)
            y_cent = (by - w) + np.average(ys, weights=weights)
            return np.array([x_cent, y_cent], dtype=np.float32)
            
        pt_top = find_subpixel_prom(cx, cy - 1.05 * r)
        pt_rt = find_subpixel_prom(cx + 1.05 * r, cy)
        pt_lt = find_subpixel_prom(cx - 1.05 * r, cy)
        
        records.append({
            "idx": idx, "fname": fname,
            "moon_c": (cx, cy),
            "top": pt_top, "rt": pt_rt, "lt": pt_lt
        })
        
    print("Prominence 2D coordinates:")
    ref = records[0]
    p_top0 = ref["top"]
    p_rt0 = ref["rt"]
    p_lt0 = ref["lt"]
    
    # Vector from Top to Right in reference frame:
    v_tr0 = p_rt0 - p_top0
    dist_tr0 = np.linalg.norm(v_tr0)
    ang_tr0 = np.rad2deg(np.arctan2(v_tr0[1], v_tr0[0]))
    
    print(f"Reference [{ref['idx']}] {ref['fname']}:")
    print(f"  Top: ({p_top0[0]:.2f}, {p_top0[1]:.2f}), Right: ({p_rt0[0]:.2f}, {p_rt0[1]:.2f}), Left: ({p_lt0[0]:.2f}, {p_lt0[1]:.2f})")
    print(f"  Top->Right Vector: length = {dist_tr0:.2f} px, angle = {ang_tr0:.3f}°\n")
    
    print(f"{'Idx':<4} | {'Filename':<15} | {'Top->Right Ang':<16} | {'Delta Rot':<12} | {'Scale Ratio':<12}")
    print("-" * 70)
    
    for r in records:
        if r["top"] is not None and r["rt"] is not None:
            v_tr = r["rt"] - r["top"]
            dist_tr = np.linalg.norm(v_tr)
            ang_tr = np.rad2deg(np.arctan2(v_tr[1], v_tr[0]))
            d_rot = ang_tr - ang_tr0
            s_ratio = dist_tr / dist_tr0
            print(f"{r['idx']:<4} | {r['fname']:<15} | {ang_tr:10.3f}°      | {d_rot:+8.3f}°   | {s_ratio:8.4f}")
        else:
            print(f"{r['idx']:<4} | {r['fname']:<15} | MISSING POINTS")

if __name__ == "__main__":
    track_2d_prominences()

