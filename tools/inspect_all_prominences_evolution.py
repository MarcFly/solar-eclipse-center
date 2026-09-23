import os
import glob
import cv2
import numpy as np
import tifffile
from detector import EclipseCircleDetector

def inspect_all_prominences_evolution():
    files = sorted(glob.glob("../EclipseTry2/3_Totality/*.tif"))
    det = EclipseCircleDetector()
    
    # Let's inspect 10 frames with good prominence visibility
    # e.g., short to medium exposure frames
    print("Evolution of 3 prominences across totality:")
    print(f"{'Idx':<4} | {'Filename':<15} | {'Left Dist':<10} | {'Top Dist':<10} | {'Right Dist':<10} | {'Angle T-R':<10} | {'Angle L-R':<10}")
    print("-" * 80)
    
    for idx in [0, 10, 25, 45, 60, 75, 90, 110, 125, 140, 151]:
        f = files[idx]
        fname = os.path.basename(f)
        img = tifffile.imread(f)
        cx, cy, r = det.detect(img)
        
        # Red excess
        ha = np.maximum(0, img[:, :, 0].astype(np.float32) - 0.5*(img[:, :, 1].astype(np.float32) + img[:, :, 2].astype(np.float32)))
        
        # Find peak along radial rays in 3 directions:
        def get_ray_peak(deg):
            th = np.deg2rad(deg)
            # sample from 0.98*r to 1.15*r
            rads = np.linspace(r * 0.98, r * 1.18, 100)
            xs = (cx + rads * np.cos(th)).astype(np.float32)
            ys = (cy + rads * np.sin(th)).astype(np.float32)
            vals = cv2.remap(ha, xs, ys, interpolation=cv2.INTER_LINEAR).ravel()
            max_idx = np.argmax(vals)
            peak_r = rads[max_idx]
            peak_val = vals[max_idx]
            peak_x = cx + peak_r * np.cos(th)
            peak_y = cy + peak_r * np.sin(th)
            return np.array([peak_x, peak_y]), peak_r, peak_val

        # Search around Top (~265°), Right (~0°), Left (~180°)
        # Search angular window of +/- 10 degrees to find max
        def find_prom_in_sector(center_deg, span=15):
            degs = np.linspace(center_deg - span, center_deg + span, 31)
            best_pt, best_r, best_v, best_deg = None, 0, -1, 0
            for d in degs:
                pt, rad, v = get_ray_peak(d)
                if v > best_v:
                    best_v = v
                    best_pt = pt
                    best_r = rad
                    best_deg = d
            return best_pt, best_r, best_v, best_deg
            
        pt_lt, r_lt, v_lt, d_lt = find_prom_in_sector(180)
        pt_tp, r_tp, v_tp, d_tp = find_prom_in_sector(265)
        pt_rt, r_rt, v_rt, d_rt = find_prom_in_sector(0)
        
        # Angle between Top and Right
        v_tr = pt_rt - pt_tp
        ang_tr = np.rad2deg(np.arctan2(v_tr[1], v_tr[0]))
        
        # Angle between Left and Right
        v_lr = pt_rt - pt_lt
        ang_lr = np.rad2deg(np.arctan2(v_lr[1], v_lr[0]))
        
        print(f"{idx:<4} | {fname:<15} | {r_lt:8.1f}   | {r_tp:8.1f}   | {r_rt:8.1f}   | {ang_tr:8.2f}°  | {ang_lr:8.2f}°")

if __name__ == "__main__":
    inspect_all_prominences_evolution()

