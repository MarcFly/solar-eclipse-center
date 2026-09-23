import os
import glob
import cv2
import numpy as np
import tifffile
from detector import EclipseCircleDetector

def track():
    det = EclipseCircleDetector()
    files = sorted(glob.glob("../EclipseTry2/3_Totality/*.tif"))
    
    # Let's inspect 12 frames across totality
    indices = np.linspace(0, len(files)-1, 12, dtype=int)
    
    results = []
    for idx in indices:
        f = files[idx]
        fname = os.path.basename(f)
        img = tifffile.imread(f)
        cx, cy, r = det.detect(img)
        
        # We want to sample three prominence patches in polar coordinates:
        # Left: 165° to 195°
        # Top: 250° to 280°
        # Right: 345° to 15° (which is -15° to +15°)
        
        # Radii: 1.005*r to 1.12*r
        n_rad = 30
        rads = np.linspace(r * 1.005, r * 1.12, n_rad)
        
        def get_sector_profile(ang_min_deg, ang_max_deg, n_ang=120):
            if ang_min_deg > ang_max_deg: # wrap around 360
                angs = np.linspace(np.deg2rad(ang_min_deg), np.deg2rad(ang_max_deg + 360), n_ang)
            else:
                angs = np.linspace(np.deg2rad(ang_min_deg), np.deg2rad(ang_max_deg), n_ang)
            grid_th, grid_r = np.meshgrid(angs, rads)
            xs = (cx + grid_r * np.cos(grid_th)).astype(np.float32)
            ys = (cy + grid_r * np.sin(grid_th)).astype(np.float32)
            
            rc = cv2.remap(img[:, :, 0].astype(np.float32), xs, ys, interpolation=cv2.INTER_LINEAR)
            gc = cv2.remap(img[:, :, 1].astype(np.float32), xs, ys, interpolation=cv2.INTER_LINEAR)
            bc = cv2.remap(img[:, :, 2].astype(np.float32), xs, ys, interpolation=cv2.INTER_LINEAR)
            
            prom = np.maximum(0, rc - 0.5 * (gc + bc))
            # 1D angular profile (sum over radial band)
            prof = np.mean(prom, axis=0)
            return angs % (2*np.pi), prof
            
        angs_l, prof_l = get_sector_profile(165, 195)
        angs_t, prof_t = get_sector_profile(250, 280)
        angs_r, prof_r = get_sector_profile(345, 15)
        
        # Find peak angle for each using centroid around max
        def find_peak_angle(angs, prof):
            max_val = np.max(prof)
            if max_val < 500: # too faint or washed out
                return None, max_val
            # centroid of points >= 0.7 * max_val
            mask = prof >= 0.7 * max_val
            # handle circular wrapping if needed
            unwrapped_angs = np.unwrap(angs)
            peak_ang = np.rad2deg(np.average(unwrapped_angs[mask], weights=prof[mask])) % 360.0
            return peak_ang, max_val
            
        ang_l, val_l = find_peak_angle(angs_l, prof_l)
        ang_t, val_t = find_peak_angle(angs_t, prof_t)
        ang_r, val_r = find_peak_angle(angs_r, prof_r)
        
        results.append({
            "idx": idx,
            "fname": fname,
            "cx": cx, "cy": cy,
            "left": (ang_l, val_l),
            "top": (ang_t, val_t),
            "right": (ang_r, val_r)
        })
        
        l_str = f"{ang_l:5.1f}° ({val_l:5.0f})" if ang_l is not None else f"  N/A ({val_l:5.0f})"
        t_str = f"{ang_t:5.1f}° ({val_t:5.0f})" if ang_t is not None else f"  N/A ({val_t:5.0f})"
        r_str = f"{ang_r:5.1f}° ({val_r:5.0f})" if ang_r is not None else f"  N/A ({val_r:5.0f})"
        print(f"[{idx:3d}] {fname} | Left: {l_str} | Top: {t_str} | Right: {r_str}")

if __name__ == "__main__":
    track()

