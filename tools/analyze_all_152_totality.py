import os
import glob
import cv2
import numpy as np
import tifffile
from detector import EclipseCircleDetector

def analyze_all_152_totality():
    files = sorted(glob.glob("../EclipseTry2/3_Totality/*.tif"))
    print(f"Total files: {len(files)}")
    det = EclipseCircleDetector()
    
    # We want to measure the orientation using the Top and Right prominences
    # for every single frame in 3_Totality.
    
    records = []
    for idx, f in enumerate(files):
        fname = os.path.basename(f)
        img = tifffile.imread(f)
        cx, cy, r = det.detect(img)
        
        # Radii: 1.005*r to 1.10*r
        n_rad = 20
        rads = np.linspace(r * 1.005, r * 1.10, n_rad)
        
        def get_peak_ang(min_deg, max_deg, n_pts=120):
            if min_deg > max_deg:
                angs = np.linspace(np.deg2rad(min_deg), np.deg2rad(max_deg + 360), n_pts)
            else:
                angs = np.linspace(np.deg2rad(min_deg), np.deg2rad(max_deg), n_pts)
            g_th, g_r = np.meshgrid(angs, rads)
            xs = (cx + g_r * np.cos(g_th)).astype(np.float32)
            ys = (cy + g_r * np.sin(g_th)).astype(np.float32)
            
            rc = cv2.remap(img[:, :, 0].astype(np.float32), xs, ys, interpolation=cv2.INTER_LINEAR)
            gc = cv2.remap(img[:, :, 1].astype(np.float32), xs, ys, interpolation=cv2.INTER_LINEAR)
            bc = cv2.remap(img[:, :, 2].astype(np.float32), xs, ys, interpolation=cv2.INTER_LINEAR)
            
            prom = np.maximum(0, rc - 0.5 * (gc + bc))
            prof = np.mean(prom, axis=0)
            max_v = np.max(prof)
            if max_v < 1500: # threshold for reliable detection
                return None, max_v
            
            mask = prof >= 0.7 * max_v
            u_angs = np.unwrap(angs)
            peak_ang = np.rad2deg(np.average(u_angs[mask], weights=prof[mask])) % 360.0
            return peak_ang, max_v
            
        ang_top, v_top = get_peak_ang(250, 280)
        ang_rt, v_rt = get_peak_ang(345, 15)
        ang_lt, v_lt = get_peak_ang(165, 195)
        
        records.append({
            "idx": idx, "fname": fname,
            "top": (ang_top, v_top),
            "rt": (ang_rt, v_rt),
            "lt": (ang_lt, v_lt)
        })
        
    # Analyze coverage
    n_top = sum(1 for r in records if r["top"][0] is not None)
    n_rt = sum(1 for r in records if r["rt"][0] is not None)
    n_both_top_rt = sum(1 for r in records if r["top"][0] is not None and r["rt"][0] is not None)
    
    print(f"Coverage out of {len(records)} frames:")
    print(f"  Top Prominence detected:   {n_top}/{len(records)} ({n_top/len(records)*100:.1f}%)")
    print(f"  Right Prominence detected: {n_rt}/{len(records)} ({n_rt/len(records)*100:.1f}%)")
    print(f"  Both Top & Right detected: {n_both_top_rt}/{len(records)} ({n_both_top_rt/len(records)*100:.1f}%)")
    
    # Print sample of angles where both detected
    print("\nSample of detected angles (Top, Right, and angle offset relative to frame 0):")
    base_top = records[0]["top"][0]
    base_rt = records[0]["rt"][0]
    for r in records[::15]:
        t_ang = r["top"][0]
        r_ang = r["rt"][0]
        t_str = f"{t_ang:6.2f}°" if t_ang else "  None "
        r_str = f"{r_ang:6.2f}°" if r_ang else "  None "
        d_top = f"{t_ang - base_top:+6.2f}°" if t_ang and base_top else "  None "
        d_rt = f"{(r_ang - base_rt + 180)%360 - 180:+6.2f}°" if r_ang and base_rt else "  None "
        print(f"  [{r['idx']:3d}] {r['fname']} | Top: {t_str} (d={d_top}) | Right: {r_str} (d={d_rt})")

if __name__ == "__main__":
    analyze_all_152_totality()

