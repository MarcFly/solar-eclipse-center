import os
import glob
import cv2
import numpy as np
import tifffile
from detector import EclipseCircleDetector

def measure_prominence_drift():
    detector = EclipseCircleDetector()
    files = sorted(glob.glob("../EclipseTry2/3_Totality/*.tif"))
    print(f"Total files in 3_Totality: {len(files)}")
    
    # Let's inspect 6 evenly spaced files:
    indices = [0, 30, 60, 90, 120, 151]
    
    for idx in indices:
        fpath = files[idx]
        fname = os.path.basename(fpath)
        img = tifffile.imread(fpath)
        cx, cy, r = detector.detect(img)
        
        # In RGB, let's find the position of the prominent eruption on the limb (around 9 o'clock / left edge)
        # Prominence has strong red excess: R - (G + B)/2
        crop_radius = int(round(r * 1.4))
        # Extract polar representation: radius from 0.95*r to 1.35*r, angle from 0 to 360 deg
        n_angles = 1440 # 0.25 deg per sample
        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        radii = np.linspace(r * 1.0, r * 1.25, 50)
        
        grid_theta, grid_r = np.meshgrid(angles, radii)
        xs = (cx + grid_r * np.cos(grid_theta)).astype(np.float32)
        ys = (cy + grid_r * np.sin(grid_theta)).astype(np.float32)
        
        r_chan = cv2.remap(img[:, :, 0].astype(np.float32), xs, ys, interpolation=cv2.INTER_LINEAR)
        g_chan = cv2.remap(img[:, :, 1].astype(np.float32), xs, ys, interpolation=cv2.INTER_LINEAR)
        b_chan = cv2.remap(img[:, :, 2].astype(np.float32), xs, ys, interpolation=cv2.INTER_LINEAR)
        
        prom = np.maximum(0, r_chan - 0.5 * (g_chan + b_chan))
        # 1D angular profile of prominence
        ang_prof = np.max(prom, axis=0)
        
        # Find peak around 180 deg (left side, 9 o'clock)
        # 180 deg is around index n_angles // 2 = 720
        win_start = int(720 - 180) # 135 deg
        win_end = int(720 + 180)   # 225 deg
        sub_prof = ang_prof[win_start:win_end]
        peak_sub_idx = np.argmax(sub_prof)
        peak_angle_deg = (win_start + peak_sub_idx) * (360.0 / n_angles)
        peak_val = sub_prof[peak_sub_idx]
        
        print(f"[{idx:3d}] {fname}: Moon center=({cx:6.1f}, {cy:6.1f}) | Prominence peak angle = {peak_angle_deg:6.2f}° (val={peak_val:.0f})")

if __name__ == "__main__":
    measure_prominence_drift()

