import os
import glob
import cv2
import numpy as np
import tifffile
from detector import EclipseCircleDetector

def scan_prominences():
    detector = EclipseCircleDetector()
    files = sorted(glob.glob("../EclipseTry2/3_Totality/*.tif"))
    print(f"Total totality files: {len(files)}")
    
    # Let's inspect a spread of 10 frames across totality
    indices = np.linspace(0, len(files)-1, 10, dtype=int)
    
    for idx in indices:
        fpath = files[idx]
        fname = os.path.basename(fpath)
        img = tifffile.imread(fpath)
        cx, cy, r = detector.detect(img)
        
        # Check max brightness in RGB
        max_r = img[:, :, 0].max()
        mean_r = img[:, :, 0].mean()
        
        # Unroll limb between 1.005*r and 1.08*r
        n_angles = 720 # 0.5 deg
        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        radii = np.linspace(r * 1.005, r * 1.08, 25)
        
        grid_theta, grid_r = np.meshgrid(angles, radii)
        xs = (cx + grid_r * np.cos(grid_theta)).astype(np.float32)
        ys = (cy + grid_r * np.sin(grid_theta)).astype(np.float32)
        
        r_c = cv2.remap(img[:, :, 0].astype(np.float32), xs, ys, interpolation=cv2.INTER_LINEAR)
        g_c = cv2.remap(img[:, :, 1].astype(np.float32), xs, ys, interpolation=cv2.INTER_LINEAR)
        b_c = cv2.remap(img[:, :, 2].astype(np.float32), xs, ys, interpolation=cv2.INTER_LINEAR)
        
        # Chromospheric/prominence signal: Red excess over Green/Blue
        # In H-alpha, prominences are vivid pink/magenta (strong R, lower G, some B)
        prom = np.maximum(0, r_c - 0.5 * (g_c + b_c))
        # Max along radial slice
        prof = np.max(prom, axis=0)
        
        # Smooth profile
        prof_s = cv2.GaussianBlur(prof[:, None], (11, 1), 2.0).ravel()
        
        # Look for peaks across 3 distinct sectors:
        # Sector 1: Right (~3 o'clock): 315° to 45° (or -45° to +45°)
        # Sector 2: Left (~9 o'clock): 135° to 225°
        # Sector 3: Top (~12 o'clock): 225° to 315°
        
        # Sector 1: Right (around 0° / 360°)
        right_indices = np.concatenate([np.where(angles >= np.deg2rad(315))[0], np.where(angles <= np.deg2rad(45))[0]])
        right_peak_idx = right_indices[np.argmax(prof_s[right_indices])]
        right_ang = np.rad2deg(angles[right_peak_idx])
        right_val = prof_s[right_peak_idx]
        
        # Sector 2: Left (around 180°)
        left_indices = np.where((angles >= np.deg2rad(135)) & (angles <= np.deg2rad(225)))[0]
        left_peak_idx = left_indices[np.argmax(prof_s[left_indices])]
        left_ang = np.rad2deg(angles[left_peak_idx])
        left_val = prof_s[left_peak_idx]
        
        # Sector 3: Top (around 270°)
        top_indices = np.where((angles >= np.deg2rad(225)) & (angles <= np.deg2rad(315)))[0]
        top_peak_idx = top_indices[np.argmax(prof_s[top_indices])]
        top_ang = np.rad2deg(angles[top_peak_idx])
        top_val = prof_s[top_peak_idx]
        
        print(f"[{idx:3d}] {fname} | MeanR={mean_r:5.0f} | Left(~9h): {left_ang:5.1f}° (v={left_val:5.0f}) | Top(~12h): {top_ang:5.1f}° (v={top_val:5.0f}) | Right(~3h): {right_ang:5.1f}° (v={right_val:5.0f})")

if __name__ == "__main__":
    scan_prominences()

