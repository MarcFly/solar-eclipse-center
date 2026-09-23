import os
import glob
import cv2
import numpy as np
import tifffile
from detector import EclipseCircleDetector

def polar_unwarp(img, cx, cy, r_inner, r_outer, out_w=720, out_h=100):
    """
    Unwarps an annular ring [r_inner, r_outer] around (cx, cy) into polar coordinates (r, theta).
    Horizontal axis is theta (0 to 360 deg), vertical axis is radius (r_inner to r_outer).
    """
    angles = np.linspace(0, 2*np.pi, out_w, endpoint=False)
    radii = np.linspace(r_inner, r_outer, out_h)
    
    # Grid of (theta, r)
    grid_theta, grid_r = np.meshgrid(angles, radii)
    
    xs = (cx + grid_r * np.cos(grid_theta)).astype(np.float32)
    ys = (cy + grid_r * np.sin(grid_theta)).astype(np.float32)
    
    unwarped = cv2.remap(img, xs, ys, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return unwarped

def estimate_rotation(polar_ref, polar_tgt):
    """
    Estimates angular shift (degrees) using 1D phase correlation along theta axis.
    """
    # Average across radial dimension to get a 1D azimuthal profile of prominences/corona
    prof_ref = np.mean(polar_ref, axis=0).astype(np.float32)
    prof_tgt = np.mean(polar_tgt, axis=0).astype(np.float32)
    
    # Subtract mean / high-pass filter
    prof_ref -= np.mean(prof_ref)
    prof_tgt -= np.mean(prof_tgt)
    
    n = len(prof_ref)
    # FFT cross-correlation
    f_ref = np.fft.rfft(prof_ref)
    f_tgt = np.fft.rfft(prof_tgt)
    cross = f_ref * np.conj(f_tgt)
    # Phase correlation
    norm = np.abs(cross)
    norm[norm == 0] = 1.0
    cross /= norm
    corr = np.fft.irfft(cross, n=n)
    
    shift_idx = np.argmax(corr)
    if shift_idx > n // 2:
        shift_idx -= n
        
    angle_deg = shift_idx * (360.0 / n)
    return angle_deg, corr[shift_idx]

def test():
    detector = EclipseCircleDetector()
    files = sorted(glob.glob("../EclipseTry2/3_Totality/*.tif"))
    print(f"Total images in 3_Totality: {len(files)}")
    
    # Reference image: first image in totality
    ref_file = files[0]
    ref_img = tifffile.imread(ref_file)
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY) if ref_img.ndim == 3 else ref_img
    ref_cx, ref_cy, ref_r = detector.detect(ref_img)
    
    # Prominence & inner corona ring: r to 1.35 * r
    polar_ref = polar_unwarp(ref_gray.astype(np.float32), ref_cx, ref_cy, ref_r * 1.01, ref_r * 1.35)
    
    # Sample every 15 images across totality
    sample_indices = list(range(0, len(files), 15))
    if sample_indices[-1] != len(files) - 1:
        sample_indices.append(len(files) - 1)
        
    print(f"\nReference: {os.path.basename(ref_file)} (cx={ref_cx:.1f}, cy={ref_cy:.1f}, r={ref_r:.1f})")
    print(f"{'Index':<6} | {'Filename':<15} | {'Center (cx, cy)':<18} | {'Rot Angle (deg)':<16} | {'Confidence':<10}")
    print("-" * 75)
    
    for idx in sample_indices:
        fpath = files[idx]
        fname = os.path.basename(fpath)
        img = tifffile.imread(fpath)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        cx, cy, r = detector.detect(img)
        
        polar_tgt = polar_unwarp(gray.astype(np.float32), cx, cy, r * 1.01, r * 1.35)
        angle_deg, conf = estimate_rotation(polar_ref, polar_tgt)
        
        print(f"{idx:<6} | {fname:<15} | ({cx:6.1f}, {cy:6.1f}) | {angle_deg:+10.3f} deg   | {conf:8.4f}")

if __name__ == "__main__":
    test()

