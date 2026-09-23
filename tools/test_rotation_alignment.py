import os
import glob
import cv2
import numpy as np
import tifffile
from detector import EclipseCircleDetector

def get_polar_fingerprint(img, cx, cy, r, n_angles=1440, n_radii=120):
    """
    Extracts an annular polar representation around (cx, cy) covering:
    r_inner = 1.01 * r (just outside lunar limb)
    r_outer = 1.35 * r (covering prominences and inner coronal loops)
    
    Combines:
    - Luminance gradient (coronal arches/streamers)
    - Chromatic prominence signal (H-alpha excess: R - 0.5*(G+B))
    """
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False, dtype=np.float32)
    radii = np.linspace(r * 1.01, r * 1.35, n_radii, dtype=np.float32)
    
    grid_theta, grid_r = np.meshgrid(angles, radii)
    xs = cx + grid_r * np.cos(grid_theta)
    ys = cy + grid_r * np.sin(grid_theta)
    
    if img.ndim == 3:
        r_chan = cv2.remap(img[:, :, 0].astype(np.float32), xs, ys, interpolation=cv2.INTER_LINEAR)
        g_chan = cv2.remap(img[:, :, 1].astype(np.float32), xs, ys, interpolation=cv2.INTER_LINEAR)
        b_chan = cv2.remap(img[:, :, 2].astype(np.float32), xs, ys, interpolation=cv2.INTER_LINEAR)
        
        # Luminance
        lum = 0.299 * r_chan + 0.587 * g_chan + 0.114 * b_chan
        # Prominence excess (H-alpha red)
        prom = np.maximum(0, r_chan - 0.5 * (g_chan + b_chan))
    else:
        lum = cv2.remap(img.astype(np.float32), xs, ys, interpolation=cv2.INTER_LINEAR)
        prom = np.zeros_like(lum)
        
    # Log stretch luminance
    log_lum = np.log1p(lum)
    # High-pass filter along theta to emphasize structural streamers & arches
    grad_theta = cv2.Sobel(log_lum, cv2.CV_32F, 1, 0, ksize=3)
    
    # Combined fingerprint: normalized gradient + normalized prominence
    def norm(a):
        s = np.std(a)
        return (a - np.mean(a)) / (s if s > 0 else 1.0)
        
    fingerprint = norm(grad_theta) + 2.0 * norm(np.log1p(prom))
    return fingerprint

def measure_rotation_angle(fp_ref, fp_tgt, n_angles=1440):
    """
    Measures rotation angle (in degrees) using 2D phase correlation along theta axis.
    """
    # Compute 2D phase correlation with Hann window
    hann_y = np.hanning(fp_ref.shape[0])[:, None]
    hann_x = np.hanning(fp_ref.shape[1])[None, :]
    window = hann_y * hann_x
    
    w_ref = fp_ref * window
    w_tgt = fp_tgt * window
    
    # FFT along theta axis (x axis)
    F_ref = np.fft.rfft2(w_ref)
    F_tgt = np.fft.rfft2(w_tgt)
    
    cross = F_ref * np.conj(F_tgt)
    norm = np.abs(cross)
    norm[norm == 0] = 1.0
    cross /= norm
    
    corr = np.fft.irfft2(cross, s=fp_ref.shape)
    
    # Shift correlation so 0 is center
    h, w = corr.shape
    # We only care about shift along theta (horizontal axis w)
    # Average correlation across small vertical shift range around 0
    v_slice = np.concatenate([corr[-3:, :], corr[:4, :]], axis=0)
    theta_corr = np.mean(v_slice, axis=0)
    
    peak_idx = int(np.argmax(theta_corr))
    # Sub-pixel parabolic interpolation
    left = theta_corr[(peak_idx - 1) % w]
    center = theta_corr[peak_idx]
    right = theta_corr[(peak_idx + 1) % w]
    
    denom = 2.0 * (2.0 * center - left - right)
    if abs(denom) > 1e-6:
        sub_shift = (left - right) / denom
    else:
        sub_shift = 0.0
        
    shift = peak_idx + sub_shift
    if shift > w / 2.0:
        shift -= w
        
    angle_deg = -shift * (360.0 / w)
    peak_val = float(center)
    return angle_deg, peak_val

def test():
    detector = EclipseCircleDetector()
    files = sorted(glob.glob("../EclipseTry2/3_Totality/*.tif"))
    print(f"Total files in 3_Totality: {len(files)}")
    
    # Reference frame: mid-totality (frame 75)
    ref_idx = len(files) // 2
    ref_file = files[ref_idx]
    ref_img = tifffile.imread(ref_file)
    ref_cx, ref_cy, ref_r = detector.detect(ref_img)
    ref_fp = get_polar_fingerprint(ref_img, ref_cx, ref_cy, ref_r)
    
    print(f"\nMaster Reference Frame [{ref_idx}]: {os.path.basename(ref_file)} (Center: {ref_cx:.1f}, {ref_cy:.1f}, R={ref_r:.1f})")
    print(f"{'Idx':<4} | {'Filename':<15} | {'Sun Center (cx, cy)':<20} | {'Rotation Angle':<16} | {'Peak Corr':<10}")
    print("-" * 75)
    
    test_indices = [0, 15, 30, 45, 60, ref_idx, 90, 105, 120, 135, 151]
    
    angles_found = []
    for idx in test_indices:
        fpath = files[idx]
        fname = os.path.basename(fpath)
        img = tifffile.imread(fpath)
        cx, cy, r = detector.detect(img)
        fp = get_polar_fingerprint(img, cx, cy, r)
        
        angle_deg, score = measure_rotation_angle(ref_fp, fp)
        angles_found.append((idx, fname, angle_deg, score))
        print(f"{idx:<4} | {fname:<15} | ({cx:6.1f}, {cy:6.1f})     | {angle_deg:+10.3f}°        | {score:8.4f}")

if __name__ == "__main__":
    test()

