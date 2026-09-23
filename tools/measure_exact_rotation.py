import cv2
import numpy as np
import tifffile
from detector import EclipseCircleDetector

det = EclipseCircleDetector()

f0 = "../EclipseTry2/3_Totality/_DSF5184.tif"
f4 = "../EclipseTry2/3_Totality/_DSF5335.tif"

img0 = tifffile.imread(f0)
img4 = tifffile.imread(f4)

cx0, cy0, r0 = det.detect(img0)
cx4, cy4, r4 = det.detect(img4)

print(f"Frame 0: cx={cx0:.2f}, cy={cy0:.2f}, r={r0:.2f}")
print(f"Frame 4: cx={cx4:.2f}, cy={cy4:.2f}, r={r4:.2f}")

# Extract narrow polar rings around the limb (r to 1.10*r)
# with 3600 angular samples (0.1 degree per sample)
n_ang = 3600
angs = np.linspace(0, 2*np.pi, n_ang, endpoint=False)
rads = np.linspace(r0 * 1.005, r0 * 1.10, 30)

grid_th, grid_r = np.meshgrid(angs, rads)

def unroll_ha(img, cx, cy, r):
    xs = (cx + grid_r * np.cos(grid_th)).astype(np.float32)
    ys = (cy + grid_r * np.sin(grid_th)).astype(np.float32)
    r_c = cv2.remap(img[:, :, 0].astype(np.float32), xs, ys, cv2.INTER_LINEAR)
    g_c = cv2.remap(img[:, :, 1].astype(np.float32), xs, ys, cv2.INTER_LINEAR)
    b_c = cv2.remap(img[:, :, 2].astype(np.float32), xs, ys, cv2.INTER_LINEAR)
    ha = np.maximum(0, r_c - 0.5 * (g_c + b_c))
    return np.mean(ha, axis=0)

prof0 = unroll_ha(img0, cx0, cy0, r0)
prof4 = unroll_ha(img4, cx4, cy4, r4)

# Phase correlate prof0 and prof4 to find EXACT 1D angular shift:
# High-pass filter / remove mean
p0 = prof0 - np.mean(prof0)
p4 = prof4 - np.mean(prof4)

F0 = np.fft.rfft(p0)
F4 = np.fft.rfft(p4)

cross = F0 * np.conj(F4)
cross /= (np.abs(cross) + 1e-6)
corr = np.fft.irfft(cross, n=n_ang)

shift_idx = np.argmax(corr)
if shift_idx > n_ang // 2:
    shift_idx -= n_ang
    
rot_angle_deg = shift_idx * (360.0 / n_ang)
peak_corr = corr[shift_idx % n_ang]

print(f"\n1D Global Azimuthal Phase Correlation across all 360 degrees:")
print(f"  Exact Rotation from Frame 0 to Frame 4: {rot_angle_deg:+.3f} degrees (Peak={peak_corr:.4f})")

