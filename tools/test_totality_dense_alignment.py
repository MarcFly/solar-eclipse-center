import os
import glob
import cv2
import numpy as np
import tifffile
from detector import EclipseCircleDetector

def dense_alignment_test():
    files = sorted(glob.glob("../EclipseTry2/3_Totality/*.tif"))
    print(f"Total files: {len(files)}")
    
    # Let's test on 16 evenly spaced frames
    indices = np.linspace(0, len(files)-1, 16, dtype=int)
    
    det = EclipseCircleDetector()
    
    # We will center each on its Moon center first (like Step 4 does)
    # Then we measure the residual transformation (rotation + translation) needed to lock the solar features to a master reference.
    # Master reference: frame 0 (or mid-totality frame)
    ref_idx = 0
    ref_file = files[ref_idx]
    ref_img = tifffile.imread(ref_file)
    cx0, cy0, r0 = det.detect(ref_img)
    
    # Scale factor to nominal 586.5
    s0 = 586.5 / r0
    # Center at (2000, 2000)
    M0 = np.array([
        [s0, 0, 2000.0 - s0 * cx0],
        [0, s0, 2000.0 - s0 * cy0]
    ], dtype=np.float32)
    ref_centered = cv2.warpAffine(ref_img, M0, (4000, 4000), flags=cv2.INTER_LANCZOS4)
    
    # Downscale for fast feature alignment
    scale = 0.25
    ref_s = cv2.resize(ref_centered, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # Extract prominence + inner corona feature map
    def to_feature(img_s):
        r = img_s[:, :, 0].astype(np.float32)
        g = img_s[:, :, 1].astype(np.float32)
        b = img_s[:, :, 2].astype(np.float32)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        prom = np.maximum(0, r - 0.5 * (g + b))
        # We boost the prominences significantly
        f = np.log1p(lum) + 3.0 * np.log1p(prom)
        return ((f - f.mean()) / (f.std() + 1e-6)).astype(np.float32)
        
    ref_feat = to_feature(ref_s)
    
    # Mask for solar limb/prominences: r in [595, 1200]
    Y, X = np.ogrid[:ref_s.shape[0], :ref_s.shape[1]]
    d2 = (X - 2000*scale)**2 + (Y - 2000*scale)**2
    mask = ((d2 >= (595*scale)**2) & (d2 <= (1200*scale)**2)).astype(np.uint8) * 255
    
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    
    print(f"Master Reference: [{ref_idx}] {os.path.basename(ref_file)}")
    print(f"{'Idx':<4} | {'Filename':<15} | {'Rotation (deg)':<16} | {'tx (px)':<10} | {'ty (px)':<10} | {'Corr':<8}")
    print("-" * 75)
    
    aligned_crops = []
    
    for idx in indices:
        f = files[idx]
        fname = os.path.basename(f)
        img = tifffile.imread(f)
        cx, cy, r = det.detect(img)
        s = 586.5 / r
        M = np.array([
            [s, 0, 2000.0 - s * cx],
            [0, s, 2000.0 - s * cy]
        ], dtype=np.float32)
        centered = cv2.warpAffine(img, M, (4000, 4000), flags=cv2.INTER_LANCZOS4)
        
        tgt_s = cv2.resize(centered, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        tgt_feat = to_feature(tgt_s)
        
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(ref_feat, tgt_feat, warp, cv2.MOTION_EUCLIDEAN, criteria, mask, 5)
            rot = np.rad2deg(np.arctan2(warp[1, 0], warp[0, 0]))
            tx = warp[0, 2] / scale
            ty = warp[1, 2] / scale
            print(f"{idx:<4} | {fname:<15} | {rot:+12.3f}°    | {tx:+8.2f}   | {ty:+8.2f}   | {cc:6.4f}")
        except Exception as e:
            print(f"{idx:<4} | {fname:<15} | FAILED: {e}")

if __name__ == "__main__":
    dense_alignment_test()

