import cv2
import numpy as np
import tifffile

def solve_motion():
    f1 = "../processed_eclipse/3_Totality/_DSF5184.tif"
    f2 = "../processed_eclipse/3_Totality/_DSF5259.tif"
    f3 = "../processed_eclipse/3_Totality/_DSF5335.tif"
    
    img1 = tifffile.imread(f1)
    img2 = tifffile.imread(f2)
    img3 = tifffile.imread(f3)
    
    # We want to find the rigid Euclidean transform [R | t] that aligns:
    # the solar features (prominences + corona) from imgX to img1 (or vice versa).
    
    # Let's mask out the dark Moon disk (r <= 595 px) to avoid the moving Moon influencing the fit.
    # And let's focus on the region r in [600, 1600] px where prominences and corona live.
    
    h, w = 4000, 4000
    cy, cx = 2000, 2000
    
    # Downsample to 1000x1000 for fast, robust ECC / phase correlation
    scale = 0.25
    s1 = cv2.resize(img1, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    s2 = cv2.resize(img2, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    s3 = cv2.resize(img3, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    scx, scy = cx * scale, cy * scale
    sr = 586.5 * scale
    
    # Create mask for corona & prominences (outside lunar disk, inside reasonable radius)
    Y, X = np.ogrid[:s1.shape[0], :s1.shape[1]]
    dist_sq = (X - scx)**2 + (Y - scy)**2
    
    # Mask: r from 1.02 * sr to 2.2 * sr
    mask = ((dist_sq >= (1.02 * sr)**2) & (dist_sq <= (2.2 * sr)**2)).astype(np.uint8) * 255
    
    # Let's convert to log-luminance
    def to_feature_map(img_s):
        # We can use luminance or max(R - 0.5(G+B), 0) + log(lum)
        lum = 0.299 * img_s[:, :, 0] + 0.587 * img_s[:, :, 1] + 0.114 * img_s[:, :, 2]
        r_excess = np.maximum(0, img_s[:, :, 0].astype(np.float32) - 0.5 * (img_s[:, :, 1].astype(np.float32) + img_s[:, :, 2].astype(np.float32)))
        # Log compression for dynamic range
        feat = np.log1p(lum.astype(np.float32)) + 2.0 * np.log1p(r_excess)
        # Normalize
        feat = (feat - feat.mean()) / (feat.std() + 1e-6)
        return feat.astype(np.float32)
        
    feat1 = to_feature_map(s1)
    feat2 = to_feature_map(s2)
    feat3 = to_feature_map(s3)
    
    # Find Euclidean transform (rotation + translation) using cv2.findTransformECC
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    
    warp_matrix2 = np.eye(2, 3, dtype=np.float32)
    cc2, warp_matrix2 = cv2.findTransformECC(feat1, feat2, warp_matrix2, cv2.MOTION_EUCLIDEAN, criteria, mask, 5)
    
    warp_matrix3 = np.eye(2, 3, dtype=np.float32)
    cc3, warp_matrix3 = cv2.findTransformECC(feat1, feat3, warp_matrix3, cv2.MOTION_EUCLIDEAN, criteria, mask, 5)
    
    def decompose_warp(M, sc):
        # M is 2x3: [[cos, -sin, tx], [sin, cos, ty]]
        cos_t = M[0, 0]
        sin_t = M[1, 0]
        rot_deg = np.rad2deg(np.arctan2(sin_t, cos_t))
        tx = M[0, 2] / sc
        ty = M[1, 2] / sc
        return rot_deg, tx, ty
        
    rot2, tx2, ty2 = decompose_warp(warp_matrix2, scale)
    rot3, tx3, ty3 = decompose_warp(warp_matrix3, scale)
    
    print(f"ECC Alignment relative to _DSF5184 (Reference):")
    print(f"  _DSF5259: Rotation = {rot2:+7.3f}°, tx = {tx2:+6.2f} px, ty = {ty2:+6.2f} px (correlation={cc2:.4f})")
    print(f"  _DSF5335: Rotation = {rot3:+7.3f}°, tx = {tx3:+6.2f} px, ty = {ty3:+6.2f} px (correlation={cc3:.4f})")

if __name__ == "__main__":
    solve_motion()

