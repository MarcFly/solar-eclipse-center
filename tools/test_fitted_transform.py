import cv2
import numpy as np
import tifffile
from transformer import EclipseAffineTransformer
from detector import EclipseCircleDetector

det = EclipseCircleDetector()
trans = EclipseAffineTransformer()

raw0 = tifffile.imread("../EclipseTry2/3_Totality/_DSF5184.tif")
raw4 = tifffile.imread("../EclipseTry2/3_Totality/_DSF5335.tif")

cx0, cy0, r0 = det.detect(raw0)
cx4, cy4, r4 = det.detect(raw4)

# 3 observed corresponding points in unrotated 1600x1600 crop (centered at 800, 800):
# Frame 0:
# Top:   (776, 223)
# Right: (1389, 780)
# Left:  (215, 775)
pts0 = np.array([
    [776.0, 223.0],
    [1389.0, 780.0],
    [215.0, 775.0]
], dtype=np.float32)

# Frame 4:
# Top:   (776 + 10, 223 + 1) = (786.0, 224.0)
# Right: (1389 - 14, 780 + 20) = (1375.0, 800.0)
# Left:  (215 + 3, 775 - 8) = (218.0, 767.0)
pts4 = np.array([
    [786.0, 224.0],
    [1375.0, 800.0],
    [218.0, 767.0]
], dtype=np.float32)

# Fit rigid Euclidean transform from Frame 4 to Frame 0
M_rigid, inliers = cv2.estimateAffinePartial2D(pts4, pts0)
rot_deg = np.rad2deg(np.arctan2(M_rigid[1, 0], M_rigid[0, 0]))
scale = np.sqrt(M_rigid[0, 0]**2 + M_rigid[1, 0]**2)
tx = M_rigid[0, 2]
ty = M_rigid[1, 2]

print(f"Optimal Transformation from Frame 4 to Frame 0:")
print(f"  Rotation: {rot_deg:+.3f} degrees")
print(f"  Scale:    {scale:.4f}")
print(f"  tx:       {tx:+.2f} px")
print(f"  ty:       {ty:+.2f} px")

# Check residual reprojection error:
warped_pts4 = cv2.transform(pts4[None, :, :], M_rigid)[0]
err = np.linalg.norm(warped_pts4 - pts0, axis=1)
print(f"Reprojection errors on 3 prominences:")
print(f"  Top:   {err[0]:.2f} px")
print(f"  Right: {err[1]:.2f} px")
print(f"  Left:  {err[2]:.2f} px")
print(f"  Mean Error: {err.mean():.2f} px")

