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

u0 = trans.transform(raw0, cx0, cy0, r0, target_r=586.5, crop_size=4000, rotation_deg=0.0).img_transformed
u4 = trans.transform(raw4, cx4, cy4, r4, target_r=586.5, crop_size=4000, rotation_deg=0.0).img_transformed

c0 = u0[1200:2800, 1200:2800]
c4 = u4[1200:2800, 1200:2800]

def get_ha(im):
    return np.maximum(0, im[:, :, 0].astype(np.float32) - 0.5*(im[:, :, 1].astype(np.float32) + im[:, :, 2].astype(np.float32)))

ha0 = get_ha(c0)
ha4 = get_ha(c4)

# Top prominence peak (around 800, 213)
_, _, _, mt0 = cv2.minMaxLoc(ha0[150:280, 740:870])
_, _, _, mt4 = cv2.minMaxLoc(ha4[150:280, 740:870])
pt_t0 = (740 + mt0[0], 150 + mt0[1])
pt_t4 = (740 + mt4[0], 150 + mt4[1])

# Right prominence peak (around 1386, 800)
_, _, _, mr0 = cv2.minMaxLoc(ha0[740:870, 1320:1450])
_, _, _, mr4 = cv2.minMaxLoc(ha4[740:870, 1320:1450])
pt_r0 = (1320 + mr0[0], 740 + mr0[1])
pt_r4 = (1320 + mr4[0], 740 + mr4[1])

# Left prominence peak (around 213, 800)
_, _, _, ml0 = cv2.minMaxLoc(ha0[740:870, 150:280])
_, _, _, ml4 = cv2.minMaxLoc(ha4[740:870, 150:280])
pt_l0 = (150 + ml0[0], 740 + ml0[1])
pt_l4 = (150 + ml4[0], 740 + ml4[1])

print(f"UNROTATED (Centered only):")
print(f"  Top:   Frame 0 = {pt_t0}, Frame 4 = {pt_t4} -> Delta = ({pt_t4[0]-pt_t0[0]}, {pt_t4[1]-pt_t0[1]})")
print(f"  Right: Frame 0 = {pt_r0}, Frame 4 = {pt_r4} -> Delta = ({pt_r4[0]-pt_r0[0]}, {pt_r4[1]-pt_r0[1]})")
print(f"  Left:  Frame 0 = {pt_l0}, Frame 4 = {pt_l4} -> Delta = ({pt_l4[0]-pt_l0[0]}, {pt_l4[1]-pt_l0[1]})")

# Calculate angles of vectors in unrotated frame:
v_tr0 = np.array(pt_r0) - np.array(pt_t0)
v_tr4 = np.array(pt_r4) - np.array(pt_t4)
ang0 = np.rad2deg(np.arctan2(v_tr0[1], v_tr0[0]))
ang4 = np.rad2deg(np.arctan2(v_tr4[1], v_tr4[0]))
print(f"Top->Right vector angle: Frame 0 = {ang0:.3f}°, Frame 4 = {ang4:.3f}° -> Difference = {ang4 - ang0:+.3f}°")

