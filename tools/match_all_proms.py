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

# Right prominence: rows 750 to 850, cols 1340 to 1440
crop_r0 = ha0[750:850, 1340:1440]
crop_r4 = ha4[750:850, 1340:1440]

res_r = cv2.matchTemplate(crop_r4.astype(np.float32), crop_r0[20:80, 20:80].astype(np.float32), cv2.TM_CCOEFF_NORMED)
_, max_vr, _, max_lr = cv2.minMaxLoc(res_r)
dx_r = max_lr[0] - 20
dy_r = max_lr[1] - 20
print(f"Right prominence match: score={max_vr:.3f}, dx={dx_r}, dy={dy_r}")

# Left prominence: rows 750 to 850, cols 160 to 260
crop_l0 = ha0[750:850, 160:260]
crop_l4 = ha4[750:850, 160:260]

res_l = cv2.matchTemplate(crop_l4.astype(np.float32), crop_l0[20:80, 20:80].astype(np.float32), cv2.TM_CCOEFF_NORMED)
_, max_vl, _, max_ll = cv2.minMaxLoc(res_l)
dx_l = max_ll[0] - 20
dy_l = max_ll[1] - 20
print(f"Left prominence match:  score={max_vl:.3f}, dx={dx_l}, dy={dy_l}")

