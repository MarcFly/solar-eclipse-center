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

# Look at rows 160 to 250, cols 760 to 810
crop_t0 = ha0[160:250, 760:810]
crop_t4 = ha4[160:250, 760:810]

# Normalized cross correlation between crop_t0 and crop_t4
res = cv2.matchTemplate(crop_t4.astype(np.float32), crop_t0[20:70, 10:40].astype(np.float32), cv2.TM_CCOEFF_NORMED)
_, max_v, _, max_l = cv2.minMaxLoc(res)
print(f"Top prominence match: score={max_v:.3f}, loc={max_l}")

# Let's save the two patches side by side as an image
vis0 = (crop_t0 / crop_t0.max() * 255).astype(np.uint8)
vis4 = (crop_t4 / crop_t4.max() * 255).astype(np.uint8)
combined = np.hstack([vis0, vis4])
cv2.imwrite("../samples/top_prominence_side_by_side.jpg", cv2.resize(combined, (400, 360), interpolation=cv2.INTER_NEAREST))
print("Saved ../samples/top_prominence_side_by_side.jpg")

