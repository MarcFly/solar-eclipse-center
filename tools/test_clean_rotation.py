import tifffile
import numpy as np
import cv2

f0 = "../samples/step4_rotation_alignment/aligned_16bit__DSF5184.tif"
f4 = "../samples/step4_rotation_alignment/aligned_16bit__DSF5335.tif"

# Let's inspect the two images when transformed WITH rotation but WITHOUT dx, dy:
from transformer import EclipseAffineTransformer
from detector import EclipseCircleDetector

det = EclipseCircleDetector()
trans = EclipseAffineTransformer()

raw0 = tifffile.imread("../EclipseTry2/3_Totality/_DSF5184.tif")
raw4 = tifffile.imread("../EclipseTry2/3_Totality/_DSF5335.tif")

cx0, cy0, r0 = det.detect(raw0)
cx4, cy4, r4 = det.detect(raw4)

# Pure rotation around center, no dx/dy:
res0_clean = trans.transform(raw0, cx0, cy0, r0, target_r=586.5, crop_size=4000, rotation_deg=0.0)
res4_clean = trans.transform(raw4, cx4, cy4, r4, target_r=586.5, crop_size=4000, rotation_deg=-2.585)

# Check Top and Right prominence positions in 4000x4000 canvas (center is 2000, 2000):
# Top is around (2000, 2000 - 586.5) = (2000, 1413.5)
# Right is around (2000 + 586.5, 2000) = (2586.5, 2000)

crop0 = res0_clean.img_transformed[1200:2800, 1200:2800]
crop4 = res4_clean.img_transformed[1200:2800, 1200:2800]

def get_halpha(img):
    return np.maximum(0, img[:, :, 0].astype(np.float32) - 0.5*(img[:, :, 1].astype(np.float32) + img[:, :, 2].astype(np.float32)))

ha0 = get_halpha(crop0)
ha4 = get_halpha(crop4)

# Top prominence peak (in 1600x1600 crop, center is 800, 800)
# Top is at x ~ 800, y ~ 800 - 586.5 = 213.5
top_box0 = ha0[150:280, 740:870]
top_box4 = ha4[150:280, 740:870]
_, _, _, max_t0 = cv2.minMaxLoc(top_box0)
_, _, _, max_t4 = cv2.minMaxLoc(top_box4)
pt_t0 = (740 + max_t0[0], 150 + max_t0[1])
pt_t4 = (740 + max_t4[0], 150 + max_t4[1])

# Right prominence peak
rt_box0 = ha0[740:870, 1320:1450]
rt_box4 = ha4[740:870, 1320:1450]
_, _, _, max_r0 = cv2.minMaxLoc(rt_box0)
_, _, _, max_r4 = cv2.minMaxLoc(rt_box4)
pt_r0 = (1320 + max_r0[0], 740 + max_r0[1])
pt_r4 = (1320 + max_r4[0], 740 + max_r4[1])

print(f"Top prominence:   frame0={pt_t0}, frame4={pt_t4} -> delta={(pt_t4[0]-pt_t0[0], pt_t4[1]-pt_t0[1])}")
print(f"Right prominence: frame0={pt_r0}, frame4={pt_r4} -> delta={(pt_r4[0]-pt_r0[0], pt_r4[1]-pt_r0[1])}")

