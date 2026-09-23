import cv2
import numpy as np
import tifffile

f0_al = "../samples/step4_rotation_alignment/aligned_16bit__DSF5184.tif"
f4_al = "../samples/step4_rotation_alignment/aligned_16bit__DSF5335.tif"

img0 = tifffile.imread(f0_al)
img4 = tifffile.imread(f4_al)

# Let's inspect the two aligned images:
# Crop central 1600x1600 (center is 2000, 2000)
c = 2000
w = 800
c0 = img0[c-w:c+w, c-w:c+w]
c4 = img4[c-w:c+w, c-w:c+w]

# Extract H-alpha excess:
ha0 = np.maximum(0, c0[:, :, 0].astype(np.float32) - 0.5*(c0[:, :, 1].astype(np.float32) + c0[:, :, 2].astype(np.float32)))
ha4 = np.maximum(0, c4[:, :, 0].astype(np.float32) - 0.5*(c4[:, :, 1].astype(np.float32) + c4[:, :, 2].astype(np.float32)))

# Measure match of Top prominence (y in [180, 260], x in [740, 830])
tpl_top0 = ha0[190:250, 755:815]
res_top = cv2.matchTemplate(ha4[170:270, 735:835].astype(np.float32), tpl_top0.astype(np.float32), cv2.TM_CCOEFF_NORMED)
_, max_vt, _, max_lt = cv2.minMaxLoc(res_top)
dx_t = max_lt[0] - 20
dy_t = max_lt[1] - 20
print(f"Top Prominence after Rotation Alignment:  dx={dx_t:+2d}, dy={dy_t:+2d} (Correlation={max_vt:.4f})")

# Measure match of Right prominence (y in [760, 830], x in [1350, 1430])
tpl_rt0 = ha0[760:820, 1360:1420]
res_rt = cv2.matchTemplate(ha4[740:840, 1340:1440].astype(np.float32), tpl_rt0.astype(np.float32), cv2.TM_CCOEFF_NORMED)
_, max_vr, _, max_lr = cv2.minMaxLoc(res_rt)
dx_r = max_lr[0] - 20
dy_r = max_lr[1] - 20
print(f"Right Prominence after Rotation Alignment: dx={dx_r:+2d}, dy={dy_r:+2d} (Correlation={max_vr:.4f})")

# Measure match of Left prominence (y in [760, 830], x in [170, 240])
tpl_lt0 = ha0[760:820, 180:240]
res_lt = cv2.matchTemplate(ha4[740:840, 160:260].astype(np.float32), tpl_lt0.astype(np.float32), cv2.TM_CCOEFF_NORMED)
_, max_vl, _, max_ll = cv2.minMaxLoc(res_lt)
dx_l = max_ll[0] - 20
dy_l = max_ll[1] - 20
print(f"Left Prominence after Rotation Alignment:  dx={dx_l:+2d}, dy={dy_l:+2d} (Correlation={max_vl:.4f})")

