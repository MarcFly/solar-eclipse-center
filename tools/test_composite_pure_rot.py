import cv2
import numpy as np
import tifffile
from transformer import EclipseAffineTransformer
from detector import EclipseCircleDetector

det = EclipseCircleDetector()
trans = EclipseAffineTransformer()

f0 = "../EclipseTry2/3_Totality/_DSF5184.tif"
f2 = "../EclipseTry2/3_Totality/_DSF5259.tif"
f4 = "../EclipseTry2/3_Totality/_DSF5335.tif"

img0 = tifffile.imread(f0)
img2 = tifffile.imread(f2)
img4 = tifffile.imread(f4)

cx0, cy0, r0 = det.detect(img0)
cx2, cy2, r2 = det.detect(img2)
cx4, cy4, r4 = det.detect(img4)

# Pure rotation around center (rot0 = 0.0, rot2 = -1.242, rot4 = -2.585)
t0 = trans.transform(img0, cx0, cy0, r0, target_r=586.5, crop_size=4000, rotation_deg=0.0).img_transformed
t2 = trans.transform(img2, cx2, cy2, r2, target_r=586.5, crop_size=4000, rotation_deg=-1.242).img_transformed
t4 = trans.transform(img4, cx4, cy4, r4, target_r=586.5, crop_size=4000, rotation_deg=-2.585).img_transformed

# Unrotated
u0 = trans.transform(img0, cx0, cy0, r0, target_r=586.5, crop_size=4000, rotation_deg=0.0).img_transformed
u2 = trans.transform(img2, cx2, cy2, r2, target_r=586.5, crop_size=4000, rotation_deg=0.0).img_transformed
u4 = trans.transform(img4, cx4, cy4, r4, target_r=586.5, crop_size=4000, rotation_deg=0.0).img_transformed

def make_comp(c0, c2, c4):
    w = 800
    c = 2000
    p0 = c0[c-w:c+w, c-w:c+w]
    p2 = c2[c-w:c+w, c-w:c+w]
    p4 = c4[c-w:c+w, c-w:c+w]
    
    def norm(im):
        g = 0.299*im[:, :, 0] + 0.587*im[:, :, 1] + 0.114*im[:, :, 2]
        lg = np.log1p(g.astype(np.float32))
        return ((lg - lg.min()) / (lg.max() - lg.min() + 1e-6) * 255.0).astype(np.uint8)
        
    comp = np.zeros((2*w, 2*w, 3), dtype=np.uint8)
    comp[:, :, 2] = norm(p0) # Red: Frame 0
    comp[:, :, 1] = norm(p2) # Green: Frame 75
    comp[:, :, 0] = norm(p4) # Blue: Frame 151
    return comp

comp_unrot = make_comp(u0, u2, u4)
comp_rot = make_comp(t0, t2, t4)

# Measure chromatic fringing in the annular prominence/inner corona band: r in [570, 750]
Y, X = np.ogrid[:1600, :1600]
dist_c = np.sqrt((X - 800)**2 + (Y - 800)**2)
mask = (dist_c >= 570) & (dist_c <= 750)

def prom_fringing(comp):
    r = comp[:, :, 2].astype(np.float32)
    g = comp[:, :, 1].astype(np.float32)
    b = comp[:, :, 0].astype(np.float32)
    m = (r + g + b) / 3.0
    var = ((r - m)**2 + (g - m)**2 + (b - m)**2) / 3.0
    return float(np.mean(np.sqrt(var[mask])))

f_unrot = prom_fringing(comp_unrot)
f_rot = prom_fringing(comp_rot)

print(f"Inner Prominence Band Fringing:")
print(f"  Unrotated:        {f_unrot:.2f} counts")
print(f"  Rotation-Aligned: {f_rot:.2f} counts")
print(f"  Improvement:      {(f_unrot - f_rot)/f_unrot * 100.0:.1f}% reduction in chromatic fringing!")

