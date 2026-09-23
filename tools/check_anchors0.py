from rotator import EclipseRotationAligner
import tifffile
from detector import EclipseCircleDetector
import numpy as np

aligner = EclipseRotationAligner()
detector = EclipseCircleDetector()

f0 = "../EclipseTry2/3_Totality/_DSF5184.tif"
img0 = tifffile.imread(f0)
cx0, cy0, r0 = detector.detect(img0)
anchors0 = aligner.initialize_reference(img0, cx0, cy0, r0, filename="_DSF5184.tif")

print("Anchors in img0:")
for k, v in anchors0.items():
    print(f"  {k}: pt={v.point}, rel={v.rel_center}")

v_tr0 = anchors0["right"].point - anchors0["top"].point
ang_tr0 = np.rad2deg(np.arctan2(v_tr0[1], v_tr0[0]))
print(f"v_tr0: {v_tr0}, ang={ang_tr0:.3f} deg, dist={np.linalg.norm(v_tr0):.1f} px")

