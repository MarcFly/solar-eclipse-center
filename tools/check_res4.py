from rotator import EclipseRotationAligner
import tifffile
from detector import EclipseCircleDetector

aligner = EclipseRotationAligner()
detector = EclipseCircleDetector()

f0 = "../EclipseTry2/3_Totality/_DSF5184.tif"
f4 = "../EclipseTry2/3_Totality/_DSF5335.tif"

img0 = tifffile.imread(f0)
img4 = tifffile.imread(f4)

cx0, cy0, r0 = detector.detect(img0)
cx4, cy4, r4 = detector.detect(img4)

aligner.initialize_reference(img0, cx0, cy0, r0, filename="_DSF5184.tif")
res4 = aligner.estimate_frame(img4, cx4, cy4, r4, filename="_DSF5335.tif")

print(f"res4: rot={res4.rotation_deg:.3f} deg, dx={res4.dx:.2f}, dy={res4.dy:.2f}")
print("Anchors in img4:")
for k, v in res4.anchors.items():
    print(f"  {k}: pt={v.point}, rel={v.rel_center}, score={v.score:.3f}")

