import cv2
import numpy as np
import tifffile
from detector import EclipseCircleDetector

det = EclipseCircleDetector()

f1 = "../EclipseTry2/3_Totality/_DSF5184.tif"
f2 = "../EclipseTry2/3_Totality/_DSF5259.tif"
f3 = "../EclipseTry2/3_Totality/_DSF5335.tif"

for path in [f1, f2, f3]:
    img = tifffile.imread(path)
    cx, cy, r = det.detect(img)
    print(f"\n{path.split('/')[-1]}: Detected Moon Center=({cx:.2f}, {cy:.2f}), R={r:.2f}")
    
    # Let's inspect the limb profile along the entire 360 degrees
    # Unroll limb at r = 586.5
    n_a = 360
    angs = np.linspace(0, 2*np.pi, n_a, endpoint=False)
    xs = (cx + r * np.cos(angs)).astype(np.float32)
    ys = (cy + r * np.sin(angs)).astype(np.float32)
    
    # Sample intensity 5 pixels inside limb (moon) and 5 pixels outside limb (prominence/corona)
    xs_in = (cx + (r - 10) * np.cos(angs)).astype(np.float32)
    ys_in = (cy + (r - 10) * np.sin(angs)).astype(np.float32)
    xs_out = (cx + (r + 10) * np.cos(angs)).astype(np.float32)
    ys_out = (cy + (r + 10) * np.sin(angs)).astype(np.float32)
    
    val_in = cv2.remap(img[:, :, 0].astype(np.float32), xs_in, ys_in, interpolation=cv2.INTER_LINEAR)
    val_out = cv2.remap(img[:, :, 0].astype(np.float32), xs_out, ys_out, interpolation=cv2.INTER_LINEAR)
    
    print(f"  Limb contrast inside moon (mean R): {val_in.mean():.1f}")
    print(f"  Limb contrast outside moon (mean R): {val_out.mean():.1f}")

