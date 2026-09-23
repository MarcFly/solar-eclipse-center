import os
import glob
import cv2
import numpy as np
import tifffile
from detector import EclipseCircleDetector

def inspect_eruptions():
    detector = EclipseCircleDetector()
    files = [
        "../EclipseTry2/3_Totality/_DSF5184.tif",
        "../EclipseTry2/3_Totality/_DSF5206.tif",
        "../EclipseTry2/3_Totality/_DSF5250.tif",
        "../EclipseTry2/3_Totality/_DSF5300.tif",
        "../EclipseTry2/3_Totality/_DSF5335.tif",
    ]
    
    scale = 0.5
    for path in files:
        if not os.path.exists(path):
            continue
        fname = os.path.basename(path)
        img = tifffile.imread(path)
        cx, cy, r = detector.detect(img)
        h, w = img.shape[:2]
        
        # Crop 1600x1600 around sun center
        half = 800
        x1 = max(0, int(round(cx - half)))
        y1 = max(0, int(round(cy - half)))
        x2 = min(w, int(round(cx + half)))
        y2 = min(h, int(round(cy + half)))
        
        crop = img[y1:y2, x1:x2]
        # In RGB, H-alpha prominence is R > G and R > B
        # Let's check prominence excess: R - 0.5*(G + B)
        r_chan = crop[:, :, 0].astype(np.float32)
        g_chan = crop[:, :, 1].astype(np.float32)
        b_chan = crop[:, :, 2].astype(np.float32)
        
        halpha_excess = r_chan - 0.5 * (g_chan + b_chan)
        
        # Save a normalized preview
        crop_s = cv2.resize(crop, (400, 400), interpolation=cv2.INTER_AREA)
        crop_s_log = np.log1p(crop_s.astype(np.float32))
        crop_s_log = ((crop_s_log - crop_s_log.min()) / (crop_s_log.max() - crop_s_log.min()) * 255).astype(np.uint8)
        crop_bgr = cv2.cvtColor(crop_s_log, cv2.COLOR_RGB2BGR)
        
        out_name = f"../samples/prominence_{fname.split('.')[0]}.jpg"
        cv2.imwrite(out_name, crop_bgr)
        print(f"Saved {out_name} for inspection")

if __name__ == "__main__":
    inspect_eruptions()

