import os
import cv2
import numpy as np
import tifffile

def compare_three():
    f1 = "../processed_eclipse/3_Totality/_DSF5184.tif"
    f2 = "../processed_eclipse/3_Totality/_DSF5259.tif"
    f3 = "../processed_eclipse/3_Totality/_DSF5335.tif"
    
    img1 = tifffile.imread(f1)
    img2 = tifffile.imread(f2)
    img3 = tifffile.imread(f3)
    
    print(f"Loaded: img1={img1.shape}, img2={img2.shape}, img3={img3.shape}")
    
    # Crop central 1600x1600 around (2000, 2000)
    c = 2000
    w = 800
    crop1 = img1[c-w:c+w, c-w:c+w]
    crop2 = img2[c-w:c+w, c-w:c+w]
    crop3 = img3[c-w:c+w, c-w:c+w]
    
    # Save RGB previews with high visibility of corona/prominences
    for name, crop in [("f1_5184", crop1), ("f2_5259", crop2), ("f3_5335", crop3)]:
        sf = cv2.resize(crop, (800, 800), interpolation=cv2.INTER_AREA).astype(np.float32)
        slog = np.log1p(sf)
        slog = ((slog - slog.min()) / (slog.max() - slog.min()) * 255).astype(np.uint8)
        bgr = cv2.cvtColor(slog, cv2.COLOR_RGB2BGR)
        cv2.circle(bgr, (400, 400), int(round(586.5 * 800 / 1600)), (0, 255, 0), 1)
        cv2.drawMarker(bgr, (400, 400), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
        cv2.imwrite(f"../samples/{name}_central.jpg", bgr)
        print(f"Saved ../samples/{name}_central.jpg")
        
    # Also save a color-coded overlay: img1 in Red, img2 in Green, img3 in Blue!
    # If aligned and identical, it will look monochrome/white.
    # Where there is rotation/motion, there will be bright color fringing!
    gray1 = cv2.cvtColor(crop1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(crop2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray3 = cv2.cvtColor(crop3, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Normalize each channel by median
    def norm(g):
        lg = np.log1p(g)
        return (lg - lg.min()) / (lg.max() - lg.min()) * 255.0
        
    n1 = norm(gray1)
    n2 = norm(gray2)
    n3 = norm(gray3)
    
    rgb_composite = np.zeros((1600, 1600, 3), dtype=np.uint8)
    rgb_composite[:, :, 0] = n3.astype(np.uint8) # Blue: f3
    rgb_composite[:, :, 1] = n2.astype(np.uint8) # Green: f2
    rgb_composite[:, :, 2] = n1.astype(np.uint8) # Red: f1
    
    comp_small = cv2.resize(rgb_composite, (800, 800), interpolation=cv2.INTER_AREA)
    cv2.circle(comp_small, (400, 400), int(round(586.5 * 800 / 1600)), (255, 255, 255), 1)
    cv2.imwrite("../samples/rgb_motion_composite.jpg", comp_small)
    print("Saved ../samples/rgb_motion_composite.jpg")

if __name__ == "__main__":
    compare_three()

