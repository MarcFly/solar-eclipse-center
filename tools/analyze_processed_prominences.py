import cv2
import numpy as np
import tifffile

def analyze_processed_prominences():
    f1 = "../processed_eclipse/3_Totality/_DSF5184.tif"
    f2 = "../processed_eclipse/3_Totality/_DSF5259.tif"
    f3 = "../processed_eclipse/3_Totality/_DSF5335.tif"
    
    img1 = tifffile.imread(f1)
    img2 = tifffile.imread(f2)
    img3 = tifffile.imread(f3)
    
    # In processed images, center is at (2000, 2000) and R is 586.5
    cx, cy = 2000.0, 2000.0
    r = 586.5
    
    # Let's inspect the 3 images in 3 prominence regions:
    # 1. Left prominence (~9 o'clock, x < 2000, y ~ 2000)
    # 2. Top prominence (~12 o'clock, x ~ 2000, y < 2000)
    # 3. Right prominence (~3 o'clock, x > 2000, y ~ 2000)
    
    regions = {
        "Left (~9h)": (int(cx - r * 1.15), int(cy - 150), int(cx - r * 0.95), int(cy + 150)),
        "Top (~12h)": (int(cx - 150), int(cy - r * 1.15), int(cx + 150), int(cy - r * 0.95)),
        "Right (~3h)": (int(cx + r * 0.95), int(cy - 150), int(cx + r * 1.15), int(cy + 150)),
    }
    
    for name, (x1, y1, x2, y2) in regions.items():
        print(f"\n=== Region: {name} (Box: [{x1}:{x2}, {y1}:{y2}]) ===")
        for fname, img in [("_DSF5184", img1), ("_DSF5259", img2), ("_DSF5335", img3)]:
            crop = img[y1:y2, x1:x2]
            # Red excess
            prom = np.maximum(0, crop[:, :, 0].astype(np.float32) - 0.5 * (crop[:, :, 1].astype(np.float32) + crop[:, :, 2].astype(np.float32)))
            max_val = np.max(prom)
            # Find centroid of top 1% brightest prominence pixels
            thresh = max_val * 0.7
            ys, xs = np.where(prom > thresh)
            if len(xs) > 0:
                weights = prom[ys, xs]
                centroid_x = x1 + np.average(xs, weights=weights)
                centroid_y = y1 + np.average(ys, weights=weights)
                dx = centroid_x - cx
                dy = centroid_y - cy
                dist = np.sqrt(dx**2 + dy**2)
                angle_deg = np.rad2deg(np.arctan2(dy, dx)) % 360.0
                print(f"  {fname}: centroid=({centroid_x:6.2f}, {centroid_y:6.2f}) | dist={dist:5.1f} px | angle={angle_deg:6.2f}° | max={max_val:5.0f}")
            else:
                print(f"  {fname}: No prominence detected above threshold (max={max_val:5.0f})")

if __name__ == "__main__":
    analyze_processed_prominences()

