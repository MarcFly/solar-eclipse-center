import os
import glob
import cv2
import numpy as np
import tifffile

def inspect_processed_totality():
    files = sorted(glob.glob("../processed_eclipse/3_Totality/*.tif"))
    print(f"Total processed files in 3_Totality: {len(files)}")
    if not files:
        files = sorted(glob.glob("../processed_sample_test/3_Totality/*.tif"))
        print(f"Using sample files: {len(files)}")
        
    # Let's take files across the sequence
    test_indices = [0, 25, 50, 75, 100, 125, 150]
    test_indices = [i for i in test_indices if i < len(files)]
    
    # Check center (2000, 2000)
    # Let's crop a 1400x1400 patch centered at (2000, 2000) for each test image
    # and compare the angle of features around the limb
    ref_img = tifffile.imread(files[0])
    ref_crop = ref_img[1300:2700, 1300:2700]
    
    print("\nComparing frames against frame 0:")
    for idx in test_indices:
        f = files[idx]
        fname = os.path.basename(f)
        img = tifffile.imread(f)
        crop = img[1300:2700, 1300:2700]
        
        # Save a thumbnail of the crop to visually inspect
        sf = cv2.resize(crop, (400, 400), interpolation=cv2.INTER_AREA).astype(np.float32)
        slog = np.log1p(sf)
        slog = ((slog - slog.min()) / (slog.max() - slog.min()) * 255).astype(np.uint8)
        bgr = cv2.cvtColor(slog, cv2.COLOR_RGB2BGR)
        
        # Draw green circle at (200, 200) radius 586.5 * (400/1400) = 167.6 px
        cv2.circle(bgr, (200, 200), int(round(586.5 * 400 / 1400)), (0, 255, 0), 1)
        # Draw red center cross
        cv2.drawMarker(bgr, (200, 200), (0, 0, 255), cv2.MARKER_CROSS, 16, 1)
        
        out_name = f"../samples/stack_inspect_{idx:03d}_{fname.split('.')[0]}.jpg"
        cv2.imwrite(out_name, bgr)
        print(f"Saved {out_name}")

if __name__ == "__main__":
    inspect_processed_totality()

