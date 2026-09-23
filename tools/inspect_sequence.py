import os
import glob
import cv2
import numpy as np
import tifffile

def inspect_sequence():
    files = sorted(glob.glob("../EclipseTry2/3_Totality/*.tif"))
    print(f"Total files: {len(files)}")
    
    # Inspect first 20 files
    print(f"{'Index':<5} | {'Filename':<14} | {'Mean':<8} | {'Max':<6} | {'Corner Med':<10}")
    print("-" * 55)
    for i in range(20):
        f = files[i]
        fname = os.path.basename(f)
        img = tifffile.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        corner = np.median(gray[:40, :40])
        print(f"{i:5d} | {fname:<14} | {np.mean(gray):8.1f} | {np.max(gray):6d} | {corner:10.1f}")

if __name__ == "__main__":
    inspect_sequence()

