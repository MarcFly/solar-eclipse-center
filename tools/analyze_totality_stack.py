import os
import glob
import cv2
import numpy as np
import tifffile

def analyze_totality_stack():
    # Load all processed images in 3_Totality if they exist, or process a sequence of them
    # Let's check what images exist in ../EclipseTry2/3_Totality/
    raw_files = sorted(glob.glob("../EclipseTry2/3_Totality/*.tif"))
    print(f"Total raw totality files: {len(raw_files)}")
    
    # Let's inspect 5 raw images across the totality sequence:
    # 0, 35, 75, 115, 151
    indices = [0, 35, 75, 115, 151]
    
    from detector import EclipseCircleDetector
    det = EclipseCircleDetector()
    
    for idx in indices:
        f = raw_files[idx]
        fname = os.path.basename(f)
        img = tifffile.imread(f)
        cx, cy, r = det.detect(img)
        print(f"Frame [{idx:3d}] {fname}: Moon center=({cx:6.1f}, {cy:6.1f}), R={r:.1f}")

if __name__ == "__main__":
    analyze_totality_stack()

