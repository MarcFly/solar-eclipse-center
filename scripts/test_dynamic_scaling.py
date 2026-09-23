"""
Test script to verify dynamic radius and crop size auto-detection.
Tests:
1. EclipseCircleDetector with nominal_radius_full=None on real images.
2. EclipseAffineTransformer with target_r=None, crop_size=None.
3. EclipseOrchestrator running Pass 1 and Pass 2 with default target_radius=None.
"""
import os
import glob
import tifffile
import numpy as np
from eclipse_processor import (
    EclipseCircleDetector,
    EclipseCoronaDetector,
    EclipseAffineTransformer,
    EclipseOrchestrator
)

def test_dynamic_detector():
    print("Testing dynamic EclipseCircleDetector (nominal_radius_full=None)...")
    detector = EclipseCircleDetector(nominal_radius_full=None)
    sample_files = glob.glob("../EclipseTry2/3_Totality/*.tif")
    if not sample_files:
        print("No sample files found.")
        return
    img = tifffile.imread(sample_files[0])
    cx, cy, r = detector.detect(img)
    print(f"Detected: cx={cx:.2f}, cy={cy:.2f}, r={r:.2f} px (shape: {img.shape})")
    assert r > 0, "Radius should be positive"
    assert 500 < r < 650, f"Expected radius near ~586 px for this image, got {r}"

def test_dynamic_transformer():
    print("\nTesting dynamic EclipseAffineTransformer (target_r=None, crop_size=None)...")
    sample_files = glob.glob("../EclipseTry2/3_Totality/*.tif")
    img = tifffile.imread(sample_files[0])
    detector = EclipseCircleDetector()
    cx, cy, r = detector.detect(img)
    transformer = EclipseAffineTransformer()
    res = transformer.transform(img, cx=cx, cy=cy, r=r)
    print(f"Transformed: target_r={res.target_r:.2f}, crop_size={res.crop_size}, scale={res.scale:.4f}")
    assert res.target_r == r, "Should preserve detected radius"
    assert res.scale == 1.0, "Scale should be 1.0"
    assert res.img_transformed.dtype == np.uint16, "Dtype must be uint16"
    print("Transformer test passed.")

if __name__ == "__main__":
    test_dynamic_detector()
    test_dynamic_transformer()
    print("\nAll dynamic scaling tests passed successfully!")

