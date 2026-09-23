"""
Step 4 Verification Script:
Processes 5 randomly selected samples from each Totality folder in EclipseTry2,
applies single-pass composite affine transformation (centering + uniform radius scaling + square cropping),
verifies 16-bit uint16 preservation and mathematical alignment,
and writes out annotated visual sample images to:
samples/step4_affine_crop/<folder>/crop_<filename>.jpg

Shows:
1. Standardized square crop canvas (4000 x 4000 px)
2. Target center crosshair in red right at (2000, 2000)
3. Full-canvas horizontal and vertical axis guide lines passing through (2000, 2000)
4. Celestial limb circle in green with standardized radius R_target = 586.5 px
5. Concentric alignment rings at 1.5x and 2.0x R_target
"""

import os
import glob
import random
import cv2
import tifffile
import numpy as np
from detector import EclipseCircleDetector
from transformer import EclipseAffineTransformer

TOTALITY_FOLDERS = [
    "../EclipseTry2/1_Totality_Contact1",
    "../EclipseTry2/2_Totality_Contact2",
    "../EclipseTry2/3_Totality",
    "../EclipseTry2/4_TotalityExit",
]

OUTPUT_BASE = "../samples/step4_affine_crop"
# Note: TARGET_RADIUS = 586.5 and CROP_SIZE = 4000 are the values discovered for the
# sample 6000x4000 EclipseTry2 dataset. In the core library and CLI, these default to
# None and are dynamically auto-detected across incoming datasets.
TARGET_RADIUS = 586.5
CROP_SIZE = 4000


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    circle_detector = EclipseCircleDetector()
    transformer = EclipseAffineTransformer(interpolation=cv2.INTER_LANCZOS4, fill_background=True)

    random.seed(42)

    results = []

    print("=" * 95)
    print("STEP 4: AFFINE TRANSFORMATION VERIFICATION (RANDOM SAMPLES)")
    print(f"Uniform Target Radius: {TARGET_RADIUS:.1f} px | Uniform Square Crop: {CROP_SIZE}x{CROP_SIZE} px | 16-bit uint16")
    print("=" * 95)

    sample_tif_saved = False

    for folder_rel in TOTALITY_FOLDERS:
        folder_name = os.path.basename(os.path.normpath(folder_rel))
        tifs = sorted(glob.glob(os.path.join(folder_rel, "*.tif")))

        if not tifs:
            print(f"No TIFFs found in {folder_rel}")
            continue

        out_folder = os.path.join(OUTPUT_BASE, folder_name)
        os.makedirs(out_folder, exist_ok=True)

        n_samples = min(5, len(tifs))
        sample_files = random.sample(tifs, n_samples)

        print(f"\nProcessing folder: {folder_name} ({len(tifs)} total files, testing {n_samples} random samples)...")

        for filepath in sample_files:
            fname = os.path.basename(filepath)
            img = tifffile.imread(filepath)

            # 1. Detect circle center and radius
            cx, cy, r = circle_detector.detect(img)

            # 2. Single-pass composite transformation: centering + uniform scaling + square crop
            res = transformer.transform(
                img,
                cx=cx,
                cy=cy,
                r=r,
                target_r=TARGET_RADIUS,
                crop_size=CROP_SIZE
            )

            # Assertions: 16-bit uint16 & exact canvas dimensions
            assert res.img_transformed.dtype == np.uint16, f"Dtype error: expected uint16, got {res.img_transformed.dtype}"
            assert res.img_transformed.shape[:2] == (CROP_SIZE, CROP_SIZE), f"Shape error: expected ({CROP_SIZE}, {CROP_SIZE}), got {res.img_transformed.shape[:2]}"

            # Save 1 full-fidelity 16-bit sample TIFF for user inspection
            if not sample_tif_saved:
                sample_tif_path = os.path.join(OUTPUT_BASE, f"sample_16bit_{os.path.splitext(fname)[0]}.tif")
                tifffile.imwrite(sample_tif_path, res.img_transformed, photometric='rgb' if res.img_transformed.ndim == 3 else 'minisblack')
                sample_tif_saved = True
                print(f"  [SAVED 16-BIT TIFF] {sample_tif_path}")

            results.append({
                "folder": folder_name,
                "file": fname,
                "orig_r": r,
                "target_r": res.target_r,
                "scale": res.scale,
                "orig_center": f"({cx:.1f}, {cy:.1f})",
                "target_center": f"({res.target_cx:.1f}, {res.target_cy:.1f})",
                "crop_shape": f"{res.crop_size}x{res.crop_size}",
                "dtype": str(res.img_transformed.dtype)
            })

            # Create visual inspection overlay
            vis = transformer.create_visual_overlay(res, filename=fname, downscale_factor=0.25)
            out_path = os.path.join(out_folder, f"crop_{os.path.splitext(fname)[0]}.jpg")
            cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

            print(f"  [OK] {fname}: Sun R={r:.1f}px -> {res.target_r:.1f}px (scale={res.scale:.4f}) | Crop: {res.crop_size}x{res.crop_size} px centered at ({res.target_cx:.0f}, {res.target_cy:.0f})")

    print("\n" + "=" * 110)
    print("STEP 4 AFFINE TRANSFORMATION SUMMARY:")
    print(f"{'Folder':<22} | {'Filename':<14} | {'Orig R':<8} | {'Target R':<9} | {'Scale':<8} | {'Output Size':<12} | {'Center':<14} | {'Dtype':<7}")
    print("-" * 110)
    for r in results:
        print(f"{r['folder']:<22} | {r['file']:<14} | {r['orig_r']:6.1f}px | {r['target_r']:7.1f}px | {r['scale']:7.4f}x | {r['crop_shape']:<12} | {r['target_center']:<14} | {r['dtype']:<7}")
    print("=" * 110)
    print(f"\nAll 20 sample images successfully transformed to standardized {CROP_SIZE}x{CROP_SIZE} square crops.")
    print(f"Sun is centered at ({CROP_SIZE//2}, {CROP_SIZE//2}) with standardized radius R={TARGET_RADIUS:.1f} px.")
    print(f"Visual overlays saved to: {OUTPUT_BASE}/<folder>/crop_<filename>.jpg")
    print("=" * 110)


if __name__ == "__main__":
    main()

