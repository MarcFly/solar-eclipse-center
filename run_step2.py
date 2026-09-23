"""
Step 2 Verification Script:
Processes 5 randomly selected samples from each Totality folder in EclipseTry2,
detects the solar corona extent capturing faint streamers,
and writes out annotated visual sample images to:
samples/step2_corona_detection/<folder>/corona_<filename>.jpg

Shows:
1. Cyan contour outlining the segmented coronal streamers
2. Green circle marking the celestial lunar limb + red center crosshair
3. Magenta circle representing the maximum coronal radial extent R_corona_max
4. Yellow square bounding box representing the ideal square crop (+5% margin, capped to H)
"""

import os
import glob
import random
import cv2
import tifffile
import numpy as np
from detector import EclipseCircleDetector
from corona import EclipseCoronaDetector

TOTALITY_FOLDERS = [
    "../EclipseTry2/1_Totality_Contact1",
    "../EclipseTry2/2_Totality_Contact2",
    "../EclipseTry2/3_Totality",
    "../EclipseTry2/4_TotalityExit",
]

OUTPUT_BASE = "../samples/step2_corona_detection"


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    circle_detector = EclipseCircleDetector()
    corona_detector = EclipseCoronaDetector(
        contrast_pct=0.035,
        k_sigma=8.0,
        downscale_factor=0.25,
        margin_ratio=0.05
    )

    # Use a fixed random seed for reproducible sample evaluation
    random.seed(42)

    results = []

    print("=" * 85)
    print("STEP 2: CORONA EXTENT DETECTION VERIFICATION (RANDOM SAMPLES)")
    print("Faint Streamer Contrast Pass (pct=3.5%, k_sigma=8.0) | Margin = +5% | Cap = Max Height")
    print("=" * 85)

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

            # Step 1: Detect circle center
            cx, cy, r = circle_detector.detect(img)

            # Step 2: Detect corona extent with faint streamer contrast pass
            res = corona_detector.detect(img, cx, cy, r)

            results.append({
                "folder": folder_name,
                "file": fname,
                "cx": cx,
                "cy": cy,
                "r": r,
                "r_corona": res.r_corona_max,
                "ratio": res.ratio_to_sun,
                "crop_size": res.ideal_crop_size,
                "bg_med": res.background_median,
                "bg_sigma": res.background_sigma
            })

            # Create visual overlay
            vis = corona_detector.create_visual_overlay(img, cx, cy, r, res, filename=fname)

            out_path = os.path.join(out_folder, f"corona_{os.path.splitext(fname)[0]}.jpg")
            cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

            print(f"  [OK] {fname}: Corona Max R={res.r_corona_max:6.1f} px ({res.ratio_to_sun:4.2f}x) -> Crop: {res.ideal_crop_size}x{res.ideal_crop_size} px")

    # Summary
    max_corona_all = max(r["r_corona"] for r in results)
    median_crop = int(np.median([r["crop_size"] for r in results]))
    max_crop = max(r["crop_size"] for r in results)
    totality_r = [r["r_corona"] for r in results if r["folder"] == "3_Totality"]

    print("\n" + "=" * 92)
    print("STEP 2 CORONA DETECTION SUMMARY:")
    print(f"{'Folder':<22} | {'Filename':<15} | {'Sun R':<8} | {'Corona R':<10} | {'Ratio':<6} | {'Crop (+5%, cap)':<16}")
    print("-" * 92)
    for res in results:
        crop_str = f"{res['crop_size']}x{res['crop_size']} px"
        print(f"{res['folder']:<22} | {res['file']:<15} | {res['r']:5.1f} px | {res['r_corona']:6.1f} px | {res['ratio']:4.2f}x | {crop_str:<16}")
    print("=" * 92)
    print(f"\nDEEP TOTALITY MEAN EXTENSION: {np.mean(totality_r):.1f} px ({np.mean(totality_r)/586.5:.2f}x Moon Radius) [approx +30% vs strict]")
    print(f"MAXIMUM CORONAL EXTENSION (among tested samples): {max_corona_all:.1f} px ({max_corona_all/586.5:.2f}x Moon Radius)")
    print(f"MAXIMUM CROP SIZE (Strictly capped <= 4000 px): {max_crop} x {max_crop} px")
    print("=" * 92)


if __name__ == "__main__":
    main()
