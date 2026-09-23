"""
Step 1 Verification Script:
Processes 5 randomly selected samples from each Totality folder in EclipseTry2,
detects the Moon circle using the flare-immune detector, and writes out annotated
visual sample images to samples/step1_circle_detection/<folder>/ including:
1. Full overview images with detected circle and center crosshair.
2. Close-up zoomed limb crops with 4 cardinal tick marks (top, bottom, left, right)
   to easily verify limb boundary alignment against flares/prominences.
"""

import os
import glob
import random
import cv2
import tifffile
import numpy as np
from detector import EclipseCircleDetector

# Focusing strictly on totality steps as requested
TOTALITY_FOLDERS = [
    "../EclipseTry2/1_Totality_Contact1",
    "../EclipseTry2/2_Totality_Contact2",
    "../EclipseTry2/3_Totality",
    "../EclipseTry2/4_TotalityExit",
]

OUTPUT_BASE = "../samples/step1_circle_detection"


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    detector = EclipseCircleDetector()

    # Set random seed for reproducibility while selecting random samples
    random.seed(42)

    results = []

    print("=" * 80)
    print("STEP 1: TOTALITY PHASES CIRCLE DETECTION VERIFICATION (RANDOM SAMPLES)")
    print("=" * 80)

    for folder_rel in TOTALITY_FOLDERS:
        folder_name = os.path.basename(os.path.normpath(folder_rel))
        tifs = sorted(glob.glob(os.path.join(folder_rel, "*.tif")))

        if not tifs:
            print(f"No TIFFs found in {folder_rel}")
            continue

        out_folder = os.path.join(OUTPUT_BASE, folder_name)
        zoom_folder = os.path.join(out_folder, "zoomed_limb")
        os.makedirs(out_folder, exist_ok=True)
        os.makedirs(zoom_folder, exist_ok=True)

        n_samples = min(5, len(tifs))
        sample_files = random.sample(tifs, n_samples)

        print(f"\nProcessing folder: {folder_name} ({len(tifs)} total files, randomly testing {n_samples} samples)...")

        for filepath in sample_files:
            fname = os.path.basename(filepath)
            img = tifffile.imread(filepath)

            # Detect circle at full resolution
            cx, cy, r = detector.detect(img)
            results.append({
                "folder": folder_name,
                "file": fname,
                "shape": img.shape,
                "cx": cx,
                "cy": cy,
                "r": r
            })

            # 1. Full image overview (downscaled for fast viewing)
            preview_scale = 0.25
            pw = int(round(img.shape[1] * preview_scale))
            ph = int(round(img.shape[0] * preview_scale))
            small = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)

            if small.ndim == 3:
                s8 = (small / 256).astype(np.uint8) if small.dtype == np.uint16 else small.astype(np.uint8)
                bgr = cv2.cvtColor(s8, cv2.COLOR_RGB2BGR)
            else:
                s8 = (small / 256).astype(np.uint8) if small.dtype == np.uint16 else small.astype(np.uint8)
                bgr = cv2.cvtColor(s8, cv2.COLOR_GRAY2BGR)

            pcx = int(round(cx * preview_scale))
            pcy = int(round(cy * preview_scale))
            pr = int(round(r * preview_scale))

            # Green circle on limb
            cv2.circle(bgr, (pcx, pcy), pr, (0, 255, 0), 2, cv2.LINE_AA)
            # Red center crosshair
            cv2.drawMarker(bgr, (pcx, pcy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
            label = f"{fname}: cx={cx:.1f}, cy={cy:.1f}, r={r:.1f}px"
            cv2.putText(bgr, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

            out_path = os.path.join(out_folder, f"detected_{os.path.splitext(fname)[0]}.jpg")
            cv2.imwrite(out_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

            # 2. Close-up Zoomed Limb Crop
            pad = 80
            x1 = max(0, int(round(cx - r - pad)))
            y1 = max(0, int(round(cy - r - pad)))
            x2 = min(img.shape[1], int(round(cx + r + pad)))
            y2 = min(img.shape[0], int(round(cy + r + pad)))

            crop = img[y1:y2, x1:x2]
            # Use log stretch to reveal dark moon disk and faint corona simultaneously
            cf = crop.astype(np.float32)
            clog = np.log1p(cf)
            clog = ((clog - clog.min()) / (clog.max() - clog.min()) * 255).astype(np.uint8)
            crop_bgr = cv2.cvtColor(clog, cv2.COLOR_RGB2BGR) if crop.ndim == 3 else cv2.cvtColor(clog, cv2.COLOR_GRAY2BGR)

            ccx = int(round(cx - x1))
            ccy = int(round(cy - y1))
            cr = int(round(r))

            # Draw green limb circle
            cv2.circle(crop_bgr, (ccx, ccy), cr, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.drawMarker(crop_bgr, (ccx, ccy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

            # Draw cardinal alignment dots
            cv2.circle(crop_bgr, (ccx, ccy - cr), 4, (0, 255, 255), -1)  # Top
            cv2.circle(crop_bgr, (ccx, ccy + cr), 4, (0, 255, 255), -1)  # Bottom
            cv2.circle(crop_bgr, (ccx - cr, ccy), 4, (0, 255, 255), -1)  # Left
            cv2.circle(crop_bgr, (ccx + cr, ccy), 4, (0, 255, 255), -1)  # Right

            zoom_path = os.path.join(zoom_folder, f"zoom_{os.path.splitext(fname)[0]}.jpg")
            cv2.imwrite(zoom_path, crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

            print(f"  [OK] {fname}: cx={cx:6.1f}, cy={cy:6.1f}, r={r:5.1f} px -> {out_path}")

    print("\n" + "=" * 80)
    print("STEP 1 DETECTION SUMMARY (RANDOMLY SELECTED SAMPLES):")
    print(f"{'Folder':<22} | {'Filename':<18} | {'Shape':<14} | {'Center (cx, cy)':<18} | {'Radius':<8}")
    print("-" * 88)
    for res in results:
        center_str = f"({res['cx']:.1f}, {res['cy']:.1f})"
        shape_str = f"{res['shape'][0]}x{res['shape'][1]}"
        print(f"{res['folder']:<22} | {res['file']:<18} | {shape_str:<14} | {center_str:<18} | {res['r']:5.1f} px")
    print("=" * 80)


if __name__ == "__main__":
    main()
