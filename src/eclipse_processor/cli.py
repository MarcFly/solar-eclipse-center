"""
Solar Eclipse Bulk Orchestrator.
Processes eclipse TIFF images in bulk across folders:
Pass 1: Circle and coronal extent detection across all images to determine global maximum coronal extension,
        target solar radius, and uniform square crop dimensions.
Pass 2: Single-pass composite affine transformation (sub-pixel centering, uniform solar radius scaling,
        and square cropping) exporting full-fidelity 16-bit uint16 TIFF images.

Supports:
- Root folders with subdirectories or explicit list of folders/files
- Sampling mode (--sample N) for quick verification runs
- Multithreaded execution for high throughput
- Optional visual inspection overlays (--generate-overlays)
- Lossless compression options (--compression deflate/lzw/none)
"""

import os
import sys
import glob
import time
import argparse
import random
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import tifffile
from tqdm import tqdm

from eclipse_processor.core.detector import EclipseCircleDetector
from eclipse_processor.core.corona import EclipseCoronaDetector
from eclipse_processor.core.transformer import EclipseAffineTransformer, TransformationResult
from eclipse_processor.core.rotator import EclipseRotationAligner, RotationResult


def find_image_tasks(inputs: List[str], sample_n: Optional[int] = None) -> List[Tuple[str, str, str]]:
    """
    Discovers all TIFF images from given input paths.
    Returns list of (folder_name, file_name, full_path).
    If sample_n is specified, samples up to sample_n images per folder.
    """
    folder_to_files: Dict[str, List[str]] = {}

    for inp in inputs:
        inp_path = os.path.abspath(inp)
        if os.path.isfile(inp_path) and inp_path.lower().endswith(('.tif', '.tiff')):
            folder_name = os.path.basename(os.path.dirname(inp_path))
            folder_to_files.setdefault(folder_name, []).append(inp_path)
        elif os.path.isdir(inp_path):
            # Check if this directory directly has tifs
            direct_tifs = sorted(glob.glob(os.path.join(inp_path, "*.tif")) + glob.glob(os.path.join(inp_path, "*.tiff")))
            if direct_tifs:
                folder_name = os.path.basename(os.path.normpath(inp_path))
                folder_to_files.setdefault(folder_name, []).extend(direct_tifs)
            
            # Check subdirectories
            for root, dirs, files in os.walk(inp_path):
                tifs = sorted([os.path.join(root, f) for f in files if f.lower().endswith(('.tif', '.tiff'))])
                if tifs:
                    folder_name = os.path.basename(os.path.normpath(root))
                    folder_to_files.setdefault(folder_name, []).extend(tifs)

    tasks: List[Tuple[str, str, str]] = []
    for folder_name, filepaths in folder_to_files.items():
        # Deduplicate preserving order
        seen = set()
        unique_fps = []
        for fp in filepaths:
            if fp not in seen:
                seen.add(fp)
                unique_fps.append(fp)

        if sample_n and sample_n > 0 and len(unique_fps) > sample_n:
            # Evenly space sample images across the folder
            indices = np.linspace(0, len(unique_fps) - 1, sample_n, dtype=int)
            selected_fps = [unique_fps[i] for i in sorted(list(set(indices)))]
        else:
            selected_fps = unique_fps

        for fp in selected_fps:
            tasks.append((folder_name, os.path.basename(fp), fp))

    return tasks


class EclipseOrchestrator:
    def __init__(self,
                 target_radius: Optional[float] = None,
                 margin_ratio: float = 0.05,
                 cap_height: bool = True,
                 workers: int = 4,
                 compression: str = "none",
                 align_rotation: bool = False):
        self.target_radius = target_radius
        self.margin_ratio = margin_ratio
        self.cap_height = cap_height
        self.workers = workers
        self.compression = compression
        self.align_rotation = align_rotation

        self.circle_detector = EclipseCircleDetector()
        self.corona_detector = EclipseCoronaDetector(
            contrast_pct=0.035,
            k_sigma=8.0,
            downscale_factor=0.25,
            margin_ratio=self.margin_ratio
        )
        self.transformer = EclipseAffineTransformer(
            interpolation=cv2.INTER_LANCZOS4,
            fill_background=True
        )
        self.rotator = EclipseRotationAligner() if self.align_rotation else None

    def analyze_single_image(self, task: Tuple[str, str, str]) -> Dict:
        folder_name, fname, fpath = task
        img = tifffile.imread(fpath)
        h, w = img.shape[:2]

        cx, cy, r = self.circle_detector.detect(img)
        res_cor = self.corona_detector.detect(img, cx, cy, r)

        return {
            "folder": folder_name,
            "filename": fname,
            "filepath": fpath,
            "height": h,
            "width": w,
            "cx": cx,
            "cy": cy,
            "r": r,
            "r_corona": res_cor.r_corona_max,
            "ratio_to_sun": res_cor.ratio_to_sun,
            "ideal_crop": res_cor.ideal_crop_size,
            "bg_med": res_cor.background_median,
            "bg_sig": res_cor.background_sigma
        }

    def run_pass1(self, tasks: List[Tuple[str, str, str]]) -> Tuple[List[Dict], float, int]:
        """
        Pass 1: Detects circles and coronal extents across all images.
        Determines uniform target radius and uniform square crop size.
        """
        print("\n" + "=" * 95)
        print(f"PASS 1: GLOBAL ANALYSIS & CORONA EXTENT ESTIMATION ({len(tasks)} images)")
        print("=" * 95)

        results = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self.analyze_single_image, t): t for t in tasks}
            with tqdm(total=len(tasks), desc="Pass 1 Analysis", unit="img") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    results.append(res)
                    pbar.update(1)

        # Sort results by folder and filename
        results.sort(key=lambda x: (x["folder"], x["filename"]))

        # Determine target radius
        detected_radii = [r["r"] for r in results if r["r"] > 0]
        if self.target_radius:
            target_r = float(self.target_radius)
        elif detected_radii:
            target_r = float(np.median(detected_radii))
        else:
            max_h_sample = max((r["height"] for r in results), default=4000)
            target_r = float(max_h_sample * 0.15)

        # Determine uniform crop size
        max_corona = max(r["r_corona"] for r in results)
        max_h = max(r["height"] for r in results)

        raw_crop = int(round(2.0 * (1.0 + self.margin_ratio) * max_corona))
        if self.cap_height:
            crop_size = min(raw_crop, max_h)
        else:
            crop_size = raw_crop

        if crop_size % 2 != 0:
            crop_size -= 1

        # Optional Step 4.5: Multi-Point Prominence Rotation Alignment
        if self.align_rotation and self.rotator:
            print("\n" + "-" * 95)
            print("STEP 4.5: MULTI-POINT PROMINENCE ROTATION ALIGNMENT (Top, Right, Left Anchors)")
            print("-" * 95)
            
            # Group by folder
            folder_groups: Dict[str, List[Dict]] = {}
            for r in results:
                folder_groups.setdefault(r["folder"], []).append(r)

            for folder, f_results in folder_groups.items():
                print(f"  Estimating rotation sequence for folder: {folder} ({len(f_results)} frames)...")
                file_paths = [r["filepath"] for r in f_results]
                rot_results = self.rotator.process_sequence(file_paths, detector=self.circle_detector)
                for r_info, rot_res in zip(f_results, rot_results):
                    r_info["rotation_deg"] = rot_res.rotation_deg
                    r_info["rot_conf"] = rot_res.confidence
                    r_info["rot_res"] = rot_res
                
                valid_confs = [r["rot_conf"] for r in f_results if r["rot_conf"] > 0]
                avg_conf = np.mean(valid_confs) if valid_confs else 0.0
                rot_range = max(r["rotation_deg"] for r in f_results) - min(r["rotation_deg"] for r in f_results)
                print(f"    Completed: Avg Conf={avg_conf:.3f}, Total Field Rotation Range={rot_range:.3f}°")
        else:
            for r in results:
                r["rotation_deg"] = 0.0
                r["dx"] = 0.0
                r["dy"] = 0.0

        print("\n" + "-" * 95)
        print("PASS 1 SUMMARY RESULTS:")
        print(f"  Total Images Analyzed:       {len(results)}")
        print(f"  Standard Target Sun Radius:  {target_r:.1f} px")
        print(f"  Global Maximum Corona Extent:{max_corona:.1f} px ({max_corona/target_r:.2f}x Sun Radius)")
        print(f"  Calculated Raw Crop (+{self.margin_ratio*100:.0f}%):   {raw_crop} x {raw_crop} px")
        print(f"  Final Uniform Crop (Capped): {crop_size} x {crop_size} px (Canvas Height: {max_h} px)")
        if self.align_rotation:
            print(f"  Rotation Alignment:          ENABLED (Multi-point prominence tracking)")
        else:
            print(f"  Rotation Alignment:          DISABLED (Preserving original camera orientation)")
        print("-" * 95)

        return results, target_r, crop_size

    def transform_and_export_single_image(self,
                                          info: Dict,
                                          target_r: float,
                                          crop_size: int,
                                          output_base: str,
                                          generate_overlays: bool = False) -> Dict:
        fpath = info["filepath"]
        folder = info["folder"]
        fname = info["filename"]

        img = tifffile.imread(fpath)

        res = self.transformer.transform(
            img,
            cx=info["cx"],
            cy=info["cy"],
            r=info["r"],
            target_r=target_r,
            crop_size=crop_size,
            rotation_deg=info.get("rotation_deg", 0.0)
        )

        out_folder = os.path.join(output_base, folder)
        os.makedirs(out_folder, exist_ok=True)
        out_tif_path = os.path.join(out_folder, fname)

        # Write 16-bit uint16 TIFF
        photometric = 'rgb' if res.img_transformed.ndim == 3 else 'minisblack'
        comp = None if self.compression == "none" else self.compression
        tifffile.imwrite(out_tif_path, res.img_transformed, photometric=photometric, compression=comp)

        # Optional visual overlay
        if generate_overlays:
            overlay_folder = os.path.join(output_base, "overlays", folder)
            os.makedirs(overlay_folder, exist_ok=True)
            vis = self.transformer.create_visual_overlay(res, filename=fname, downscale_factor=0.25)
            overlay_path = os.path.join(overlay_folder, f"overlay_{os.path.splitext(fname)[0]}.jpg")
            cv2.imwrite(overlay_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

        return {
            "folder": folder,
            "filename": fname,
            "out_path": out_tif_path,
            "scale": res.scale,
            "crop_size": crop_size,
            "bytes": os.path.getsize(out_tif_path)
        }

    def run_pass2(self,
                  analysis_results: List[Dict],
                  target_r: float,
                  crop_size: int,
                  output_base: str,
                  generate_overlays: bool = False) -> List[Dict]:
        """
        Pass 2: Transforms, standardizes, square-crops, and exports all 16-bit TIFF images.
        """
        print("\n" + "=" * 95)
        print(f"PASS 2: EXPORTING 16-BIT STANDARDIZED TIFF IMAGES ({len(analysis_results)} images)")
        print(f"Output Directory: {os.path.abspath(output_base)}")
        print(f"Uniform Crop Size: {crop_size}x{crop_size} px | Target Solar Radius: {target_r:.1f} px | Dtype: uint16")
        print("=" * 95)

        export_results = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(
                    self.transform_and_export_single_image,
                    info,
                    target_r,
                    crop_size,
                    output_base,
                    generate_overlays
                ): info for info in analysis_results
            }
            with tqdm(total=len(analysis_results), desc="Pass 2 Export", unit="img") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    export_results.append(res)
                    pbar.update(1)

        elapsed = time.time() - start_time
        total_mb = sum(r["bytes"] for r in export_results) / (1024 * 1024)

        print("\n" + "=" * 95)
        print("PROCESSING COMPLETE!")
        print(f"  Successfully Processed: {len(export_results)} images")
        print(f"  Total Exported Data:    {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")
        print(f"  Total Elapsed Time:     {elapsed:.2f} s ({len(export_results)/elapsed:.2f} images/s)")
        print(f"  Output Directory:       {os.path.abspath(output_base)}")
        print("=" * 95 + "\n")

        return export_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solar Eclipse Bulk TIFF Orchestrator: Centering, Uniform Resizing, and Square Cropping"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["../EclipseTry2"],
        help="Input folder(s), image file(s), or parent directory containing eclipse folders (default: ../EclipseTry2)"
    )
    parser.add_argument(
        "-o", "--output",
        default="../processed_eclipse",
        help="Destination directory for processed 16-bit TIFFs (default: ../processed_eclipse)"
    )
    parser.add_argument(
        "-s", "--sample",
        type=int,
        default=None,
        help="Sample mode: processes up to N evenly spaced sample images per folder (useful for quick trial verification)"
    )
    parser.add_argument(
        "-c", "--crop-size",
        type=int,
        default=None,
        help="Explicit square crop size in pixels (default: calculated automatically from max corona extent, capped to image height)"
    )
    parser.add_argument(
        "-r", "--target-radius",
        type=float,
        default=None,
        help="Standardized solar/lunar radius in pixels across all output images (default: None, auto-detected from dataset median)"
    )
    parser.add_argument(
        "-m", "--margin",
        type=float,
        default=0.05,
        help="Margin ratio above max coronal extension (default: 0.05 for +5%%)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads (default: 4)"
    )
    parser.add_argument(
        "--no-cap-height",
        action="store_true",
        help="Do not cap the crop size to the canvas height (default: cap to height)"
    )
    parser.add_argument(
        "--generate-overlays",
        action="store_true",
        help="Also export downscaled visual verification overlays (.jpg) alongside TIFFs"
    )
    parser.add_argument(
        "--compression",
        choices=["none", "deflate", "lzw", "zstd"],
        default="none",
        help="Lossless TIFF compression algorithm (default: none for fastest writing)"
    )
    parser.add_argument(
        "--align-rotation",
        action="store_true",
        default=False,
        help="Enable multi-point prominence rotation alignment and solar registration (ideal for stacking)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = find_image_tasks(args.inputs, sample_n=args.sample)

    if not tasks:
        print(f"Error: No TIFF images found in inputs: {args.inputs}")
        sys.exit(1)

    print("\n" + "=" * 95)
    print("SOLAR ECLIPSE BULK ORCHESTRATOR")
    print(f"Inputs: {args.inputs}")
    print(f"Found {len(tasks)} images across {len(set(t[0] for t in tasks))} folder(s)")
    if args.sample:
        print(f"SAMPLING MODE ACTIVE: {args.sample} images per folder")
    if args.align_rotation:
        print("ROTATION ALIGNMENT: ENABLED (Multi-point prominence tracking)")
    print("=" * 95)

    orchestrator = EclipseOrchestrator(
        target_radius=args.target_radius,
        margin_ratio=args.margin,
        cap_height=not args.no_cap_height,
        workers=args.workers,
        compression=args.compression,
        align_rotation=args.align_rotation
    )

    # Pass 1: Global Analysis
    analysis_results, target_r, auto_crop_size = orchestrator.run_pass1(tasks)

    # Use explicit crop size if provided, otherwise auto
    crop_size = args.crop_size if args.crop_size else auto_crop_size

    # Pass 2: Transformation & Export
    orchestrator.run_pass2(
        analysis_results=analysis_results,
        target_r=target_r,
        crop_size=crop_size,
        output_base=args.output,
        generate_overlays=args.generate_overlays
    )


if __name__ == "__main__":
    main()

