# Solar Eclipse Image Processor (`eclipse-processor`)

---

## AI Disclaimer
This tool has been entirely vibecoded using Antigravity + Google Gemini 3.8 Flash - High reasoning. Most probably is crap, but it has helped me power through aligning 400 heavy images to facilitate merging later with Affinity Canvas. I took about 30min to align 30 images manually, I have spent around 4h vibecoding this bullcrap, worth it for me not so much for the earth's ecosystem.

It's not perfect, but it gets closer than myself with less time. Runs at around 9 images/second per step, my test images being around 400 images at 21Mpx 6000px x 4000px, separated in 5 totality folders. No true benchmarks atm nor planned.

(Edit) - That feeling of searchings the day after and finding that there is obviously already tools that do this. I had tried a ton of astrophotography software like PIPP, AutoStakkert!, Auto Align from Affinity Canvas / Lightroom, Siril, Sequator, all with terrible results with my terrible photos from my Fuji XE3, manual adjustment of tripod, manually made filter which I believe was wavy and wonky and an adapted (although great) 500mm f5.8 Nikon E Lens.
Pending to try:
 - Solar Eclipse Timeline Aligner - https://github.com/hotdogee/solar-eclipse-timelapse-aligner?st_source=ai_overview
 - Umbra - https://umbraprocessing.com/

---

A high-precision, 16-bit astronomical image processing pipeline engineered for solar eclipse photography. It automates limb detection, coronal streamer measurement, sub-pixel centering, solar radius normalization, multi-point prominence field rotation alignment, and uniform square cropping—strictly preserving native **16-bit `uint16` depth** and photometric accuracy for composite HDR stacking.

---

## Key Capabilities

- **Flare- & Bead-Immune Circle Detection**: Robustly identifies the solar/lunar limb across all eclipse phases (Contact 1/2, Totality, Exit, Post-Totality) without distortion from diamond ring flares, Baily's beads, or eruptive prominences.
- **Dynamic-Range Coronal Extent Estimation**: Measures the maximum radial extent of faint coronal streamers using an adaptive contrast pass grounded in sky background noise ($\sigma_{bg}$) and inner coronal dynamic range.
- **Multi-Point Prominence Rotation Alignment**: Tracks three limb landmarks—**Top (~12h)**, **Right (~3h)**, and **Left (~9h)** prominences—using sub-pixel normalized cross-correlation (NCC). The ultra-stable Top-Right baseline isolates field rotation ($\sim 1.5^\circ - 3^\circ$) without distortion from dynamic local eruptions.
- **Single-Pass Composite Affine Transformation**: Composes centering, radius scaling ($R \to R_{target}$), field rotation ($\theta$), and square cropping into a **single `cv2.INTER_LANCZOS4` interpolation pass**, eliminating generational resampling blur.
- **Photometric & Dtype Integrity**: Guarantees zero bit-depth reduction, zero tonal clipping, and zero color manipulation. Output is strictly native 16-bit `uint16` TIFF.
- **Two-Pass Bulk Orchestrator**: Multi-threaded batch processor with automatic recursive folder discovery, sampling mode (`--sample N`), lossless compression, and optional visual verification overlays.

---

## Project Structure

```
eclipse_processor/
├── pyproject.toml                      # Package specifications & dependencies
├── README.md                           # Quickstart guide & CLI usage reference
├── orchestrator.py                     # Convenience CLI wrapper
│
├── src/
│   └── eclipse_processor/              # Installable Python package
│       ├── __init__.py                 # Public package API exports
│       ├── cli.py                      # Bulk orchestrator engine & argument parser
│       └── core/                       # Core algorithmic modules
│           ├── __init__.py
│           ├── detector.py             # Step 1: Solar/lunar limb circle detector
│           ├── corona.py               # Step 2: Coronal streamer extent detector
│           ├── centering.py            # Step 3: Sub-pixel translation
│           ├── transformer.py          # Step 4: Single-pass composite affine transformer
│           └── rotator.py              # Step 4.5: Multi-point prominence rotation aligner
│
├── scripts/                            # Step runners & verification scripts
│   ├── run_step1_circle_detection.py
│   ├── run_step2_corona_detection.py
│   ├── run_step3_centering.py
│   ├── run_step4_affine_crop.py
│   └── run_step4_5_rotation_alignment.py
│
├── tools/                              # Diagnostic, research, and inspection utilities
│   ├── measure_exact_rotation.py
│   ├── inspect_prominences.py
│   ├── test_composite_pure_rot.py
│   └── ... (scratch analysis tools)
│
└── docs/                               # In-depth architectural & mathematical documentation
    └── ARCHITECTURE.md                 # Technical deep-dive into each pipeline step
```

---

## Installation & Environment

This project is managed via [`uv`](https://github.com/astral-sh/uv) (or standard `pip` in Python $\ge 3.13$):

```powershell
# Navigate to the processor directory
cd eclipse_processor

# Sync virtual environment and install dependencies
uv sync
```

---

## Quickstart & CLI Usage

The orchestrator can be invoked either directly via `uv run python orchestrator.py` or via the installed console entrypoint `uv run eclipse-processor`.

### 1. Fast Trial Run (Sampling Mode)
Process a small sample of images (e.g. 2 images per phase folder) to verify output and generate visual diagnostic overlays:
```powershell
uv run python orchestrator.py ../EclipseTry2 -s 2 -o ../processed_trial --generate-overlays
```

### 2. Full Batch Processing for Stacking (with Rotation Alignment)
Automatically detects consensus solar radius and optimal square crop dimensions, locks field rotation using multi-point prominence tracking, and exports aligned 16-bit TIFFs:
```powershell
uv run python orchestrator.py ../EclipseTry2 -o ../processed_eclipse --align-rotation -w 6
```
*(For the sample $6000\times 4000$ APS-C dataset, this automatically resolves to $R_{target} \approx 586.5\text{ px}$ and $S_{crop} = 4000 \times 4000\text{ px}$.)*

### 3. Full Batch Processing (Raw Camera Orientation)
Preserves raw camera orientation while centering, normalizing radius, and square-cropping:
```powershell
uv run python orchestrator.py ../EclipseTry2 -o ../processed_eclipse -w 6
```

### 4. Lossless Compression (Reduces Disk Usage by ~40%)
Enables Deflate or LZW lossless TIFF compression:
```powershell
uv run python orchestrator.py ../EclipseTry2 -o ../processed_eclipse --compression deflate -w 6
```

---

## Dynamic Scale & Resolution Independence

The pipeline contains **zero hardcoded pixel assumptions**. It dynamically adapts to any optical focal length, sensor resolution, and aspect ratio:

- **Arbitrary Sensor Resolution**: Whether processing 12MP, 24MP ($6000\times 4000$), 45MP ($8256\times 5504$), 61MP ($9504\times 6336$), or video frames (4K/8K), the limb detector scales its search space relative to image dimensions ($\min(H, W)$).
- **Arbitrary Optical Focal Lengths**: From wide-field telephotos ($200\text{mm}$) to large catadioptric telescopes ($2000\text{mm}$), the celestial body radius is dynamically discovered from the imagery via algebraic circle fitting.
- **Dataset Consensus Radius**: The default target radius (`--target-radius None`) computes the statistical **median radius** of all detected frames across the dataset, ensuring perfect 1:1 scale normalization without distortion.
- **Dynamic Sensor-Height Crop Cap**: The uniform square crop canvas is calculated from the global maximum coronal streamer extent ($2.0 \times 1.05 \times \max R_{corona}$), dynamically capped at the native frame height $H_{canvas}$ ($\min(\dots, H_{canvas})$) to maximize coronal retention without padding artificial borders.

---

## Command-Line Arguments Reference

| Option | Type | Default | Description |
|---|---|---|---|
| `inputs` | Positional | `../EclipseTry2` | Input directory, subdirectories, or list of TIFF files. |
| `-o`, `--output` | String | `../processed_eclipse` | Output directory for standardized 16-bit TIFFs. |
| `-s`, `--sample` | Integer | `None` | Sample mode: processes up to $N$ evenly spaced frames per folder. |
| `--align-rotation` | Flag | `False` | Enables multi-point prominence rotation alignment and solar registration. |
| `-r`, `--target-radius` | Float | `Auto (median)` | Standardized solar/lunar radius in output pixels. If omitted, uses median detected radius. |
| `-c`, `--crop-size` | Integer | `Auto` | Square crop dimension. If omitted, calculated from max coronal extent capped at $H_{canvas}$. |
| `-m`, `--margin` | Float | `0.05` | Margin ratio above maximum coronal extent (default: `+5%`). |
| `--no-cap-height` | Flag | `False` | Disables capping crop size to sensor height (allows expanding beyond canvas). |
| `--generate-overlays` | Flag | `False` | Exports diagnostic `.jpg` overlays with center crosshairs, axes, and anchor points. |
| `--compression` | Choice | `none` | Lossless TIFF compression: `none`, `deflate`, `lzw`, `zstd`. |
| `-w`, `--workers` | Integer | `4` | Number of concurrent worker threads. |

---

## Step-by-Step Overview

| Step | Module | Purpose |
|---|---|---|
| **Step 1: Circle Detection** | [`detector.py`](file:///c:/Users/mtorr/Desktop/TestProgram/eclipse_processor/src/eclipse_processor/core/detector.py) | Flare-immune sub-pixel Sun/Moon limb detection with dynamic optical radius auto-discovery. |
| **Step 2: Corona Detection** | [`corona.py`](file:///c:/Users/mtorr/Desktop/TestProgram/eclipse_processor/src/eclipse_processor/core/corona.py) | Dynamic-range contrast pass measuring streamer extent (+5% margin, dynamically capped to $H_{canvas}$). |
| **Step 3: Centering** | [`centering.py`](file:///c:/Users/mtorr/Desktop/TestProgram/eclipse_processor/src/eclipse_processor/core/centering.py) | Sub-pixel translation to canvas center $(W/2, H/2)$. |
| **Step 4: Affine Transform** | [`transformer.py`](file:///c:/Users/mtorr/Desktop/TestProgram/eclipse_processor/src/eclipse_processor/core/transformer.py) | Single-pass composite matrix: centering + scale + rotation + square crop. |
| **Step 4.5: Rotation Alignment** | [`rotator.py`](file:///c:/Users/mtorr/Desktop/TestProgram/eclipse_processor/src/eclipse_processor/core/rotator.py) | Multi-anchor prominence tracking (Top, Right, Left) for stacking lock. |
| **Step 5: Bulk Orchestrator** | [`cli.py`](file:///c:/Users/mtorr/Desktop/TestProgram/eclipse_processor/src/eclipse_processor/cli.py) | Multi-threaded two-pass batch engine exporting full 16-bit TIFFs. |

For an in-depth mathematical and algorithmic explanation of each module, see [**docs/ARCHITECTURE.md**](file:///c:/Users/mtorr/Desktop/TestProgram/eclipse_processor/docs/ARCHITECTURE.md).

---

## Python API Usage

The core modules can be imported and integrated directly into custom astrophotography scripts:

```python
import tifffile
from eclipse_processor import (
    EclipseCircleDetector,
    EclipseCoronaDetector,
    EclipseAffineTransformer,
    EclipseRotationAligner
)

# 1. Detect celestial body
img = tifffile.imread("totality_frame.tif")
detector = EclipseCircleDetector()
cx, cy, r = detector.detect(img)

# 2. Measure corona extent
corona_detector = EclipseCoronaDetector(margin_ratio=0.05)
corona_res = corona_detector.detect(img, cx, cy, r)
print(f"Max Corona: {corona_res.r_corona_max:.1f} px | Ideal Crop: {corona_res.ideal_crop_size} px")

# 3. Transform with optional rotation alignment (preserves native scale if target_r=None)
transformer = EclipseAffineTransformer()
res = transformer.transform(
    img,
    cx=cx, cy=cy, r=r,
    target_r=None,                          # None = native detected radius; or pass float
    crop_size=corona_res.ideal_crop_size,   # Dynamically sized to coronal extent
    rotation_deg=-1.57                      # Optional field rotation alignment
)

# 4. Save 16-bit uint16 result
tifffile.imwrite("standardized_output.tif", res.img_transformed, photometric='rgb')
```
