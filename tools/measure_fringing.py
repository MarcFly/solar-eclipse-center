import cv2
import numpy as np

def measure_fringing_reduction():
    side_by_side = cv2.imread("../samples/step4_rotation_alignment/rgb_motion_comparison_side_by_side.jpg")
    w = side_by_side.shape[1] // 2
    comp_before = side_by_side[:, :w, :]
    comp_after = side_by_side[:, w:, :]
    
    # In an RGB motion composite of 3 frames (Red=F0, Green=F75, Blue=F151):
    # Where features are perfectly locked, R == G == B (neutral monochrome, low color saturation / chromatic variance).
    # Where features are displaced / unaligned / rotating, R != G != B (high chromatic variance, bright rainbow fringing).
    
    def calculate_chromatic_fringing(comp):
        r = comp[:, :, 2].astype(np.float32)
        g = comp[:, :, 1].astype(np.float32)
        b = comp[:, :, 0].astype(np.float32)
        
        # Color variance at each pixel: variance among (R, G, B)
        mean_rgb = (r + g + b) / 3.0
        var_rgb = ((r - mean_rgb)**2 + (g - mean_rgb)**2 + (b - mean_rgb)**2) / 3.0
        
        # Mask out dark sky and saturated core to focus on corona and prominences
        mask = (mean_rgb > 30) & (mean_rgb < 240)
        
        avg_fringing = np.mean(np.sqrt(var_rgb[mask]))
        return avg_fringing
        
    fringe_before = calculate_chromatic_fringing(comp_before)
    fringe_after = calculate_chromatic_fringing(comp_after)
    reduction = (fringe_before - fringe_after) / fringe_before * 100.0
    
    print(f"Chromatic Fringing (Dispersion / Misalignment):")
    print(f"  Before Alignment: {fringe_before:.2f} counts")
    print(f"  After Alignment:  {fringe_after:.2f} counts")
    print(f"  Fringing Reduction: {reduction:.1f}%")

if __name__ == "__main__":
    measure_fringing_reduction()

