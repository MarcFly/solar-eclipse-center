import cv2
import numpy as np
import tifffile

def measure_alignment_f1_f3():
    # Load processed images where Moon is already centered at (2000, 2000)
    f1 = "../processed_eclipse/3_Totality/_DSF5184.tif"
    f3 = "../processed_eclipse/3_Totality/_DSF5335.tif"
    
    img1 = tifffile.imread(f1)
    img3 = tifffile.imread(f3)
    
    c = 2000
    r = 586.5
    
    # We want to measure the relative transformation between img1 and img3:
    # 1. On the lunar disk (inside r)
    # 2. On the prominences and inner corona (between 1.01*r and 1.35*r)
    # 3. In the outer corona (between 1.35*r and 2.5*r)
    
    # Let's inspect the prominent features:
    # Top prominence (~12 o'clock): (cx, cy - r) = (2000, 1413)
    # Right prominence (~3 o'clock): (cx + r, cy) = (2586, 2000)
    # Left prominence (~9 o'clock): (cx - r, cy) = (1413, 2000)
    
    # Let's extract 128x128 patches around Top and Right prominences in both images
    # Top prominence
    top_y = 1380
    top_x = 2000
    w = 64
    
    patch_top1 = img1[top_y-w:top_y+w, top_x-w:top_x+w, :]
    patch_top3 = img3[top_y-w:top_y+w, top_x-w:top_x+w, :]
    
    # Right prominence
    rt_y = 2000
    rt_x = 2630
    patch_rt1 = img1[rt_y-w:rt_y+w, rt_x-w:rt_x+w, :]
    patch_rt3 = img3[rt_y-w:rt_y+w, rt_x-w:rt_x+w, :]
    
    # Left prominence
    lt_y = 2000
    lt_x = 1410
    patch_lt1 = img1[lt_y-w:lt_y+w, lt_x-w:lt_x+w, :]
    patch_lt3 = img3[lt_y-w:lt_y+w, lt_x-w:lt_x+w, :]
    
    def match_subpixel(p1, p3):
        # Match using phase correlation or template matching on red excess (prominence)
        r1 = p1[:, :, 0].astype(np.float32) - 0.5 * (p1[:, :, 1].astype(np.float32) + p1[:, :, 2].astype(np.float32))
        r3 = p3[:, :, 0].astype(np.float32) - 0.5 * (p3[:, :, 1].astype(np.float32) + p3[:, :, 2].astype(np.float32))
        r1 = np.maximum(0, r1)
        r3 = np.maximum(0, r3)
        
        # Subpixel phase correlation
        hann = cv2.createHanningWindow((2*w, 2*w), cv2.CV_32F)
        shift, response = cv2.phaseCorrelate(r1, r3, hann)
        return shift, response
        
    shift_top, resp_top = match_subpixel(patch_top1, patch_top3)
    shift_rt, resp_rt = match_subpixel(patch_rt1, patch_rt3)
    shift_lt, resp_lt = match_subpixel(patch_lt1, patch_lt3)
    
    print("Relative shift from _DSF5184 to _DSF5335 (in centered coordinates):")
    print(f"  Top Prominence (12h):   dx = {shift_top[0]:+6.2f} px, dy = {shift_top[1]:+6.2f} px (resp={resp_top:.3f})")
    print(f"  Right Prominence (3h): dx = {shift_rt[0]:+6.2f} px, dy = {shift_rt[1]:+6.2f} px (resp={resp_rt:.3f})")
    print(f"  Left Prominence (9h):  dx = {shift_lt[0]:+6.2f} px, dy = {shift_lt[1]:+6.2f} px (resp={resp_lt:.3f})")

if __name__ == "__main__":
    measure_alignment_f1_f3()

