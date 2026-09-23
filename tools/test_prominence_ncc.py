import os
import glob
import cv2
import numpy as np
import tifffile
from detector import EclipseCircleDetector

def test_prominence_ncc():
    files = sorted(glob.glob("../EclipseTry2/3_Totality/*.tif"))
    det = EclipseCircleDetector()
    
    # Let's take frame 0 as template reference
    ref_file = files[0]
    ref_img = tifffile.imread(ref_file)
    cx0, cy0, r0 = det.detect(ref_img)
    
    # Extract H-alpha / red excess
    def get_halpha(img):
        r = img[:, :, 0].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        b = img[:, :, 2].astype(np.float32)
        return np.maximum(0, r - 0.5 * (g + b))
        
    ref_ha = get_halpha(ref_img)
    
    # Let's define the 3 prominence templates from ref_img:
    # 1. Top prominence: cx0, cy0 - r0
    # 2. Right prominence: cx0 + r0, cy0
    # 3. Left prominence: cx0 - r0, cy0
    
    # We locate their precise subpixel peaks in ref_img
    w_box = 80
    
    def find_peak_in_box(ha_map, cx_est, cy_est, half=60):
        x1, y1 = int(round(cx_est - half)), int(round(cy_est - half))
        x2, y2 = int(round(cx_est + half)), int(round(cy_est + half))
        crop = ha_map[y1:y2, x1:x2]
        # Gaussian smooth to find major peak
        crop_s = cv2.GaussianBlur(crop, (9, 9), 1.5)
        min_v, max_v, min_l, max_l = cv2.minMaxLoc(crop_s)
        px = x1 + max_l[0]
        py = y1 + max_l[1]
        return px, py
        
    p_top_x, p_top_y = find_peak_in_box(ref_ha, cx0, cy0 - 1.05 * r0)
    p_rt_x, p_rt_y = find_peak_in_box(ref_ha, cx0 + 1.05 * r0, cy0)
    p_lt_x, p_lt_y = find_peak_in_box(ref_ha, cx0 - 1.05 * r0, cy0)
    
    print(f"Reference [{os.path.basename(ref_file)}]:")
    print(f"  Moon Center: ({cx0:.1f}, {cy0:.1f}), R={r0:.1f}")
    print(f"  Top Prominence Peak:   ({p_top_x}, {p_top_y})")
    print(f"  Right Prominence Peak: ({p_rt_x}, {p_rt_y})")
    print(f"  Left Prominence Peak:  ({p_lt_x}, {p_lt_y})")
    
    # Extract template patches (size 64x64) around each peak
    tpl_sz = 32
    tpl_top = ref_ha[p_top_y-tpl_sz:p_top_y+tpl_sz, p_top_x-tpl_sz:p_top_x+tpl_sz]
    tpl_rt = ref_ha[p_rt_y-tpl_sz:p_rt_y+tpl_sz, p_rt_x-tpl_sz:p_rt_x+tpl_sz]
    tpl_lt = ref_ha[p_lt_y-tpl_sz:p_lt_y+tpl_sz, p_lt_x-tpl_sz:p_lt_x+tpl_sz]
    
    # Normalize templates
    tpl_top = (tpl_top - tpl_top.mean()) / (tpl_top.std() + 1e-6)
    tpl_rt = (tpl_rt - tpl_rt.mean()) / (tpl_rt.std() + 1e-6)
    tpl_lt = (tpl_lt - tpl_lt.mean()) / (tpl_lt.std() + 1e-6)
    
    # Test on a set of 12 frames across totality
    indices = np.linspace(0, len(files)-1, 12, dtype=int)
    
    print(f"\n{'Idx':<4} | {'Filename':<15} | {'Top Peak':<18} | {'Right Peak':<18} | {'Left Peak':<18} | {'Rot (T->R)':<12}")
    print("-" * 90)
    
    def match_template_subpixel(target_ha, cx_est, cy_est, tpl, search_half=100):
        x1, y1 = int(round(cx_est - search_half)), int(round(cy_est - search_half))
        x2, y2 = int(round(cx_est + search_half)), int(round(cy_est + search_half))
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(target_ha.shape[1], x2); y2 = min(target_ha.shape[0], y2)
        search_area = target_ha[y1:y2, x1:x2]
        
        # Normalize search area
        sa_norm = (search_area - search_area.mean()) / (search_area.std() + 1e-6)
        
        res = cv2.matchTemplate(sa_norm.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        min_v, max_v, min_l, max_l = cv2.minMaxLoc(res)
        
        if max_v < 0.3:
            return None, max_v
            
        # Subpixel refinement using quadratic peak fit
        rx, ry = max_l
        # offset by tpl_sz to get center of template
        px = x1 + rx + tpl_sz
        py = y1 + ry + tpl_sz
        return (px, py), max_v
        
    ref_vec_tr = np.array([p_rt_x - p_top_x, p_rt_y - p_top_y], dtype=np.float32)
    ref_ang_tr = np.rad2deg(np.arctan2(ref_vec_tr[1], ref_vec_tr[0]))
    
    for idx in indices:
        f = files[idx]
        fname = os.path.basename(f)
        img = tifffile.imread(f)
        cx, cy, r = det.detect(img)
        ha = get_halpha(img)
        
        # Estimate search positions based on Moon center and nominal radius
        # Top: cx, cy - r
        # Right: cx + r, cy
        # Left: cx - r, cy
        pt_top, v_top = match_template_subpixel(ha, cx, cy - 1.05 * r, tpl_top)
        pt_rt, v_rt = match_template_subpixel(ha, cx + 1.05 * r, cy, tpl_rt)
        pt_lt, v_lt = match_template_subpixel(ha, cx - 1.05 * r, cy, tpl_lt)
        
        t_str = f"({pt_top[0]:.1f}, {pt_top[1]:.1f})" if pt_top else "  None"
        r_str = f"({pt_rt[0]:.1f}, {pt_rt[1]:.1f})" if pt_rt else "  None"
        l_str = f"({pt_lt[0]:.1f}, {pt_lt[1]:.1f})" if pt_lt else "  None"
        
        if pt_top and pt_rt:
            vec_tr = np.array([pt_rt[0] - pt_top[0], pt_rt[1] - pt_top[1]], dtype=np.float32)
            ang_tr = np.rad2deg(np.arctan2(vec_tr[1], vec_tr[0]))
            d_rot = ang_tr - ref_ang_tr
            rot_str = f"{d_rot:+7.3f}°"
        else:
            rot_str = "  N/A  "
            
        print(f"{idx:<4} | {fname:<15} | {t_str:<18} | {r_str:<18} | {l_str:<18} | {rot_str:<12}")

if __name__ == "__main__":
    test_prominence_ncc()

