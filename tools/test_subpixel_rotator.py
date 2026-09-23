import os
import glob
import cv2
import numpy as np
import tifffile
from detector import EclipseCircleDetector

def subpixel_peak(res, max_loc):
    """
    Fits 2D quadratic surface around max_loc in correlation surface `res`
    to find true sub-pixel peak position (dx, dy).
    """
    x, y = max_loc
    h, w = res.shape
    if 1 <= x < w - 1 and 1 <= y < h - 1:
        # Patch of 3x3 around peak
        patch = res[y-1:y+2, x-1:x+2]
        # Parabolic fit in 1D along x and y
        dx = (patch[1, 2] - patch[1, 0]) / (2.0 * (2.0 * patch[1, 1] - patch[1, 0] - patch[1, 2]) + 1e-6)
        dy = (patch[2, 1] - patch[0, 1]) / (2.0 * (2.0 * patch[1, 1] - patch[0, 1] - patch[2, 1]) + 1e-6)
        # clamp to [-0.5, 0.5]
        dx = np.clip(dx, -0.5, 0.5)
        dy = np.clip(dy, -0.5, 0.5)
        return float(x + dx), float(y + dy)
    return float(x), float(y)

def test():
    files = sorted(glob.glob("../EclipseTry2/3_Totality/*.tif"))
    det = EclipseCircleDetector()
    
    ref_f = files[0]
    ref_img = tifffile.imread(ref_f)
    cx0, cy0, r0 = det.detect(ref_img)
    
    def get_ha(img):
        r = img[:, :, 0].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        b = img[:, :, 2].astype(np.float32)
        ha = np.maximum(0, r - 0.5 * (g + b))
        # Log-stretch to make faint parts visible while compressing bright saturation
        return np.log1p(ha)
        
    ref_ha = get_ha(ref_img)
    
    # Locate reference anchor centers
    # Top: cx, cy - 1.05 * r
    # Right: cx + 1.05 * r, cy
    # Left: cx - 1.05 * r, cy
    tpl_sz = 30
    
    def extract_tpl(ha, bx, by):
        bx, by = int(round(bx)), int(round(by))
        p = ha[by-tpl_sz:by+tpl_sz, bx-tpl_sz:bx+tpl_sz]
        std = np.std(p)
        return (p - np.mean(p)) / (std if std > 0 else 1.0)
        
    tpl_top = extract_tpl(ref_ha, cx0, cy0 - 1.05 * r0)
    tpl_rt = extract_tpl(ref_ha, cx0 + 1.05 * r0, cy0)
    tpl_lt = extract_tpl(ref_ha, cx0 - 1.05 * r0, cy0)
    
    # Match in ref itself to get reference coordinates
    def match(ha, bx, by, tpl, search=40):
        bx, by = int(round(bx)), int(round(by))
        x1, y1 = max(0, bx - search - tpl_sz), max(0, by - search - tpl_sz)
        x2, y2 = min(ha.shape[1], bx + search + tpl_sz), min(ha.shape[0], by + search + tpl_sz)
        sa = ha[y1:y2, x1:x2]
        std = np.std(sa)
        if std == 0:
            return None, 0.0
        sa_norm = (sa - np.mean(sa)) / std
        res = cv2.matchTemplate(sa_norm.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_l = cv2.minMaxLoc(res)
        sub_x, sub_y = subpixel_peak(res, max_l)
        px = x1 + sub_x + tpl_sz
        py = y1 + sub_y + tpl_sz
        return np.array([px, py], dtype=np.float32), max_v
        
    ref_pt_top, _ = match(ref_ha, cx0, cy0 - 1.05 * r0, tpl_top)
    ref_pt_rt, _ = match(ref_ha, cx0 + 1.05 * r0, cy0, tpl_rt)
    ref_pt_lt, _ = match(ref_ha, cx0 - 1.05 * r0, cy0, tpl_lt)
    
    v_tr0 = ref_pt_rt - ref_pt_top
    ref_ang_tr = np.rad2deg(np.arctan2(v_tr0[1], v_tr0[0]))
    ref_dist_tr = np.linalg.norm(v_tr0)
    
    print(f"Master Reference: {os.path.basename(ref_f)}")
    print(f"  Top: {ref_pt_top}, Right: {ref_pt_rt}, Left: {ref_pt_lt}")
    print(f"  Top->Right Vector: angle={ref_ang_tr:.3f}°, dist={ref_dist_tr:.2f} px\n")
    
    indices = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 151]
    print(f"{'Idx':<4} | {'Filename':<15} | {'Rot (deg)':<12} | {'Scale':<10} | {'Score_T':<8} | {'Score_R':<8} | {'Score_L':<8}")
    print("-" * 80)
    
    for idx in indices:
        f = files[idx]
        fname = os.path.basename(f)
        img = tifffile.imread(f)
        cx, cy, r = det.detect(img)
        ha = get_ha(img)
        
        # Estimate search centers using current cx, cy, r
        pt_top, sc_t = match(ha, cx, cy - 1.05 * r, tpl_top)
        pt_rt, sc_r = match(ha, cx + 1.05 * r, cy, tpl_rt)
        pt_lt, sc_l = match(ha, cx - 1.05 * r, cy, tpl_lt)
        
        if sc_t > 0.4 and sc_r > 0.4:
            v_tr = pt_rt - pt_top
            ang_tr = np.rad2deg(np.arctan2(v_tr[1], v_tr[0]))
            dist_tr = np.linalg.norm(v_tr)
            d_rot = ang_tr - ref_ang_tr
            scale = dist_tr / ref_dist_tr
            print(f"{idx:<4} | {fname:<15} | {d_rot:+8.3f}°   | {scale:8.4f}   | {sc_t:6.3f}   | {sc_r:6.3f}   | {sc_l:6.3f}")
        else:
            print(f"{idx:<4} | {fname:<15} | LOW CONFIDENCE | sc_t={sc_t:.2f}, sc_r={sc_r:.2f}, sc_l={sc_l:.2f}")

if __name__ == "__main__":
    test()

