import os
import glob
import cv2
import numpy as np
import tifffile
from detector import EclipseCircleDetector

def test_triangle():
    files = sorted(glob.glob("../EclipseTry2/3_Totality/*.tif"))
    det = EclipseCircleDetector()
    
    # Take frame 0 as reference
    ref_file = files[0]
    ref_img = tifffile.imread(ref_file)
    cx0, cy0, r0 = det.detect(ref_img)
    
    def get_ha(img):
        r = img[:, :, 0].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        b = img[:, :, 2].astype(np.float32)
        return np.maximum(0, r - 0.5 * (g + b))
        
    ref_ha = get_ha(ref_img)
    
    def find_peak(ha_map, cx_e, cy_e, half=60):
        x1, y1 = int(round(cx_e - half)), int(round(cy_e - half))
        x2, y2 = int(round(cx_e + half)), int(round(cy_e + half))
        crop = ha_map[y1:y2, x1:x2]
        crop_s = cv2.GaussianBlur(crop, (9, 9), 1.5)
        _, _, _, max_l = cv2.minMaxLoc(crop_s)
        return x1 + max_l[0], y1 + max_l[1]
        
    p0_top = np.array(find_peak(ref_ha, cx0, cy0 - 1.05 * r0), dtype=np.float32)
    p0_rt = np.array(find_peak(ref_ha, cx0 + 1.05 * r0, cy0), dtype=np.float32)
    p0_lt = np.array(find_peak(ref_ha, cx0 - 1.05 * r0, cy0), dtype=np.float32)
    
    ref_pts = np.array([p0_top, p0_rt, p0_lt], dtype=np.float32)
    
    tpl_sz = 32
    tpl_top = ref_ha[int(p0_top[1])-tpl_sz:int(p0_top[1])+tpl_sz, int(p0_top[0])-tpl_sz:int(p0_top[0])+tpl_sz]
    tpl_rt = ref_ha[int(p0_rt[1])-tpl_sz:int(p0_rt[1])+tpl_sz, int(p0_rt[0])-tpl_sz:int(p0_rt[0])+tpl_sz]
    tpl_lt = ref_ha[int(p0_lt[1])-tpl_sz:int(p0_lt[1])+tpl_sz, int(p0_lt[0])-tpl_sz:int(p0_lt[0])+tpl_sz]
    
    tpl_top = (tpl_top - tpl_top.mean()) / (tpl_top.std() + 1e-6)
    tpl_rt = (tpl_rt - tpl_rt.mean()) / (tpl_rt.std() + 1e-6)
    tpl_lt = (tpl_lt - tpl_lt.mean()) / (tpl_lt.std() + 1e-6)
    
    def match_tpl(ha, cx_e, cy_e, tpl, half=90):
        x1, y1 = max(0, int(round(cx_e - half))), max(0, int(round(cy_e - half)))
        x2, y2 = min(ha.shape[1], int(round(cx_e + half))), min(ha.shape[0], int(round(cy_e + half)))
        search = ha[y1:y2, x1:x2]
        s_norm = (search - search.mean()) / (search.std() + 1e-6)
        res = cv2.matchTemplate(s_norm.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_l = cv2.minMaxLoc(res)
        return np.array([x1 + max_l[0] + tpl_sz, y1 + max_l[1] + tpl_sz], dtype=np.float32), max_v

    indices = np.linspace(0, len(files)-1, 12, dtype=int)
    print(f"{'Idx':<4} | {'Filename':<15} | {'Rot (deg)':<12} | {'Scale':<10} | {'tx (px)':<10} | {'ty (px)':<10} | {'Fit Error':<10}")
    print("-" * 80)
    
    for idx in indices:
        f = files[idx]
        fname = os.path.basename(f)
        img = tifffile.imread(f)
        cx, cy, r = det.detect(img)
        ha = get_ha(img)
        
        pt_top, v_t = match_tpl(ha, cx, cy - 1.05 * r, tpl_top)
        pt_rt, v_r = match_tpl(ha, cx + 1.05 * r, cy, tpl_rt)
        pt_lt, v_l = match_tpl(ha, cx - 1.05 * r, cy, tpl_lt)
        
        tgt_pts = np.array([pt_top, pt_rt, pt_lt], dtype=np.float32)
        
        # Fit similarity transform (scale, rotation, translation) from target to reference
        M, inliers = cv2.estimateAffinePartial2D(tgt_pts, ref_pts)
        if M is not None:
            # M maps tgt_pts to ref_pts
            # Decompose M: [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]
            s = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
            rot = np.rad2deg(np.arctan2(M[1, 0], M[0, 0]))
            tx = M[0, 2]
            ty = M[1, 2]
            
            # Reprojection error
            warped_pts = cv2.transform(tgt_pts[None, :, :], M)[0]
            err = np.mean(np.linalg.norm(warped_pts - ref_pts, axis=1))
            print(f"{idx:<4} | {fname:<15} | {rot:+8.3f}°   | {s:8.4f}   | {tx:+8.2f}   | {ty:+8.2f}   | {err:6.2f} px")
        else:
            print(f"{idx:<4} | {fname:<15} | FAILED")

if __name__ == "__main__":
    test_triangle()

