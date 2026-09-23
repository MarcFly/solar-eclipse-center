import os
import glob
import cv2
import numpy as np
import tifffile
from detector import EclipseCircleDetector

class EclipseRotationAligner:
    def __init__(self, ref_image_path: str = None):
        self.detector = EclipseCircleDetector()
        self.ref_image_path = ref_image_path
        self.ref_anchors = None # (p_top, p_right, p_left) in relative offset from center
        self.ref_angle_tr = None
        self.ref_dist_tr = None
        self.tpl_top = None
        self.tpl_rt = None
        self.tpl_lt = None
        self.tpl_size = 40

    def _extract_halpha(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            r = img[:, :, 0].astype(np.float32)
            g = img[:, :, 1].astype(np.float32)
            b = img[:, :, 2].astype(np.float32)
            return np.maximum(0, r - 0.5 * (g + b))
        return img.astype(np.float32)

    def initialize_reference(self, ref_img: np.ndarray, cx: float, cy: float, r: float):
        """
        Extracts 3 prominence templates (Top ~12h, Right ~3h, Left ~9h) from reference frame.
        """
        ha = self._extract_halpha(ref_img)
        sz = self.tpl_size
        
        # 1. Top prominence (~12 o'clock: x ~ cx, y ~ cy - r)
        pt_top = self._find_prominence_peak(ha, cx, cy - 1.05 * r, search_rad=50)
        # 2. Right prominence (~3 o'clock: x ~ cx + r, y ~ cy)
        pt_rt = self._find_prominence_peak(ha, cx + 1.05 * r, cy, search_rad=50)
        # 3. Left prominence (~9 o'clock: x ~ cx - r, y ~ cy)
        pt_lt = self._find_prominence_peak(ha, cx - 1.05 * r, cy, search_rad=50)
        
        self.ref_anchors = {
            "top": pt_top - np.array([cx, cy]),
            "right": pt_rt - np.array([cx, cy]),
            "left": pt_lt - np.array([cx, cy]),
        }
        
        v_tr = pt_rt - pt_top
        self.ref_dist_tr = float(np.linalg.norm(v_tr))
        self.ref_angle_tr = float(np.rad2deg(np.arctan2(v_tr[1], v_tr[0])))
        
        # Extract normalized templates
        self.tpl_top = self._get_normalized_patch(ha, pt_top[0], pt_top[1], sz)
        self.tpl_rt = self._get_normalized_patch(ha, pt_rt[0], pt_rt[1], sz)
        self.tpl_lt = self._get_normalized_patch(ha, pt_lt[0], pt_lt[1], sz)
        
        return pt_top, pt_rt, pt_lt

    def _find_prominence_peak(self, ha: np.ndarray, x_est: float, y_est: float, search_rad: int = 50):
        x1, y1 = max(0, int(round(x_est - search_rad))), max(0, int(round(y_est - search_rad)))
        x2, y2 = min(ha.shape[1], int(round(x_est + search_rad))), min(ha.shape[0], int(round(y_est + search_rad)))
        crop = ha[y1:y2, x1:x2]
        crop_s = cv2.GaussianBlur(crop, (9, 9), 1.5)
        _, _, _, max_l = cv2.minMaxLoc(crop_s)
        return np.array([x1 + max_l[0], y1 + max_l[1]], dtype=np.float32)

    def _get_normalized_patch(self, img: np.ndarray, x: float, y: float, sz: int):
        x1, y1 = int(round(x - sz)), int(round(y - sz))
        x2, y2 = int(round(x + sz)), int(round(y + sz))
        p = img[y1:y2, x1:x2]
        std = np.std(p)
        return (p - np.mean(p)) / (std if std > 0 else 1.0)

    def estimate_rotation(self, img: np.ndarray, cx: float, cy: float, r: float):
        """
        Estimates rotation angle (deg) and scale relative to reference using Top and Right prominences.
        """
        ha = self._extract_halpha(img)
        sz = self.tpl_size
        
        # Match Top and Right templates
        pt_top, score_top = self._match_patch(ha, cx + self.ref_anchors["top"][0], cy + self.ref_anchors["top"][1], self.tpl_top)
        pt_rt, score_rt = self._match_patch(ha, cx + self.ref_anchors["right"][0], cy + self.ref_anchors["right"][1], self.tpl_rt)
        pt_lt, score_lt = self._match_patch(ha, cx + self.ref_anchors["left"][0], cy + self.ref_anchors["left"][1], self.tpl_lt)
        
        # Check if Top and Right are reliably found
        if pt_top is not None and pt_rt is not None and score_top > 0.25 and score_rt > 0.25:
            v_tr = pt_rt - pt_top
            dist_tr = float(np.linalg.norm(v_tr))
            ang_tr = float(np.rad2deg(np.arctan2(v_tr[1], v_tr[0])))
            
            d_rot = ang_tr - self.ref_angle_tr
            scale = dist_tr / self.ref_dist_tr
            confidence = min(score_top, score_rt)
            return d_rot, scale, confidence, (pt_top, pt_rt, pt_lt)
            
        return None, 1.0, 0.0, (pt_top, pt_rt, pt_lt)

    def _match_patch(self, img: np.ndarray, x_est: float, y_est: float, tpl: np.ndarray, search_rad: int = 40):
        sz = tpl.shape[0] // 2
        x1, y1 = max(0, int(round(x_est - search_rad - sz))), max(0, int(round(y_est - search_rad - sz)))
        x2, y2 = min(img.shape[1], int(round(x_est + search_rad + sz))), min(img.shape[0], int(round(y_est + search_rad + sz)))
        sa = img[y1:y2, x1:x2]
        std = np.std(sa)
        if std == 0:
            return None, 0.0
        sa_norm = (sa - np.mean(sa)) / std
        res = cv2.matchTemplate(sa_norm.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_l = cv2.minMaxLoc(res)
        
        # Sub-pixel peak estimation using parabolic fit around max_l
        px = x1 + max_l[0] + sz
        py = y1 + max_l[1] + sz
        return np.array([px, py], dtype=np.float32), max_v

def run_test():
    files = sorted(glob.glob("../EclipseTry2/3_Totality/*.tif"))
    aligner = EclipseRotationAligner()
    
    # Initialize with frame 0
    ref_f = files[0]
    ref_img = tifffile.imread(ref_f)
    cx0, cy0, r0 = aligner.detector.detect(ref_img)
    pt_top0, pt_rt0, pt_lt0 = aligner.initialize_reference(ref_img, cx0, cy0, r0)
    
    print(f"Master Reference: {os.path.basename(ref_f)}")
    print(f"  Top: {pt_top0}, Right: {pt_rt0}, Left: {pt_lt0}")
    print(f"  Top->Right Angle: {aligner.ref_angle_tr:.3f}°, Dist: {aligner.ref_dist_tr:.1f} px\n")
    
    indices = np.linspace(0, len(files)-1, 15, dtype=int)
    print(f"{'Idx':<4} | {'Filename':<15} | {'Rot (deg)':<12} | {'Scale':<10} | {'Conf':<8} | {'Top Peak':<18} | {'Right Peak':<18}")
    print("-" * 95)
    
    for idx in indices:
        f = files[idx]
        fname = os.path.basename(f)
        img = tifffile.imread(f)
        cx, cy, r = aligner.detector.detect(img)
        rot, scale, conf, pts = aligner.estimate_rotation(img, cx, cy, r)
        
        rot_str = f"{rot:+8.3f}°" if rot is not None else "  None  "
        scale_str = f"{scale:8.4f}" if rot is not None else "  None  "
        t_str = f"({pts[0][0]:.1f}, {pts[0][1]:.1f})" if pts[0] is not None else "None"
        r_str = f"({pts[1][0]:.1f}, {pts[1][1]:.1f})" if pts[1] is not None else "None"
        
        print(f"{idx:<4} | {fname:<15} | {rot_str:<12} | {scale_str:<10} | {conf:6.3f} | {t_str:<18} | {r_str:<18}")

if __name__ == "__main__":
    run_test()

