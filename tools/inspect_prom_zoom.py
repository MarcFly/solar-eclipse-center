import cv2
import numpy as np

def inspect_prom_zoom():
    zoom = cv2.imread("../samples/step4_rotation_alignment/prominence_alignment_zoom.jpg")
    print(f"Zoom shape: {zoom.shape}")
    
    # In zoom:
    # Top half: [0:240, :]
    #   Left: Top Before [0:240, 0:240]
    #   Right: Top Aligned [0:240, 240:480]
    # Bottom half: [240:480, :]
    #   Left: Right Before [240:480, 0:240]
    #   Right: Right Aligned [240:480, 240:480]
    
    top_b = zoom[0:240, 0:240]
    top_a = zoom[0:240, 240:480]
    rt_b = zoom[240:480, 0:240]
    rt_a = zoom[240:480, 240:480]
    
    # Measure distance between Red and Blue peaks in Top prominence Before vs Aligned
    def peak_dist(patch):
        r = patch[:, :, 2]
        b = patch[:, :, 0]
        # Ignore label text (rows 0-30)
        r = r[35:, :]
        b = b[35:, :]
        _, _, _, loc_r = cv2.minMaxLoc(r)
        _, _, _, loc_b = cv2.minMaxLoc(b)
        dist = np.sqrt((loc_r[0] - loc_b[0])**2 + (loc_r[1] - loc_b[1])**2)
        return dist, loc_r, loc_b
        
    d_top_b, lr_tb, lb_tb = peak_dist(top_b)
    d_top_a, lr_ta, lb_ta = peak_dist(top_a)
    
    d_rt_b, lr_rb, lb_rb = peak_dist(rt_b)
    d_rt_a, lr_ra, lb_ra = peak_dist(rt_a)
    
    print("Prominence Feature Peak Displacements (Red Frame 0 vs Blue Frame 151):")
    print(f"  Top Prominence:")
    print(f"    Before Alignment: {d_top_b:.1f} px displacement (Red={lr_tb}, Blue={lb_tb})")
    print(f"    After Alignment:  {d_top_a:.1f} px displacement (Red={lr_ta}, Blue={lb_ta})")
    print(f"  Right Prominence:")
    print(f"    Before Alignment: {d_rt_b:.1f} px displacement (Red={lr_rb}, Blue={lb_rb})")
    print(f"    After Alignment:  {d_rt_a:.1f} px displacement (Red={lr_ra}, Blue={lb_ra})")

if __name__ == "__main__":
    inspect_prom_zoom()

