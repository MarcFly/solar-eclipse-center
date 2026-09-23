import cv2
import numpy as np

def inspect_visual_composite():
    comp = cv2.imread("../samples/rgb_motion_composite.jpg")
    print(f"Composite shape: {comp.shape}")
    
    # In rgb_motion_composite.jpg:
    # Blue = f3_5335
    # Green = f2_5259
    # Red = f1_5184
    
    # Let's check the color fringing in the 3 prominence regions:
    # 1. Left (9 o'clock): y=400, x=400 - 586.5*0.5 = 400, 107
    # 2. Top (12 o'clock): x=400, y=400 - 586.5*0.5 = 400, 107
    # 3. Right (3 o'clock): y=400, x=400 + 586.5*0.5 = 400, 693
    
    # Let's save zoomed in crops of each of the 3 prominences and print what colors appear
    crops = {
        "left_prominence": comp[320:480, 50:180],
        "top_prominence": comp[50:180, 320:480],
        "right_prominence": comp[320:480, 620:750]
    }
    
    for name, c in crops.items():
        cv2.imwrite(f"../samples/{name}_fringing.jpg", c)
        b, g, r = c[:, :, 0], c[:, :, 1], c[:, :, 2]
        print(f"\n{name}:")
        print(f"  Red (f1) max={r.max()}, Green (f2) max={g.max()}, Blue (f3) max={b.max()}")
        # Check spatial offset between Red (f1) peak and Blue (f3) peak
        _, _, _, max_r = cv2.minMaxLoc(r)
        _, _, _, max_b = cv2.minMaxLoc(b)
        dx = max_b[0] - max_r[0]
        dy = max_b[1] - max_r[1]
        print(f"  Peak Red pos: {max_r}, Peak Blue pos: {max_b} -> Delta = ({dx}, {dy}) px")

if __name__ == "__main__":
    inspect_visual_composite()

