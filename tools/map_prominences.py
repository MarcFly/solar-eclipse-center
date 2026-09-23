import cv2
import numpy as np
import tifffile
from detector import EclipseCircleDetector

def map_prominences():
    detector = EclipseCircleDetector()
    files = [
        "../EclipseTry2/3_Totality/_DSF5184.tif",
        "../EclipseTry2/3_Totality/_DSF5206.tif",
        "../EclipseTry2/3_Totality/_DSF5259.tif",
        "../EclipseTry2/3_Totality/_DSF5335.tif",
    ]
    
    for fpath in files:
        fname = fpath.split('/')[-1]
        img = tifffile.imread(fpath)
        cx, cy, r = detector.detect(img)
        
        # Unroll a narrow ring right along the limb: 0.99*r to 1.10*r
        n_angles = 720 # 0.5 deg per step
        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        radii = np.linspace(r * 0.99, r * 1.10, 30)
        
        grid_theta, grid_r = np.meshgrid(angles, radii)
        xs = (cx + grid_r * np.cos(grid_theta)).astype(np.float32)
        ys = (cy + grid_r * np.sin(grid_theta)).astype(np.float32)
        
        r_c = cv2.remap(img[:, :, 0].astype(np.float32), xs, ys, interpolation=cv2.INTER_LINEAR)
        g_c = cv2.remap(img[:, :, 1].astype(np.float32), xs, ys, interpolation=cv2.INTER_LINEAR)
        b_c = cv2.remap(img[:, :, 2].astype(np.float32), xs, ys, interpolation=cv2.INTER_LINEAR)
        
        # H-alpha prominence signal: R excess over green/blue
        prom = np.maximum(0, r_c - 0.5 * (g_c + b_c))
        # Max across radial band
        profile = np.max(prom, axis=0)
        
        # Let's smooth the profile slightly
        prof_smooth = cv2.GaussianBlur(profile[:, None], (15, 1), 3.0).ravel()
        
        # Find local peaks that are prominent (> 2000 counts)
        peaks = []
        for i in range(1, n_angles - 1):
            if prof_smooth[i] > prof_smooth[i-1] and prof_smooth[i] > prof_smooth[i+1]:
                if prof_smooth[i] > 1000:
                    ang_deg = i * (360.0 / n_angles)
                    peaks.append((ang_deg, prof_smooth[i]))
                    
        peaks.sort(key=lambda p: p[1], reverse=True)
        print(f"\n{fname}: Center=({cx:.1f}, {cy:.1f}), R={r:.1f}")
        print(f"Top 5 prominence peaks (angle in degrees 0-360, intensity):")
        for ang, val in peaks[:5]:
            clock_pos = (ang / 30.0 + 3.0) % 12.0
            print(f"  Angle: {ang:5.1f}° (~{clock_pos:4.1f} o'clock) | Peak Val: {val:6.0f}")

if __name__ == "__main__":
    map_prominences()

