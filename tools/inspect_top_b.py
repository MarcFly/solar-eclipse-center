import cv2
import numpy as np

zoom = cv2.imread("../samples/step4_rotation_alignment/prominence_alignment_zoom.jpg")
top_b = zoom[0:240, 0:240]
top_a = zoom[0:240, 240:480]

r_b = top_b[35:, :, 2]
b_b = top_b[35:, :, 0]
g_b = top_b[35:, :, 1]

print(f"top_b: R min={r_b.min()}, max={r_b.max()} | G min={g_b.min()}, max={g_b.max()} | B min={b_b.min()}, max={b_b.max()}")
print(f"Top 5 locations where R == 255 in top_b:")
ys, xs = np.where(r_b == 255)
for y, x in zip(ys[:5], xs[:5]):
    print(f"  (x={x}, y={y}): R={r_b[y, x]}, G={g_b[y, x]}, B={b_b[y, x]}")

print(f"\nTop 5 locations where B == 255 in top_b:")
ys, xs = np.where(b_b == 255)
for y, x in zip(ys[:5], xs[:5]):
    print(f"  (x={x}, y={y}): R={r_b[y, x]}, G={g_b[y, x]}, B={b_b[y, x]}")

