import cv2
import numpy as np

img = cv2.imread("../samples/top_prominence_fringing.jpg")
print(f"Top prominence fringing crop size: {img.shape}")

# Let's inspect the Red channel (f1_5184) and Blue channel (f3_5335)
r = img[:, :, 2]
b = img[:, :, 0]
g = img[:, :, 1]

# Where are the prominences in this crop?
# In this crop:
# crop was: comp[50:180, 320:480]
# In comp (800x800), center is (400, 400).
# y goes from 50 to 180 (above center).
# x goes from 320 to 480 (centered horizontally around 400).
# The prominence at 12 o'clock is at x=400 in comp, which is x=80 in the crop!
print(f"At x=80 in crop (x=400 in comp):")
print(f"  Red profile along y: {r[:, 80].tolist()[:15]}")
print(f"  Blue profile along y: {b[:, 80].tolist()[:15]}")

# Let's find all local peaks in Red and Blue across the crop
def find_all_peaks(chan, thresh=150):
    peaks = []
    h, w = chan.shape
    for y in range(2, h-2):
        for x in range(2, w-2):
            val = chan[y, x]
            if val > thresh:
                if val >= chan[y-1:y+2, x-1:x+2].max():
                    peaks.append((x, y, val))
    return sorted(peaks, key=lambda p: p[2], reverse=True)

print("\nRed (f1_5184) peaks in crop:")
for p in find_all_peaks(r)[:5]:
    print(f"  x={p[0]} (comp_x={320+p[0]}), y={p[1]} (comp_y={50+p[1]}), val={p[2]}")

print("\nBlue (f3_5335) peaks in crop:")
for p in find_all_peaks(b)[:5]:
    print(f"  x={p[0]} (comp_x={320+p[0]}), y={p[1]} (comp_y={50+p[1]}), val={p[2]}")


