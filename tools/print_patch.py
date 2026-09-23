import cv2
import numpy as np

img = cv2.imread("../samples/top_prominence_fringing.jpg")
r = img[:, :, 2]
b = img[:, :, 0]
g = img[:, :, 1]

print(f"Red at x=80, y=56: {r[56, 80]}")
print(f"Blue at x=80, y=56: {b[56, 80]}")
print(f"Green at x=80, y=56: {g[56, 80]}")

print("\nRed patch around x=80, y=56:")
print(r[50:62, 75:86])

print("\nBlue patch around x=80, y=56:")
print(b[50:62, 75:86])

