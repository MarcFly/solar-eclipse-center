import tifffile
import numpy as np

files = [
    "../EclipseTry2/3_Totality/_DSF5184.tif", # idx 0
    "../EclipseTry2/3_Totality/_DSF5194.tif", # idx 10
    "../EclipseTry2/3_Totality/_DSF5204.tif", # idx 20
    "../EclipseTry2/3_Totality/_DSF5214.tif", # idx 30
    "../EclipseTry2/3_Totality/_DSF5224.tif", # idx 40
    "../EclipseTry2/3_Totality/_DSF5284.tif", # idx 100
    "../EclipseTry2/3_Totality/_DSF5335.tif", # idx 151
]

for f in files:
    img = tifffile.imread(f)
    print(f"{f.split('/')[-1]}: min={img.min()}, max={img.max()}, mean={img.mean():.1f}, >60k pixels: {(img > 60000).sum()}")

