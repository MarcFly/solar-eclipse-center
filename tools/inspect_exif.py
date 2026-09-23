import tifffile
import json

def inspect_exif():
    fpath = "../EclipseTry2/3_Totality/_DSF5184.tif"
    with tifffile.TiffFile(fpath) as tif:
        page = tif.pages[0]
        print(f"File: {fpath}")
        print("Tags found:")
        for tag in page.tags:
            if "DateTime" in tag.name or "Exposure" in tag.name or "Model" in tag.name or "Make" in tag.name:
                print(f"  {tag.name}: {tag.value}")
                
        # Also check last frame
        fpath2 = "../EclipseTry2/3_Totality/_DSF5335.tif"
        with tifffile.TiffFile(fpath2) as tif2:
            page2 = tif2.pages[0]
            print(f"\nFile: {fpath2}")
            for tag in page2.tags:
                if "DateTime" in tag.name or "Exposure" in tag.name:
                    print(f"  {tag.name}: {tag.value}")

if __name__ == "__main__":
    inspect_exif()

