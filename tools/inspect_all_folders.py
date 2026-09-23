import os
import glob
import tifffile

def inspect_all_folders():
    folders = [
        "../EclipseTry2/1_Totality_Contact1",
        "../EclipseTry2/2_Totality_Contact2",
        "../EclipseTry2/3_Totality",
        "../EclipseTry2/4_TotalityExit",
        "../EclipseTry2/5_PostTotality"
    ]
    for folder in folders:
        files = sorted(glob.glob(f"{folder}/*.tif*"))
        print(f"\nFolder: {folder} ({len(files)} files)")
        if files:
            first_f = files[0]
            last_f = files[-1]
            def get_dt(fp):
                try:
                    with tifffile.TiffFile(fp) as t:
                        for tag in t.pages[0].tags:
                            if "DateTime" in tag.name:
                                return tag.value
                except:
                    pass
                return "Unknown"
            print(f"  First: {os.path.basename(first_f)} ({get_dt(first_f)})")
            print(f"  Last:  {os.path.basename(last_f)} ({get_dt(last_f)})")

if __name__ == "__main__":
    inspect_all_folders()

