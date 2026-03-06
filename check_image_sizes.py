"""
Quick script to check image sizes in the dataset directory.
Usage: python check_image_sizes.py --data_dir /path/to/data
"""
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Max cases to inspect (default: 10)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cases    = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"Found {len(cases)} cases in {data_dir}\n")

    shapes   = []
    spacings = []

    for case in cases[:args.n_samples]:
        nifti_files = list(case.rglob("*.nii.gz")) + list(case.rglob("*.nii"))
        blosc_files = list(case.rglob("*.b2nd")) + list(case.rglob("*.blosc2"))

        # Try NIfTI
        for f in nifti_files:
            try:
                import nibabel as nib
                img  = nib.load(str(f))
                shape   = img.shape
                zooms   = img.header.get_zooms()[:3]
                shapes.append(shape)
                spacings.append(zooms)
                print(f"  {case.name} / {f.name}:  shape={shape}  spacing={tuple(round(z,3) for z in zooms)}")
            except Exception as e:
                print(f"  {case.name} / {f.name}: ERROR - {e}")

        # Try Blosc2
        for f in blosc_files:
            try:
                import blosc2
                arr = blosc2.open(str(f))[:]
                shapes.append(arr.shape)
                print(f"  {case.name} / {f.name}:  shape={arr.shape}")
            except Exception as e:
                print(f"  {case.name} / {f.name}: ERROR - {e}")

    if shapes:
        print(f"\n--- Summary ({len(shapes)} files inspected) ---")
        shape_counts = Counter(shapes)
        for shape, count in shape_counts.most_common():
            print(f"  shape {shape}: {count} files")
        if spacings:
            sp = np.array(spacings)
            print(f"\n  Spacing min:  {sp.min(axis=0)}")
            print(f"  Spacing max:  {sp.max(axis=0)}")
            print(f"  Spacing mean: {sp.mean(axis=0).round(3)}")

if __name__ == "__main__":
    main()