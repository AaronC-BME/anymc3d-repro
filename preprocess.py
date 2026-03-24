"""
AnyMC3D-Compatible Preprocessing Pipeline (v3 — MONAI Percentile Normalization)
================================================================================
Uses MONAI's ScaleIntensityRangePercentilesd to exactly match the first author's
preprocessing for PDCAD.

Author's tip:
    ScaleIntensityRangePercentilesd(
        keys=image_keys, lower=0, upper=99.5,
        b_min=0, b_max=1, clip=True, relative=False
    )

Raw data shape:  (300, 300, 70)  — H x W x S
Output shape:    (1, 300, 300, 70) stored as float32 .npy
                 (or (1, T, T, T) if --target_size T is given)

Pipeline:
  1. Load raw .nii.gz
  2. Reorient to RAS+ canonical orientation
  3. MONAI ScaleIntensityRangePercentiles: [0th, 99.5th] percentile → [0, 1], clipped
  4. (Optional) Resize to target_size × target_size × target_size via trilinear interpolation
  5. Reshape to (1, H, W, S) channel-first for PyTorch
  6. Save as float32 .npy

Usage:
    # Without resizing (original behavior):
    python preprocess.py \\
        --input_dir  /path/to/raw/nifti/folder \\
        --output_dir /path/to/output/folder

    # With resizing to 256×256×256:
    python preprocess.py \\
        --input_dir  /path/to/raw/nifti/folder \\
        --output_dir /path/to/output/folder \\
        --target_size 256

Requirements:
    pip install nibabel numpy tqdm monai
"""

import argparse
import json
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm

# MONAI transforms — exactly what the first author used
from monai.transforms import ScaleIntensityRangePercentiles, Resize


# ── Constants ────────────────────────────────────────────────────────────────

DTYPE = np.float32
IMAGE_KEY = "image"


# ── Core preprocessing functions ─────────────────────────────────────────────

def load_nifti(path: Path) -> np.ndarray:
    """
    Load a NIfTI file and reorient to RAS+ canonical orientation.

    Returns:
        data: np.ndarray, shape (H, W, S), float32
    """
    img = nib.load(str(path))
    img_canonical = nib.as_closest_canonical(img)
    data = img_canonical.get_fdata(dtype=np.float32)
    return data


def make_percentile_transform(lower_pct: float = 0, upper_pct: float = 99.5):
    """
    Create the MONAI percentile normalization transform matching the
    first author's exact configuration:

        ScaleIntensityRangePercentilesd(
            keys=image_keys, lower=0, upper=99.5,
            b_min=0, b_max=1, clip=True, relative=False
        )

    We use the non-dict version (ScaleIntensityRangePercentiles) since
    we're applying it to a single array, but the math is identical.
    """
    return ScaleIntensityRangePercentiles(
        lower=lower_pct,
        upper=upper_pct,
        b_min=0,
        b_max=1,
        clip=True,
        relative=False,
    )


def make_resize_transform(target_size: int):
    """
    Create a MONAI Resize transform for spatial resizing to a uniform
    cubic shape: (target_size, target_size, target_size).

    Uses trilinear interpolation (mode="trilinear") to match standard
    practice for 3D medical image resizing.

    Expects input with a channel dimension: (C, H, W, S).
    """
    return Resize(
        spatial_size=(target_size, target_size, target_size),
        mode="trilinear",
    )


def preprocess_volume(
    path: Path,
    percentile_transform,
    resize_transform=None,
) -> np.ndarray:
    """
    Full preprocessing pipeline for one volume.

    Pipeline:
        Raw .nii.gz  (H, W, S)
            -> reorient to RAS canonical          (H, W, S)
            -> MONAI percentile normalize [0, 1]  (H, W, S)
            -> add channel dim                    (1, H, W, S)
            -> (optional) resize to target_size   (1, T, T, T)

    Returns:
        np.ndarray, shape (1, H, W, S) or (1, T, T, T), float32, values in [0, 1]
    """
    data = load_nifti(path)

    # Apply percentile normalization directly to array (no dict wrapping)
    data = percentile_transform(data)

    if not isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=DTYPE)
    else:
        data = data.astype(DTYPE)

    # (1, H, W, S) — channel-first for MONAI Resize and PyTorch
    data = data[np.newaxis, ...]

    # Optional spatial resize
    if resize_transform is not None:
        data = resize_transform(data)
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=DTYPE)

    return data


# ── Worker function for parallel processing ──────────────────────────────────

def process_one(args):
    """Worker function — called in subprocess."""
    nifti_path, output_path, lower_pct, upper_pct, target_size = args
    try:
        percentile_transform = make_percentile_transform(lower_pct, upper_pct)
        resize_transform = make_resize_transform(target_size) if target_size else None
        volume = preprocess_volume(nifti_path, percentile_transform, resize_transform)
        np.save(str(output_path), volume)
        return str(nifti_path.name), volume.shape, None
    except Exception as e:
        return str(nifti_path.name), None, str(e)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_preprocessing(
    input_dir:   Path,
    output_dir:  Path,
    lower_pct:   float = 0,
    upper_pct:   float = 99.5,
    target_size: Optional[int] = None,
    pattern:     str   = "*.nii.gz",
    workers:     int   = 4,
    verify:      bool  = False,
):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(input_dir.rglob(pattern))
    if not nifti_files:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return

    print(f"\nAnyMC3D Preprocessing Pipeline v3 (MONAI Percentile Normalization)")
    print(f"===================================================================")
    print(f"Input dir:       {input_dir}")
    print(f"Output dir:      {output_dir}")
    print(f"Files found:     {len(nifti_files)}")
    print(f"Workers:         {workers}")
    print(f"Transform:       ScaleIntensityRangePercentiles(")
    print(f"                     lower={lower_pct}, upper={upper_pct},")
    print(f"                     b_min=0, b_max=1, clip=True, relative=False)")
    if target_size:
        print(f"Resize:          {target_size} x {target_size} x {target_size} (trilinear)")
    else:
        print(f"\nNOTE: No spatial resizing applied here (--target_size not set).")
        print(f"      Resize to patch_size happens in the dataset.")
    print()

    # Build work list
    work = []
    for nifti_path in nifti_files:
        rel      = nifti_path.relative_to(input_dir)
        out_name = rel.with_suffix('').with_suffix('.npy')
        out_path = output_dir / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        work.append((nifti_path, out_path, lower_pct, upper_pct, target_size))

    # Process in parallel
    results = []
    errors  = []
    shapes  = {}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, w): w for w in work}
        with tqdm(total=len(work), desc="Preprocessing") as pbar:
            for future in as_completed(futures):
                name, shape, error = future.result()
                if error:
                    errors.append((name, error))
                    tqdm.write(f"  ERROR {name}: {error}")
                else:
                    results.append(name)
                    shapes[name] = shape
                pbar.update(1)

    # Summary
    print(f"\n{'='*50}")
    print(f"Done: {len(results)} succeeded, {len(errors)} failed")

    if shapes:
        unique_shapes = {}
        for s in shapes.values():
            key = str(s)
            unique_shapes[key] = unique_shapes.get(key, 0) + 1
        print(f"\nOutput shapes:")
        for shape_str, count in sorted(unique_shapes.items()):
            print(f"  {shape_str}  →  {count} file(s)")

    if errors:
        print(f"\nFailed files:")
        for name, err in errors:
            print(f"  {name}: {err}")

    # Save manifest
    manifest = {
        "input_dir":       str(input_dir),
        "output_dir":      str(output_dir),
        "normalization":   f"MONAI ScaleIntensityRangePercentiles(lower={lower_pct}, upper={upper_pct}, b_min=0, b_max=1, clip=True, relative=False)",
        "target_size":     target_size,
        "n_processed":     len(results),
        "n_failed":        len(errors),
        "output_shapes":   {k: list(v) for k, v in shapes.items()},
        "errors":          {n: e for n, e in errors},
    }
    manifest_path = output_dir / "preprocessing_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved -> {manifest_path}")

    # Optional verification
    if verify and results:
        print(f"\nVerifying first 3 outputs...")
        for name in results[:3]:
            out_path = output_dir / Path(name).with_suffix('.npy')
            if out_path.exists():
                arr = np.load(str(out_path))
                print(f"  {name}")
                print(f"    shape:   {arr.shape}")
                print(f"    dtype:   {arr.dtype}")
                print(f"    min/max: {arr.min():.4f} / {arr.max():.4f}")
                print(f"    mean:    {arr.mean():.4f}")
                print(f"    std:     {arr.std():.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="AnyMC3D preprocessing using MONAI percentile normalization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",  type=str, required=True,
        help="Root folder containing raw .nii.gz files (searched recursively)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Folder where preprocessed .npy files will be saved"
    )
    parser.add_argument(
        "--lower_pct", type=float, default=0,
        help="Lower percentile for intensity clipping"
    )
    parser.add_argument(
        "--upper_pct", type=float, default=99.5,
        help="Upper percentile for intensity clipping"
    )
    parser.add_argument(
        "--target_size", type=int, default=None,
        help="If set, resize all volumes to target_size × target_size × target_size "
             "using trilinear interpolation. If not set, volumes keep their original shape."
    )
    parser.add_argument(
        "--pattern",    type=str, default="*.nii.gz",
        help="Glob pattern for finding NIfTI files"
    )
    parser.add_argument(
        "--workers",    type=int, default=4,
        help="Number of parallel worker processes"
    )
    parser.add_argument(
        "--verify",     action="store_true",
        help="After processing, verify and print stats for first 3 outputs"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_preprocessing(
        input_dir   = args.input_dir,
        output_dir  = args.output_dir,
        lower_pct   = args.lower_pct,
        upper_pct   = args.upper_pct,
        target_size = args.target_size,
        pattern     = args.pattern,
        workers     = args.workers,
        verify      = args.verify,
    )