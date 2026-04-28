"""
AnyMC3D-Compatible Preprocessing Pipeline (v4)
===============================================
Adds resampling and foreground cropping on top of v3, plus a choice
between z-score normalization and percentile clipping.

Pipeline (applied in this order):
  1. Load raw .nii.gz + read voxel spacing from header
  2. Crop to nonzero bounding box (configurable margin)
  3. Normalize — one of:
       z-score  : (x - mean) / std computed on foreground voxels only
       percentile: ScaleIntensityRangePercentiles [lower, upper] → [0, 1]
  4. Resample to target spacing via MONAI Spacing (trilinear for image)
  5. Add channel dim → (1, H, W, S)
  6. (Optional) Resize to target_size³ via trilinear interpolation
  7. Save as float32 .npy

Order rationale:
  Crop before normalize: removes background padding so foreground stats
    (mean/std for z-score; percentiles for clipping) are not corrupted
    by large zero regions outside the body.
  Normalize before resample: the nonzero foreground mask must align
    perfectly with the image. Resampling first introduces interpolated
    non-zero values at the boundary, corrupting the mask and therefore
    the normalization statistics. Normalization MUST happen before
    resampling.

Target spacing:
  If --target_spacing is given (e.g. "1.0,1.0,1.0"), that is used directly.
  Otherwise a two-pass approach is used:
    Pass 1 — scan all NIfTI headers to compute the per-axis median spacing
    Pass 2 — resample every volume to that median spacing

Usage examples:
  # Z-score, auto median spacing, no resize:
  python preprocess.py \\
      --input_dir  /data/BMLMPS_FLAIR/imagesTr \\
      --output_dir /data/BMLMPS_FLAIR/preprocessed

  # Percentile clip (0.5–99.5), fixed spacing 1mm iso, resize to 256³:
  python preprocess.py \\
      --input_dir   /data/PDCAD \\
      --output_dir  /data/PDCAD_preprocessed \\
      --norm        percentile \\
      --lower_pct   0.5 --upper_pct 99.5 \\
      --target_spacing 1.0,1.0,1.0 \\
      --target_size 256

Requirements:
    pip install nibabel numpy tqdm monai scipy
"""

import argparse
import json
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple, List

import nibabel as nib
import numpy as np
from tqdm import tqdm

from monai.transforms import (
    NormalizeIntensity,
    Resize,
    ScaleIntensityRangePercentiles,
)
from scipy.ndimage import zoom


# ── Constants ────────────────────────────────────────────────────────────────

DTYPE = np.float32


# ── Step 1: Load ─────────────────────────────────────────────────────────────

def load_nifti(path: Path) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """
    Load a NIfTI file, reorient to RAS+ canonical orientation, and
    return the image data together with its voxel spacing.

    Returns:
        data   : np.ndarray (H, W, S) float32
        spacing: tuple of 3 floats (mm per voxel, H / W / S axes)
    """
    img           = nib.load(str(path))
    img_canonical = nib.as_closest_canonical(img)
    spacing       = tuple(float(s) for s in img_canonical.header.get_zooms()[:3])
    data          = img_canonical.get_fdata(dtype=np.float32)
    return data, spacing


# ── Step 2: Resample ─────────────────────────────────────────────────────────

def resample_volume(
    data: np.ndarray,
    src_spacing: Tuple[float, ...],
    tgt_spacing: Tuple[float, ...],
) -> np.ndarray:
    """
    Resample a volume from src_spacing to tgt_spacing using
    scipy.ndimage.zoom (trilinear, order=1).

    Zoom factors are computed as src_spacing / tgt_spacing per axis:
    a larger target spacing means downsampling (zoom < 1) and a smaller
    target spacing means upsampling (zoom > 1).

    Returns:
        resampled: np.ndarray (H', W', S') float32
    """
    zoom_factors = tuple(s / t for s, t in zip(src_spacing, tgt_spacing))

    if np.allclose(zoom_factors, 1.0, atol=1e-3):
        return data  # nothing to do

    resampled = zoom(data, zoom_factors, order=1)  # order=1 = trilinear
    return resampled.astype(DTYPE)


# ── Step 3: Crop ─────────────────────────────────────────────────────────────

def crop_to_nonzero(data: np.ndarray, margin: int = 4) -> np.ndarray:
    """
    Crop the volume to the bounding box of non-zero voxels, with an
    optional margin on all sides (clamped to image boundaries).

    A margin of 4 voxels is recommended to protect thin structures at
    the edges of the foreground (e.g. substantia nigra, thin cortex).

    Args:
        data   : (H, W, S) float32
        margin : extra voxels to keep around the bounding box

    Returns:
        cropped: (H', W', S') float32  (same dtype, smaller or equal shape)
    """
    nz = np.argwhere(data != 0)
    if len(nz) == 0:
        return data  # entirely zero — return as-is

    lo = np.maximum(nz.min(axis=0) - margin, 0)
    hi = np.minimum(nz.max(axis=0) + margin + 1, np.array(data.shape))
    return data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]


# ── Step 4a: Z-score normalization ───────────────────────────────────────────

def zscore_normalize(data: np.ndarray) -> np.ndarray:
    """
    Per-volume z-score normalization on foreground voxels only,
    using MONAI NormalizeIntensity.

    nonzero=True computes mean/std on non-zero voxels only and sets
    background back to 0 after normalizing — equivalent to a foreground
    z-score for MRI.

    Returns:
        normalized: float32 array, same shape as input.
                    Foreground is z-scored; background is 0.
    """
    transform = NormalizeIntensity(
        nonzero=True,
        channel_wise=False,
    )
    out = transform(data)
    return np.asarray(out, dtype=DTYPE)


# ── Step 4b: Percentile normalization ────────────────────────────────────────

def percentile_normalize(
    data: np.ndarray,
    lower_pct: float = 0.5,
    upper_pct: float = 99.5,
) -> np.ndarray:
    """
    Percentile clipping followed by rescaling to [0, 1].

    Uses MONAI ScaleIntensityRangePercentiles, which computes the
    percentiles from the full volume, clips, and linearly rescales.
    Preferred for modalities where extreme intensities carry pathological
    signal (e.g. NM-MRI, some PET).

    Returns:
        normalized: float32 array in [0, 1], same shape as input.
    """
    transform = ScaleIntensityRangePercentiles(
        lower=lower_pct,
        upper=upper_pct,
        b_min=0.0,
        b_max=1.0,
        clip=True,
        relative=False,
    )
    out = transform(data)
    return np.asarray(out, dtype=DTYPE)


# ── Step 6: Resize ───────────────────────────────────────────────────────────

def resize_volume(data: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize a channel-first volume (1, H, W, S) to
    (1, target_size, target_size, target_size) using trilinear interpolation.
    """
    transform = Resize(
        spatial_size=(target_size, target_size, target_size),
        mode="trilinear",
    )
    out = transform(data)
    return np.asarray(out, dtype=DTYPE)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def preprocess_volume(
    path:        Path,
    tgt_spacing: Tuple[float, ...],
    norm:        str,
    lower_pct:   float,
    upper_pct:   float,
    crop_margin: int,
    target_size: Optional[int],
) -> np.ndarray:
    """
    Run the full preprocessing pipeline for one volume.

    Returns:
        np.ndarray  shape (1, H, W, S) or (1, T, T, T),  float32
    """
    # 1. Load + get spacing
    data, src_spacing = load_nifti(path)

    # 2. Crop to nonzero bounding box (before normalize so foreground
    #    stats are not diluted by large zero-padded background regions)
    data = crop_to_nonzero(data, margin=crop_margin)

    # 3. Normalize (before resample — nonzero mask must align perfectly
    #    with the image; resampling first corrupts the mask boundary)
    if norm == "zscore":
        data = zscore_normalize(data)
    else:
        data = percentile_normalize(data, lower_pct, upper_pct)

    # 4. Resample to target spacing
    data = resample_volume(data, src_spacing, tgt_spacing)

    # 5. Add channel dim → (1, H, W, S)
    data = data[np.newaxis, ...]

    # 6. Optional spatial resize
    if target_size is not None:
        data = resize_volume(data, target_size)

    return data


# ── Pass 1: collect spacings ──────────────────────────────────────────────────

def compute_median_spacing(nifti_files: List[Path]) -> Tuple[float, float, float]:
    """
    Scan NIfTI headers across the dataset and return the per-axis
    median voxel spacing — used as the resampling target.
    """
    spacings = []
    for p in tqdm(nifti_files, desc="Reading spacings"):
        try:
            img           = nib.load(str(p))
            img_canonical = nib.as_closest_canonical(img)
            spacings.append(img_canonical.header.get_zooms()[:3])
        except Exception as e:
            warnings.warn(f"Could not read spacing from {p.name}: {e}")

    if not spacings:
        raise RuntimeError("No valid NIfTI files found to compute median spacing.")

    arr = np.array(spacings, dtype=np.float32)   # (N, 3)
    median = tuple(float(np.median(arr[:, i])) for i in range(3))
    print(f"\nMedian spacing computed from {len(spacings)} volumes: "
          f"{median[0]:.4f} × {median[1]:.4f} × {median[2]:.4f} mm")
    return median


# ── Worker (subprocess) ───────────────────────────────────────────────────────

def process_one(args):
    (nifti_path, output_path,
     tgt_spacing, norm, lower_pct, upper_pct,
     crop_margin, target_size) = args
    try:
        vol = preprocess_volume(
            path        = nifti_path,
            tgt_spacing = tgt_spacing,
            norm        = norm,
            lower_pct   = lower_pct,
            upper_pct   = upper_pct,
            crop_margin = crop_margin,
            target_size = target_size,
        )
        np.save(str(output_path), vol)
        return str(nifti_path.name), vol.shape, None
    except Exception as e:
        return str(nifti_path.name), None, str(e)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_preprocessing(
    input_dir:       Path,
    output_dir:      Path,
    norm:            str   = "zscore",
    lower_pct:       float = 0.5,
    upper_pct:       float = 99.5,
    target_spacing:  Optional[Tuple[float, float, float]] = None,
    crop_margin:     int   = 4,
    target_size:     Optional[int] = None,
    pattern:         str   = "*.nii.gz",
    workers:         int   = 4,
    verify:          bool  = False,
):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(input_dir.rglob(pattern))
    if not nifti_files:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return

    # ── Pass 1: determine target spacing ─────────────────────────────────────
    if target_spacing is None:
        print("\nNo --target_spacing given. Computing median spacing from dataset...")
        tgt_spacing = compute_median_spacing(nifti_files)
    else:
        tgt_spacing = target_spacing
        print(f"\nUsing user-supplied target spacing: "
              f"{tgt_spacing[0]:.4f} × {tgt_spacing[1]:.4f} × {tgt_spacing[2]:.4f} mm")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nAnyMC3D Preprocessing Pipeline v4")
    print(f"==================================================")
    print(f"Input dir:       {input_dir}")
    print(f"Output dir:      {output_dir}")
    print(f"Files found:     {len(nifti_files)}")
    print(f"Workers:         {workers}")
    print(f"Target spacing:  {tgt_spacing[0]:.4f} × {tgt_spacing[1]:.4f} × {tgt_spacing[2]:.4f} mm")
    print(f"Crop margin:     {crop_margin} voxels")
    if norm == "zscore":
        print(f"Normalization:   Z-score (foreground voxels only)")
    else:
        print(f"Normalization:   Percentile clip [{lower_pct}, {upper_pct}] → [0, 1]")
    if target_size:
        print(f"Final resize:    {target_size}³ (trilinear)")
    else:
        print(f"Final resize:    None (volumes keep post-resample shape)")
    print()

    # ── Pass 2: preprocess ────────────────────────────────────────────────────
    work = []
    for nifti_path in nifti_files:
        rel      = nifti_path.relative_to(input_dir)
        out_name = rel.with_suffix('').with_suffix('.npy')
        out_path = output_dir / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        work.append((
            nifti_path, out_path,
            tgt_spacing, norm, lower_pct, upper_pct,
            crop_margin, target_size,
        ))

    results, errors, shapes = [], [], {}

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

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Done: {len(results)} succeeded, {len(errors)} failed")

    if shapes:
        unique_shapes: dict = {}
        for s in shapes.values():
            key = str(s)
            unique_shapes[key] = unique_shapes.get(key, 0) + 1
        print(f"\nOutput shapes (post-preprocessing):")
        for shape_str, count in sorted(unique_shapes.items()):
            print(f"  {shape_str}  →  {count} file(s)")

    if errors:
        print(f"\nFailed files:")
        for name, err in errors:
            print(f"  {name}: {err}")

    # ── Manifest ──────────────────────────────────────────────────────────────
    manifest = {
        "pipeline_version": "v4",
        "input_dir":        str(input_dir),
        "output_dir":       str(output_dir),
        "target_spacing_mm": list(tgt_spacing),
        "crop_margin_vox":  crop_margin,
        "normalization":    norm,
        "percentile_lower": lower_pct if norm == "percentile" else None,
        "percentile_upper": upper_pct if norm == "percentile" else None,
        "target_size":      target_size,
        "n_processed":      len(results),
        "n_failed":         len(errors),
        "output_shapes":    {k: list(v) for k, v in shapes.items()},
        "errors":           {n: e for n, e in errors},
    }
    manifest_path = output_dir / "preprocessing_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved → {manifest_path}")

    # ── Optional verification ─────────────────────────────────────────────────
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
        description="AnyMC3D preprocessing (crop → normalize → resample)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Root folder containing raw .nii.gz files (searched recursively)"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Folder where preprocessed .npy files will be saved"
    )

    # ── Normalization ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--norm", choices=["zscore", "percentile"], default="zscore",
        help=(
            "zscore     : foreground-masked z-score (recommended for structural MRI "
            "e.g. T1, T2, FLAIR). "
            "percentile : ScaleIntensityRangePercentiles → [0,1] (recommended for "
            "modalities with pathological hotspots e.g. NM-MRI, PET)."
        )
    )
    parser.add_argument(
        "--lower_pct", type=float, default=0.5,
        help="Lower percentile for --norm percentile (ignored for zscore)"
    )
    parser.add_argument(
        "--upper_pct", type=float, default=99.5,
        help="Upper percentile for --norm percentile (ignored for zscore)"
    )

    # ── Resampling ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--target_spacing", type=str, default=None,
        help=(
            "Target voxel spacing in mm, as 'H,W,S' e.g. '1.0,1.0,1.0'. "
            "If not set, the per-axis median spacing is computed from the "
            "dataset."
        )
    )

    # ── Cropping ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--crop_margin", type=int, default=4,
        help="Voxel margin around the nonzero bounding box when cropping"
    )

    # ── Resize ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--target_size", type=int, default=None,
        help="If set, resize all volumes to target_size³ after normalization"
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--pattern",  default="*.nii.gz",
                        help="Glob pattern for finding NIfTI files")
    parser.add_argument("--workers",  type=int, default=4,
                        help="Number of parallel worker processes")
    parser.add_argument("--verify",   action="store_true",
                        help="Print stats for the first 3 outputs after processing")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Parse target_spacing string → tuple
    tgt_spacing = None
    if args.target_spacing:
        parts = [float(x) for x in args.target_spacing.split(",")]
        if len(parts) != 3:
            raise ValueError("--target_spacing must be three comma-separated floats, e.g. '1.0,1.0,1.0'")
        tgt_spacing = tuple(parts)

    run_preprocessing(
        input_dir      = args.input_dir,
        output_dir     = args.output_dir,
        norm           = args.norm,
        lower_pct      = args.lower_pct,
        upper_pct      = args.upper_pct,
        target_spacing = tgt_spacing,
        crop_margin    = args.crop_margin,
        target_size    = args.target_size,
        pattern        = args.pattern,
        workers        = args.workers,
        verify         = args.verify,
    )