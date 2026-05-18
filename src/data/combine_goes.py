"""Combine 4 daily GOES ACMF snapshots (00, 06, 12, 18 UTC) into one .nc file per day
using max aggregation across the 4 time steps.

Output layout:
    <out_dir>/<satellite>/<year>/<doy>/goes_<year>_<doy>.nc

Each output file contains:
    BCM  – shape (y, x)  Max Binary Cloud Mask over 4 synoptic times  int8
    DQF  – shape (y, x)  Max Data Quality Flag over 4 synoptic times  int8
    y    – shape (y,)    geostationary grid row indices                int16
    x    – shape (x,)    geostationary grid column indices             int16

All projection / geolocation attributes from the first valid file are preserved.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
from tqdm import tqdm

TARGET_HOURS = (0, 6, 12, 18)
GOES_BASE = Path(__file__).resolve().parents[2] / "data" / "GOES"


def combine_day(
    day_dir: Path,
    out_path: Path,
    hours: tuple[int, ...] = TARGET_HOURS,
    overwrite: bool = False,
) -> bool:
    """Max-aggregate 4 hourly BCM/DQF arrays for one day into a single NetCDF.

    Returns True if the file was written, False if skipped.
    """
    if out_path.exists() and not overwrite:
        return False

    # Collect one file per requested hour (first file in sorted order)
    hour_files: dict[int, Path] = {}
    for hour in hours:
        hour_dir = day_dir / f"{hour:02d}"
        if hour_dir.is_dir():
            candidates = sorted(hour_dir.glob("*.nc"))
            if candidates:
                hour_files[hour] = candidates[0]

    if not hour_files:
        return False

    sorted_hours = sorted(hour_files)

    # Read grid shape and coords from the first file
    first_ds = nc.Dataset(hour_files[sorted_hours[0]])
    ny = first_ds.dimensions["y"].size
    nx = first_ds.dimensions["x"].size
    y_vals = first_ds.variables["y"][:]
    x_vals = first_ds.variables["x"][:]
    first_ds.close()

    # Accumulate max across all available hours
    bcm_max = np.full((ny, nx), fill_value=-1, dtype=np.int8)
    dqf_max = np.full((ny, nx), fill_value=-1, dtype=np.int8)

    ref_ds = None
    for hour in sorted_hours:
        src = nc.Dataset(hour_files[hour])
        bcm_max = np.maximum(bcm_max, src.variables["BCM"][:])
        dqf_max = np.maximum(dqf_max, src.variables["DQF"][:])
        if ref_ds is None:
            ref_ds = src
        else:
            src.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(out_path, "w", format="NETCDF4") as dst:
        # Dimensions
        dst.createDimension("y", ny)
        dst.createDimension("x", nx)

        # Coordinates
        yv = dst.createVariable("y", "i2", ("y",))
        yv[:] = y_vals
        if ref_ds is not None and "y" in ref_ds.variables:
            for attr in ref_ds.variables["y"].ncattrs():
                yv.setncattr(attr, ref_ds.variables["y"].getncattr(attr))

        xv = dst.createVariable("x", "i2", ("x",))
        xv[:] = x_vals
        if ref_ds is not None and "x" in ref_ds.variables:
            for attr in ref_ds.variables["x"].ncattrs():
                xv.setncattr(attr, ref_ds.variables["x"].getncattr(attr))

        # Data: BCM (max)
        bcm_v = dst.createVariable(
            "BCM", "i1", ("y", "x"),
            zlib=True, complevel=4, fill_value=-1,
        )
        bcm_v[:] = bcm_max
        bcm_v.long_name = "Binary Cloud Mask (daily max over 4 synoptic times)"
        bcm_v.flag_values = np.array([0, 1], dtype=np.int8)
        bcm_v.flag_meanings = "clear cloudy"
        if ref_ds is not None and "BCM" in ref_ds.variables:
            for attr in ref_ds.variables["BCM"].ncattrs():
                try:
                    bcm_v.setncattr(attr, ref_ds.variables["BCM"].getncattr(attr))
                except Exception:
                    pass

        # Data: DQF (max)
        dqf_v = dst.createVariable(
            "DQF", "i1", ("y", "x"),
            zlib=True, complevel=4, fill_value=-1,
        )
        dqf_v[:] = dqf_max
        dqf_v.long_name = "Data Quality Flag (daily max over 4 synoptic times)"
        if ref_ds is not None and "DQF" in ref_ds.variables:
            for attr in ref_ds.variables["DQF"].ncattrs():
                try:
                    dqf_v.setncattr(attr, ref_ds.variables["DQF"].getncattr(attr))
                except Exception:
                    pass

        # Copy projection variable
        if ref_ds is not None and "goes_imager_projection" in ref_ds.variables:
            proj_src = ref_ds.variables["goes_imager_projection"]
            proj_dst = dst.createVariable("goes_imager_projection", "i4")
            for attr in proj_src.ncattrs():
                proj_dst.setncattr(attr, proj_src.getncattr(attr))

        # Global attributes
        dst.Conventions = "CF-1.7"
        dst.title = "GOES ABI-L2-ACMF daily max aggregation (4 synoptic times)"
        dst.aggregation = "max"
        dst.source_hours_UTC = str(sorted_hours)
        dst.n_times_aggregated = len(sorted_hours)
        dst.history = (
            f"Created {datetime.now(timezone.utc).isoformat()} by combine_goes.py"
        )
        if ref_ds is not None:
            for attr in ("platform_ID", "instrument_type", "scene_id",
                         "spatial_resolution", "orbital_slot"):
                try:
                    dst.setncattr(attr, ref_ds.getncattr(attr))
                except Exception:
                    pass

    if ref_ds is not None:
        ref_ds.close()

    return True


def combine_all(
    goes_dir: Path = GOES_BASE,
    out_dir: Path | None = None,
    overwrite: bool = False,
) -> None:
    """Walk all satellite/year/doy directories and max-aggregate each day."""
    goes_dir = Path(goes_dir)
    if out_dir is None:
        out_dir = goes_dir / "combined"

    # Collect all day directories (goes_dir/<satellite>/<year>/<doy>)
    day_dirs = sorted(
        p for p in goes_dir.glob("*/*/*")
        if p.is_dir() and p.parent.parent.parent == goes_dir
    )

    written = skipped = 0
    for day_dir in tqdm(day_dirs, desc="Combining GOES days", unit="day"):
        satellite = day_dir.parents[1].name
        year = day_dir.parent.name
        doy = day_dir.name

        out_path = out_dir / satellite / year / f"goes_{year}_{doy}.nc"
        did_write = combine_day(day_dir, out_path, overwrite=overwrite)
        if did_write:
            written += 1
        else:
            skipped += 1

    print(f"\nDone. Written: {written}  Skipped (already exist): {skipped}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Max-aggregate 4 daily GOES snapshots into one .nc per day."
    )
    parser.add_argument(
        "--goes-dir",
        type=Path,
        default=GOES_BASE,
        help="Root GOES data directory (default: data/GOES)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output root (default: <goes-dir>/combined)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing combined files",
    )
    args = parser.parse_args()
    combine_all(goes_dir=args.goes_dir, out_dir=args.out_dir, overwrite=args.overwrite)
