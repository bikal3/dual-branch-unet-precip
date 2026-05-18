"""
Download MODIS MOD09GA daily cloud state over Hawaii and save per-day cloud
fraction arrays reprojected to the 250 m reference grid.

Requires GDAL built with HDF4 support (conda-forge default).
If rasterio raises an error on HDF4 files, install libhdf4:
    conda install -c conda-forge libhdf4 hdf4
"""

from pathlib import Path
from datetime import date, timedelta

import numpy as np
import requests
import rasterio
from rasterio.warp import reproject, Resampling
from tqdm import tqdm


CMR_BASE = "https://cmr.earthdata.nasa.gov/search"
# LP DAAC Earthdata Cloud host (filters out S3 links which need AWS credentials)
LP_DAAC_HOST = "data.lpdaac.earthdatacloud.nasa.gov"


# ---------------------------------------------------------------------------
# Earthdata session — follows redirects through URS correctly
# ---------------------------------------------------------------------------

class EarthdataSession(requests.Session):
    """requests.Session that keeps credentials when redirected to URS."""

    AUTH_HOST = "urs.earthdata.nasa.gov"

    def __init__(self, username: str, password: str):
        super().__init__()
        self.auth = (username, password)

    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        if "Authorization" in headers:
            orig = requests.utils.urlparse(response.request.url)
            dest = requests.utils.urlparse(prepared_request.url)
            if (orig.hostname != dest.hostname
                    and dest.hostname != self.AUTH_HOST
                    and orig.hostname != self.AUTH_HOST):
                del headers["Authorization"]


# ---------------------------------------------------------------------------
# CMR granule search
# ---------------------------------------------------------------------------

def find_granule_urls(day: date, bbox: dict) -> list[str]:
    """Return LP DAAC .hdf download URLs for MOD09GA granules covering bbox on day."""
    params = {
        "short_name": "MOD09GA",
        "version": "061",
        "temporal": f"{day.isoformat()}T00:00:00Z,{day.isoformat()}T23:59:59Z",
        "bounding_box": (
            f"{bbox['min_lon']},{bbox['min_lat']},"
            f"{bbox['max_lon']},{bbox['max_lat']}"
        ),
        "page_size": 20,
    }
    r = requests.get(f"{CMR_BASE}/granules.json", params=params, timeout=30)
    r.raise_for_status()
    entries = r.json().get("feed", {}).get("entry", [])
    urls = []
    for entry in entries:
        for link in entry.get("links", []):
            href = link.get("href", "")
            if href.endswith(".hdf") and LP_DAAC_HOST in href:
                urls.append(href)
    return urls


# ---------------------------------------------------------------------------
# Cloud extraction from MOD09GA HDF4
# ---------------------------------------------------------------------------

def extract_cloud_mask(hdf_path: Path) -> tuple[np.ndarray, dict]:
    """Extract binary cloud mask from state_1km_1 band of a MOD09GA HDF4 file.

    Uses pyhdf for HDF4 reading + manually constructs rasterio-compatible
    metadata from the MODIS Sinusoidal projection embedded in StructMetadata.

    state_1km bits 0-1:
        00 (0) = clear
        01 (1) = cloudy
        10 (2) = mixed
        11 (3) = not set / assumed clear

    Returns:
        cloud: float32 array, 1.0=cloud/mixed, 0.0=clear/not-set
        meta: dict with crs, transform, width, height for rasterio.warp.reproject
    """
    import re
    from pyhdf.SD import SD, SDC
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    hdf = SD(str(hdf_path), SDC.READ)
    try:
        sds = hdf.select("state_1km_1")
        state = sds.get().astype(np.uint16)
        fill = sds.attributes().get("_FillValue", 65535)

        # Parse upper-left / lower-right corners from StructMetadata.0
        struct_meta = hdf.attributes().get("StructMetadata.0", "")
        ul = re.search(r"UpperLeftPointMtrs=\(([^)]+)\)", struct_meta)
        lr = re.search(r"LowerRightMtrs=\(([^)]+)\)", struct_meta)
        ul_x, ul_y = map(float, ul.group(1).split(","))
        lr_x, lr_y = map(float, lr.group(1).split(","))
    finally:
        hdf.end()

    H, W = state.shape
    # Mark fill pixels as clear (not actually observed)
    state = np.where(state == fill, np.uint16(0), state)
    cloud_state = state & 0b11
    cloud = np.where((cloud_state == 1) | (cloud_state == 2), 1.0, 0.0).astype(
        np.float32
    )

    # MODIS Sinusoidal projection (fixed sphere R=6371007.181 m)
    crs = CRS.from_proj4(
        "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 "
        "+a=6371007.181 +b=6371007.181 +units=m +no_defs"
    )
    transform = from_bounds(ul_x, lr_y, lr_x, ul_y, W, H)
    meta = {"crs": crs, "transform": transform, "width": W, "height": H}
    return cloud, meta


def _reproject_to_ref(
    arr: np.ndarray, src_meta: dict, ref_path: str
) -> np.ndarray:
    """Bilinear reproject arr to match the reference GeoTIFF grid."""
    with rasterio.open(ref_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_shape = (ref.height, ref.width)

    dst = np.zeros(dst_shape, dtype=np.float32)
    reproject(
        source=arr,
        destination=dst,
        src_transform=src_meta["transform"],
        src_crs=src_meta["crs"],
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )
    return np.clip(dst, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Main download function
# ---------------------------------------------------------------------------

def download_modis_date_range(
    start: date,
    end: date,
    out_dir: Path,
    ref_tif_path: str,
    bbox: dict,
    username: str,
    password: str,
) -> None:
    """Download MOD09GA cloud fraction for every day in [start, end].

    For each day:
      1. Query CMR for tile URLs covering bbox.
      2. Download each HDF tile (deleted after extraction).
      3. Extract cloud mask, reproject to 250 m reference grid.
      4. Merge tiles (max = any cloud counts as cloud).
      5. Save as cloud_{YYYYMMDD}.npy (float16).

    Args:
        start, end: date range inclusive.
        out_dir: directory to save cloud_{YYYYMMDD}.npy files.
        ref_tif_path: path to 250 m reference GeoTIFF (sets output grid).
        bbox: dict with min_lon, max_lon, min_lat, max_lat.
        username, password: NASA Earthdata credentials.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = EarthdataSession(username, password)

    current = start
    pbar = tqdm(total=(end - start).days + 1, desc="MODIS download")

    while current <= end:
        date_str = current.strftime("%Y%m%d")
        out_path = out_dir / f"cloud_{date_str}.npy"

        if not out_path.exists():
            try:
                urls = find_granule_urls(current, bbox)
            except Exception as e:
                pbar.write(f"  WARNING: CMR query failed for {date_str}: {e}")
                current += timedelta(days=1)
                pbar.update(1)
                continue

            if not urls:
                pbar.write(f"  WARNING: no MOD09GA granules for {date_str}")
                current += timedelta(days=1)
                pbar.update(1)
                continue

            tile_clouds = []
            for url in urls:
                hdf_path = out_dir / Path(url).name
                try:
                    if not hdf_path.exists():
                        r = session.get(url, timeout=300)
                        r.raise_for_status()
                        hdf_path.write_bytes(r.content)

                    cloud, meta = extract_cloud_mask(hdf_path)
                    reprojected = _reproject_to_ref(cloud, meta, ref_tif_path)
                    tile_clouds.append(reprojected)
                except Exception as e:
                    pbar.write(f"  WARNING: failed tile {hdf_path.name}: {e}")
                finally:
                    if hdf_path.exists():
                        hdf_path.unlink()  # delete HDF to save disk space

            if tile_clouds:
                merged = np.max(np.stack(tile_clouds, axis=0), axis=0)
                np.save(out_path, merged.astype(np.float16))

        current += timedelta(days=1)
        pbar.update(1)

    pbar.close()
