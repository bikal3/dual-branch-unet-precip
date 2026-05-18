import time
from pathlib import Path
from datetime import date, timedelta

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

# GOES satellite coverage for Western US / Hawaii:
#   GOES-17 operational West satellite: 2019-02-12 → 2023-01-04
#   GOES-18 operational West satellite: 2023-01-04 → present
#   NOTE: GOES-17 test data exists back to mid-2018 (launched 2018-03-01).
#         If you need reliable operational data, prefer start >= 2019-02-12.
_SATELLITE_BY_YEAR = {y: "goes17" for y in range(2018, 2023)}
_SATELLITE_BY_YEAR.update({y: "goes18" for y in range(2023, 2100)})

# Four synoptic times (UTC)
DEFAULT_TARGET_HOURS = (0, 6, 12, 18)

_MAX_RETRIES = 5
_RETRY_BACKOFF = [5, 15, 30, 60, 120]


def _make_s3_client():
    return boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED),
    )


def _list_hour_keys(client, bucket: str, product: str, year: int, doy: int, hour: int) -> list[str]:
    """Return all S3 keys for a given product / year / day-of-year / hour."""
    prefix = f"{product}/{year}/{doy:03d}/{hour:02d}/"
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return sorted(keys)


def _download_with_retry(client, bucket: str, key: str, dest: Path) -> None:
    for attempt, wait in enumerate(_RETRY_BACKOFF):
        try:
            client.download_file(bucket, key, str(dest))
            return
        except Exception as e:
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(wait)


def download_goes_cloud_products(
    start: date,
    end: date,
    out_dir: Path,
    product: str = "ABI-L2-ACMF",
    target_hours: tuple[int, ...] = DEFAULT_TARGET_HOURS,
) -> list[Path]:
    """Download GOES ABI cloud-product NetCDF files from NOAA's public S3 buckets.

    Selects GOES-17 for years 2018-2022 (Western US / Hawaii coverage) and
    GOES-18 for 2023 onward. For each requested hour, the first scan of that
    UTC hour is downloaded (typically within a minute of the hour boundary).

    No AWS credentials are required — data are publicly accessible.

    Args:
        start:        First date to download (inclusive).
        end:          Last date to download (inclusive).
        out_dir:      Root output directory. Files are organised as
                      <out_dir>/<satellite>/<year>/<doy>/<hour>/<filename>.nc
        product:      ABI Level-2 product code. Common cloud products:
                        ABI-L2-ACMF  – Advanced Clear Sky Mask, Full Disk (default)
                        ABI-L2-ACHTF – Cloud Top Height, Full Disk
                        ABI-L2-CTPF  – Cloud Top Phase, Full Disk
                        ABI-L2-TPWF  – Total Precipitable Water, Full Disk
        target_hours: UTC hours to download. Default (0, 6, 12, 18) gives
                      midnight, 6 am, noon, and 6 pm snapshots.

    Returns:
        List of local paths to all successfully downloaded (or already cached) files.
    """
    out_dir = Path(out_dir)
    client = _make_s3_client()
    downloaded: list[Path] = []

    total_slots = (end - start).days + 1
    pbar = tqdm(total=total_slots, desc=f"GOES {product}", unit="day")

    current = start
    while current <= end:
        year = current.year
        doy = current.timetuple().tm_yday
        satellite = _SATELLITE_BY_YEAR.get(year, "goes18")
        bucket = f"noaa-{satellite}"

        for hour in target_hours:
            dest_dir = out_dir / satellite / f"{year}" / f"{doy:03d}" / f"{hour:02d}"
            dest_dir.mkdir(parents=True, exist_ok=True)

            try:
                keys = _list_hour_keys(client, bucket, product, year, doy, hour)
            except Exception as e:
                pbar.write(f"  WARNING: S3 listing failed {current} {hour:02d}UTC — {e}")
                continue

            if not keys:
                pbar.write(f"  WARNING: no files found for {current} {hour:02d}UTC "
                           f"({bucket}/{product})")
                continue

            # First key in sorted order = scan closest to top of hour
            key = keys[0]
            fname = Path(key).name
            dest = dest_dir / fname

            if dest.exists():
                downloaded.append(dest)
                continue

            try:
                _download_with_retry(client, bucket, key, dest)
                downloaded.append(dest)
            except Exception as e:
                pbar.write(f"  WARNING: download failed {fname} — {e}")
                dest.unlink(missing_ok=True)

        pbar.update(1)
        current += timedelta(days=1)

    pbar.close()
    return downloaded
