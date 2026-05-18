from pathlib import Path
from datetime import date, timedelta

import requests
from tqdm import tqdm

HCDP_BASE = (
    "https://ikeauth.its.hawaii.edu/files/v2/download/public/system/"
    "ikewai-annotated-data/HCDP/production/rainfall/new/day/statewide/data_map"
)


def build_hcdp_url(d: date) -> str:
    return (
        f"{HCDP_BASE}/{d.year:04d}/{d.month:02d}/"
        f"rainfall_new_day_statewide_data_map_{d.year:04d}_{d.month:02d}_{d.day:02d}.tif"
    )


def download_hcdp_date_range(start: date, end: date, out_dir: Path) -> list[Path]:
    """Download HCDP 250 m gridded daily rainfall GeoTIFFs for [start, end].

    No authentication required — data is publicly accessible.

    Returns:
        List of paths to successfully downloaded files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    downloaded: list[Path] = []
    current = start

    pbar = tqdm(total=(end - start).days + 1, desc="HCDP download")
    while current <= end:
        fname = (
            f"rainfall_new_day_statewide_data_map_"
            f"{current.year:04d}_{current.month:02d}_{current.day:02d}.tif"
        )
        out_path = out_dir / fname
        if not out_path.exists():
            r = session.get(build_hcdp_url(current), timeout=60)
            if r.status_code == 404:
                print(f"  WARNING: HCDP not found for {current}, skipping")
                current += timedelta(days=1)
                pbar.update(1)
                continue
            r.raise_for_status()
            out_path.write_bytes(r.content)

        downloaded.append(out_path)
        current += timedelta(days=1)
        pbar.update(1)

    pbar.close()
    return downloaded
