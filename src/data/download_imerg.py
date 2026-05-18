import re
import time
from pathlib import Path
from datetime import date, timedelta

import requests
from tqdm import tqdm

PPS_BASE = "https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/early"
_FNAME_RE = re.compile(
    r"3B-HHR-E\.MS\.MRG\.3IMERG\.(\d{8})-S\d{6}-E\d{6}\.\d{4}\.V\d{2}[A-Z]\.1day\.tif"
)

_MAX_RETRIES = 5
_RETRY_BACKOFF = [5, 15, 30, 60, 120]  # seconds between retries


def _get_with_retry(session: requests.Session, url: str, timeout: int) -> requests.Response:
    """GET with exponential-backoff retry on timeout or server errors."""
    for attempt, wait in enumerate(_RETRY_BACKOFF):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout):
            if attempt == _MAX_RETRIES - 1:
                raise
            print(f"\n  Timeout on attempt {attempt + 1}, retrying in {wait}s...")
            time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code in (429, 500, 502, 503, 504):
                if attempt == _MAX_RETRIES - 1:
                    raise
                print(f"\n  HTTP {e.response.status_code} on attempt {attempt + 1}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {_MAX_RETRIES} attempts: {url}")


def build_imerg_url(year: int, month: int, filename: str) -> str:
    return f"{PPS_BASE}/{year:04d}/{month:02d}/{filename}"


def list_imerg_day_filename(date_str: str, html: str) -> str | None:
    """Return the last .tif filename matching date_str (YYYYMMDD) from a PPS directory HTML page.

    Multiple 1day.tif files exist per day (running accumulations at each 30-min window).
    The last one (highest minute offset, e.g. 1410 = 23:30 UTC) is the full 24-hour total.
    """
    last = None
    for m in _FNAME_RE.finditer(html):
        if m.group(1) == date_str:
            last = m.group(0)
    return last


def download_imerg_date_range(
    start: date,
    end: date,
    out_dir: Path,
    pps_email: str,
    pps_password: str,
) -> list[Path]:
    """Download IMERG Early Run daily GeoTIFFs for every day in [start, end].

    Skips files that already exist. Retries on timeout/server errors with
    exponential backoff (up to 5 attempts per request).

    Args:
        start: first date (inclusive).
        end:   last date (inclusive).
        out_dir: destination directory.
        pps_email:    PPS account email (used as HTTP Basic Auth username).
        pps_password: PPS account password.

    Returns:
        List of paths to successfully downloaded files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.auth = (pps_email, pps_password)

    dir_cache: dict[tuple[int, int], str] = {}
    downloaded: list[Path] = []
    current = start

    pbar = tqdm(total=(end - start).days + 1, desc="IMERG download")
    while current <= end:
        ym = (current.year, current.month)
        if ym not in dir_cache:
            url = f"{PPS_BASE}/{current.year:04d}/{current.month:02d}/"
            try:
                r = _get_with_retry(session, url, timeout=60)
                dir_cache[ym] = r.text
            except Exception as e:
                pbar.write(f"  WARNING: could not fetch directory for {current.year}-{current.month:02d}: {e}")
                # Skip entire month
                while current.month == ym[1] and current <= end:
                    current += timedelta(days=1)
                    pbar.update(1)
                continue

        date_str = current.strftime("%Y%m%d")
        filename = list_imerg_day_filename(date_str, dir_cache[ym])
        if filename is None:
            pbar.write(f"  WARNING: no IMERG file for {date_str}, skipping")
            current += timedelta(days=1)
            pbar.update(1)
            continue

        out_path = out_dir / filename
        if not out_path.exists():
            url = build_imerg_url(current.year, current.month, filename)
            try:
                r = _get_with_retry(session, url, timeout=120)
                out_path.write_bytes(r.content)
            except Exception as e:
                pbar.write(f"  WARNING: failed to download {filename}: {e}")
                current += timedelta(days=1)
                pbar.update(1)
                continue

        downloaded.append(out_path)
        current += timedelta(days=1)
        pbar.update(1)

    pbar.close()
    return downloaded
