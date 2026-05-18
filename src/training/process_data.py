

import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from shapely.geometry import Point
from tqdm import tqdm

from rasterio.transform import from_bounds
from rasterio.features import rasterize
import xarray
import rioxarray as rio
import matplotlib.pyplot as plt
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio
import numpy as np
import xarray as xr

BASE_DIR = Path(__file__).resolve().parent.parent.parent
IMERG_DIR = BASE_DIR / "data" / "imerg"
GOES_DIR = BASE_DIR / "data" / "goes" / "combined" /  "goes17"
DEM_DIR = BASE_DIR / "data" / "elevation" / "geostack_30m_topo.tif"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
HCPD_DIR = BASE_DIR / "data" / "hcdp" / "rainfall_new_day_statewide_data_map_2021_01_01.tif" # this is for resolution conversion

def clip_prep(match_raster_path):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with rio.open_rasterio(match_raster_path) as src:
        # Clip to the Big Island
        big_island = src.rio.clip_box(
            minx=-156.07,
            miny=18.89,
            maxx=-154.799,
            maxy=20.277,
        )

        # Remove nodata values
        nodata_value = big_island.rio.nodata

        # Mask out the nodata values in each band of big_island
        big_island = big_island.where(big_island != nodata_value)

        # Save DEM raster in processed data folder
        dem_path = PROCESSED_DIR / 'big_island_dem.tif'
        big_island.rio.to_raster(dem_path)

    return big_island, dem_path

def resample_precip(precip_raster_path, dem_raster_path, goes_raster_path, res_raster_path, date_str: str, output_raster=None):
    if output_raster is None:
        output_raster = f'processed_input_{date_str}.tif'

    goes_data = xr.open_dataset(goes_raster_path)
    temp_goes_dir = Path('./temp_goes_data')
    temp_goes_dir.mkdir(exist_ok=True)
    goes_bcm_data = goes_data['BCM'].values
    profile = {
        'driver': 'GTiff',
        'height': goes_bcm_data.shape[0],
        'width': goes_bcm_data.shape[1],
        'count': 1,
        'dtype': goes_bcm_data.dtype,
        'crs': 'EPSG:4326', # Default to WGS84 for demonstration, adjust as needed
        'transform': rasterio.transform.from_origin(0, goes_bcm_data.shape[0], 1, 1) # Dummy transform
    }
    temp_goes_filepath = temp_goes_dir / 'goes_bcm_temp.tif'
    with rasterio.open(temp_goes_filepath, 'w', **profile) as dst:
        dst.write(goes_bcm_data, 1)
    _, match_path = clip_prep(res_raster_path)
    with rasterio.open(match_path) as res_match:
            dst_crs = res_match.crs
            dst_transform = res_match.transform
            dst_width = res_match.width
            dst_height = res_match.height

            with (
                rasterio.open(precip_raster_path) as src_precip,
                rasterio.open(temp_goes_filepath) as src_goes,
                rasterio.open(dem_raster_path) as src_dem
            ):
                # Determine the total number of bands for the output raster
                # precip_bands + goes_band + elev_band + slope_band + aspect_band
                total_output_bands = 5 # 1 for precip 1 for GOES, 3 for DEM components

                kwargs = src_precip.meta.copy()
                kwargs.update({
                    'crs': dst_crs,
                    'transform': dst_transform,
                    'width': dst_width,
                    'height': dst_height,
                    'count': total_output_bands,
                    'dtype': rasterio.float32,
                })
                output_path = PROCESSED_DIR / 'input' / output_raster
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Reproject and align
                with rasterio.open(output_path, 'w', **kwargs) as dst:
                    current_band_idx = 0
                    for i in range(1, src_precip.count + 1):
                        current_band_idx += 1
                        reproject(
                            source=rasterio.band(src_precip, i),
                            destination=rasterio.band(dst, current_band_idx),
                            src_transform=src_precip.transform,
                            src_crs=src_precip.crs,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.nearest,
                        )
                # Reproject and write GOES data (assuming band 1 from goes_raster_path)
                    current_band_idx += 1
                    goes_data_reprojected = np.zeros((dst_height, dst_width), dtype=rasterio.float32)
                    reproject(
                        source=rasterio.band(src_goes, 1),
                        destination=goes_data_reprojected,
                        src_transform=src_goes.transform,
                        src_crs=src_goes.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest,
                    )
                    dst.write_band(current_band_idx, goes_data_reprojected)

                    # Reproject and write DEM data (elevation, slope, aspect)
                    # Assuming dem_raster_path contains elevation (band 1), slope (band 2), aspect (band 3)
                    dem_bands_to_process = [
                        {'source_band': 1, 'output_name': 'elevation'},
                        {'source_band': 2, 'output_name': 'slope'},
                        {'source_band': 3, 'output_name': 'aspect'}
                    ]

                    for dem_info in dem_bands_to_process:
                        current_band_idx += 1
                        dem_band_reprojected = np.zeros((dst_height, dst_width), dtype=rasterio.float32)
                        reproject(
                            source=rasterio.band(src_dem, dem_info['source_band']),
                            destination=dem_band_reprojected,
                            src_transform=src_dem.transform,
                            src_crs=src_dem.crs,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.bilinear, # Changed to bilinear
                        )
                        dst.write_band(current_band_idx, dem_band_reprojected)
        
                

    return output_path

def extract_imerg_dates_to_csv(imerg_dir=None, output_csv=None):
    """
    Extract dates in YYYYMMDD format from IMERG filenames and create a CSV file.

    Args:
        imerg_dir (str or Path, optional): Path to IMERG directory. Defaults to RAW_DIR/IMERG
        output_csv (str or Path, optional): Output CSV path. Defaults to PROCESSED_DIR/imerg_dates.csv

    Returns:
        Path: Path to the created CSV file
    """
    if imerg_dir is None:
        imerg_dir = IMERG_DIR
    else:
        imerg_dir = Path(imerg_dir)

    if output_csv is None:
        output_csv = PROCESSED_DIR / "imerg_dates.csv"
    else:
        output_csv = Path(output_csv)

    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Get all .tif files in the IMERG directory
    imerg_files = list(imerg_dir.glob("*.tif"))

    # Extract dates from filenames
    dates = []
    for file_path in imerg_files:
        filename = file_path.name
        # Find the date pattern: 8 digits after "3IMERG."
        # Pattern: 3B-HHR-E.MS.MRG.3IMERG.YYYYMMDD-S233000-E235959.1410.V07B.1day.tif
        if "3IMERG." in filename:
            date_start = filename.find("3IMERG.") + len("3IMERG.")
            date_str = filename[date_start:date_start + 8]
            # Validate it's 8 digits
            if date_str.isdigit() and len(date_str) == 8:
                dates.append(date_str)

    # Sort dates
    dates.sort()

    # Create DataFrame and save to CSV
    df = pd.DataFrame({"date": dates})
    df.to_csv(output_csv, index=False)

    print(f"Extracted {len(dates)} dates from {len(imerg_files)} IMERG files")
    print(f"Saved to: {output_csv}")

    return output_csv

def main(csv_path=None):
    """Load dataset from CSV file."""
    if csv_path is None:
        csv_path = PROCESSED_DIR / "imerg_dates.csv"
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    catalog = pd.read_csv(csv_path)['date']

    #process each precip raster and save to processed folder
    skipped = []
    for date in tqdm(catalog, desc="Processing IMERG and station data", unit="date"):
        date_str = str(date)  # Ensure date is a string for filename operations
        year = date_str[:4]
        month = date_str[4:6]
        day = date_str[6:8]
        julian_day = pd.to_datetime(date).dayofyear
        julian_day_str = f"{julian_day:03d}"  # Format as three digits with leading zeros
        date_str = str(date)  # Convert to string for filename operations
        precip_raster_path = IMERG_DIR / f"3B-HHR-E.MS.MRG.3IMERG.{date_str}-S233000-E235959.1410.V07B.1day.tif"
        match_raster_path = HCPD_DIR
        dem_raster_path = DEM_DIR
        goes_raster_path = GOES_DIR / year / f"goes_{year}_{julian_day_str}.nc"
        output_raster = f'processed_precip_{date_str}.tif'

        # Skip dates where any required input file is missing
        missing = [p for p in [precip_raster_path, dem_raster_path, goes_raster_path] if not Path(p).exists()]
        if missing:
            skipped.append(date_str)
            tqdm.write(f"  Skipping {date_str} — missing: {[str(p) for p in missing]}")
            continue

        try:
            output_path = resample_precip(precip_raster_path, dem_raster_path, goes_raster_path, match_raster_path, date_str, output_raster)
        except Exception as e:
            skipped.append(date_str)
            tqdm.write(f"  Skipping {date_str} — error: {e}")

    if skipped:
        print(f"\nSkipped {len(skipped)} date(s): {skipped}")

main()