#This is to create the catalog for efficient dataloading
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATE_FILE_PATH = BASE_DIR / "data" / "processed" / "imerg_dates.csv"

def generate_file_path(row):
    date_obj = str(row['date'])
    year = date_obj[:4]
    month = date_obj[4:6]
    day = date_obj[6:8]

    input_filename = f"processed_precip_{date_obj}.tif"
    target_filename = f"X{year}.{month}.{day}.tif"

    input_path = BASE_DIR / "data" / "processed" / 'input' /input_filename
    target_path = BASE_DIR / "data" / "rasterized_stations" / target_filename
    return str(input_path), str(target_path)

def create_csv(imerg_dir=None, output_csv=None):
    output_csv_path = BASE_DIR / "data" / "processed" / "file_paths.csv"
    df = pd.read_csv(DATE_FILE_PATH)
    df[['input_path', 'target_path']] = df.apply(generate_file_path, axis=1, result_type='expand')
    df.to_csv(output_csv_path, index=False)

create_csv()