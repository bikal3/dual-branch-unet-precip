import pandas as pd
import numpy as np
import rasterio
from torchvision import transforms # Still used for Compose, which accepts any callable
import random
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F # Still used for F.pad for tensor input or can be replaced with np.pad for numpy
from pathlib import Path
from tqdm import tqdm


BASE_DIR = Path(__file__).parent.parent.parent

# NumPy compatible NormalizeFirstChannel
class NormalizeFirstChannelNumpy(object):
    """Normalize only the first channel of the input image numpy array
    using Min-Max scaling to map values to a [0, 1] range.
    Expects input to be a tuple (image_np, target_np).
    """
    def __call__(self, sample):
        image_np, target_np = sample

        # Calculate min and max from the first channel of the current image
        min_val_img = np.min(image_np[0])
        max_val_img = np.max(image_np[0])

        # Apply Min-Max normalization only to the first channel of the input
        if (max_val_img - min_val_img) > 0:
            image_np[0] = (image_np[0] - min_val_img) / (max_val_img - min_val_img)
        else:
            image_np[0] = image_np[0] - min_val_img

        # Normalize the target as well
        # Exclude no_data_value and NaN from min/max calculation for target
        valid_target_pixels = target_np[target_np != -9999] # Assuming -9999 is the no_data_value
        valid_target_pixels = valid_target_pixels[~np.isnan(valid_target_pixels)]

        if valid_target_pixels.size > 0:
            min_val_target = np.min(valid_target_pixels)
            max_val_target = np.max(valid_target_pixels)

            if (max_val_target - min_val_target) > 0:
                target_np = (target_np - min_val_target) / (max_val_target - min_val_target)
                # Re-apply no_data_value for consistency
                target_np[target_np == ((-9999 - min_val_target) / (max_val_target - min_val_target))] = -9999
            else:
                # If all valid values are the same, set them to 0 after subtracting min_val
                target_np[~np.isnan(target_np) & (target_np != -9999)] = 0.0
        
        return image_np, target_np

# NumPy compatible RandomSpatialAugmentation
class RandomSpatialAugmentationNumpy(object):
    """Apply random rotation and flipping consistently to image and target numpy arrays.
    Expects input to be a tuple (image_np, target_np).
    """
    def __call__(self, sample):
        image_np, target_np = sample

        # Random Rotation (using numpy.rot90 for 90-degree increments for simplicity)
        # For arbitrary angles, more advanced libraries like scipy.ndimage would be needed.
        angles = [0, 90, 180, 270]
        angle = random.choice(angles)
        if angle != 0:
            k = angle // 90 # Number of 90-degree rotations
            # Ensure CHW format for rotation around H,W axes
            image_np = np.rot90(image_np, k=k, axes=(1, 2)).copy()
            target_np = np.rot90(target_np, k=k, axes=(1, 2)).copy()

        # Random Horizontal Flip
        if random.random() > 0.5:
            image_np = np.flip(image_np, axis=2).copy() # axis=W
            target_np = np.flip(target_np, axis=2).copy()

        # Random Vertical Flip
        if random.random() > 0.5:
            image_np = np.flip(image_np, axis=1).copy() # axis=H
            target_np = np.flip(target_np, axis=1).copy()

        return image_np, target_np


class ChipDataset(torch.utils.data.Dataset):
    """Dataset that extracts 32x32 chips from numpy raster data.
    Handles padding and applies numpy-compatible transforms.
    """
    def __init__(self, input_raster_np, target_raster_np, chip_size=32, no_data_value=-9999, transform=None):
        # Expect NumPy arrays here
        self.input_raster = input_raster_np
        self.target_raster = target_raster_np
        self.chip_size = chip_size
        self.no_data_value = no_data_value
        self.transform = transform

        # Ensure target_raster is 3D (C, H, W) even if it's single channel
        if self.target_raster.ndim == 2:
            self.target_raster = np.expand_dims(self.target_raster, axis=0)

        _, self.H, self.W = self.input_raster.shape
        self.half_chip = chip_size // 2

        # Find valid pixel coordinates in the target raster (where no_data_value is not present AND not NaN)
        # Assuming target_raster is (1, H, W)
        valid_mask = ~np.isnan(self.target_raster[0]) # Now using np.isnan for numpy arrays
        self.valid_indices = np.argwhere(valid_mask).tolist() # Use np.argwhere for numpy arrays
        # Note: self.valid_indices contains [y, x] pairs after np.argwhere

        # print(f"Found {len(self.valid_indices)} valid pixels for chipping.") # Commented to reduce verbose output during __init__ loop
        if not self.valid_indices:
            raise ValueError("No valid data pixels found in the target raster.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        center_y, center_x = self.valid_indices[idx]

        # Calculate chip boundaries with padding consideration
        start_y = center_y - self.half_chip
        end_y = center_y + self.half_chip
        start_x = center_x - self.half_chip
        end_x = center_x + self.half_chip

        # Adjust for even chip size to include the rightmost/bottommost pixel
        if self.chip_size % 2 == 0:
            end_y -= 1
            end_x -= 1

        # Determine padding needed
        pad_left = max(0, -start_x)
        pad_right = max(0, end_x - self.W + 1)
        pad_top = max(0, -start_y)
        pad_bottom = max(0, end_y - self.H + 1)

        # Extract slices, handling negative indices for padding
        slice_start_y = max(0, start_y)
        slice_end_y = min(self.H, end_y + 1)
        slice_start_x = max(0, start_x)
        slice_end_x = min(self.W, end_x + 1)

        # Correct slicing to `slice_start_x:slice_end_x`
        input_chip_raw = self.input_raster[:, slice_start_y:slice_end_y, slice_start_x:slice_end_x]
        target_chip_raw = self.target_raster[:, slice_start_y:slice_end_y, slice_start_x:slice_end_x]

        # Apply padding using numpy.pad
        # np.pad expects ((before_axis0, after_axis0), (before_axis1, after_axis1), ...)
        pad_width_input = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
        pad_width_target = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))

        input_chip = np.pad(input_chip_raw, pad_width=pad_width_input, mode='constant', constant_values=0.0)
        target_chip = np.pad(target_chip_raw, pad_width=pad_width_target, mode='constant', constant_values=self.no_data_value)

        # Ensure the chips are exactly chip_size x chip_size
        input_chip = input_chip[:, :self.chip_size, :self.chip_size]
        target_chip = target_chip[:, :self.chip_size, :self.chip_size]

        sample = (input_chip, target_chip)

        if self.transform:
            sample = self.transform(sample)

        return sample # Returns numpy arrays




class LoadDataset(torch.utils.data.Dataset):
    """Custom PyTorch Dataset for loading and processing multiple raster pairs (input/target).
    Loads files, creates ChipDataset instances per file, and aggregates all chips.
    All intermediate processing (chipping, transforms) are done on NumPy arrays.
    Final conversion to PyTorch tensors happens in __getitem__.
    """
    def __init__(self, src_dir, csv_path, start_date, end_date, chip_size=32, no_data_value=-9999, transform=None, apply_normalization=True):
        self.src_dir = Path(src_dir) # Ensure Path object
        self.csv_path = Path(csv_path) # Ensure Path object
        self.start_date = pd.to_datetime(start_date, format='%Y%m%d')
        self.end_date = pd.to_datetime(end_date, format='%Y%m%d')
        self.no_data_value = no_data_value
        self.chip_size = chip_size
        self.apply_normalization = apply_normalization

        # Load the catalog and filter by date
        catalog = pd.read_csv(self.csv_path)
        catalog['Date'] = pd.to_datetime(catalog['date'], format='%Y%m%d') # Ensure 'Date' column is datetime
        self.catalog = catalog[(catalog['Date'] >= self.start_date) & (catalog['Date'] <= self.end_date)].reset_index(drop=True)

        # Compose NumPy-compatible transforms
        numpy_transforms_list = []
        if self.apply_normalization:
            numpy_transforms_list.append(NormalizeFirstChannelNumpy()) # Use NumPy version
        if transform: # If user provided a transform, it should be a NumPy-compatible callable
            numpy_transforms_list.append(RandomSpatialAugmentationNumpy()) # Use NumPy version (assuming `transform=True` implies this)

        # Using torchvision.transforms.Compose for convenience, it works with any callable objects.
        self.composed_numpy_transforms = transforms.Compose(numpy_transforms_list)

        # Aggregate all chips from all selected files during initialization
        self.all_chips = []
        skipped = 0
        for idx in tqdm(range(len(self.catalog)), desc="Processing files for chipping", unit="file"):
            input_file_path = self.src_dir / self.catalog['input_path'].iloc[idx]
            target_file_path = self.src_dir / self.catalog['target_path'].iloc[idx]

            if not input_file_path.exists() or not target_file_path.exists():
                skipped += 1
                continue

            with rasterio.open(input_file_path) as src_input:
                input_data_np = src_input.read().astype(np.float32)
                # Handle no_data_value if applicable, or potential -32768.0 from previous DEM reprojection
                input_data_np[input_data_np == -32768.0] = src_input.nodata if src_input.nodata is not None else -9999.0

            with rasterio.open(target_file_path) as src_target:
                target_data_np = src_target.read(1).astype(np.float32) # Read the first band (assuming single-channel target)
                target_no_data_val_local = src_target.nodata if src_target.nodata is not None else self.no_data_value
                # The ChipDataset will handle expanding dims for target if needed (if ndim=2)

            # Create a ChipDataset for the current raster pair (takes NumPy arrays, returns NumPy arrays)
            current_chip_dataset = ChipDataset(
                input_raster_np=input_data_np,
                target_raster_np=target_data_np,
                chip_size=self.chip_size,
                no_data_value=target_no_data_val_local,
                transform=self.composed_numpy_transforms # Pass NumPy-compatible transforms
            )
            # Extend self.all_chips with all chips from the current raster pair
            for i in range(len(current_chip_dataset)):
                self.all_chips.append(current_chip_dataset[i])

        loaded = len(self.catalog) - skipped
        print(f"LoadDatasetReal initialized with {len(self.all_chips)} total chips from {loaded} files "
              f"({skipped} skipped — missing files).")


    def __len__(self):
        return len(self.all_chips)

    def __getitem__(self, idx):
        # Get the pre-processed numpy chip (input_chip_np, target_chip_np)
        input_chip_np, target_chip_np = self.all_chips[idx]
        # Create station mask based on valid pixels in the target chip
        station_mask_chip_np = ((target_chip_np != self.no_data_value) & (~np.isnan(target_chip_np))).astype(bool)
       

        # Convert to torch tensors at the very end
        input_tensor = torch.from_numpy(input_chip_np).float()
        target_tensor = torch.from_numpy(target_chip_np).float() # Target typically float for regression tasks
        station_mask_tensor = torch.from_numpy(station_mask_chip_np).bool() # Convert mask to boolean tensor

        return input_tensor, target_tensor, station_mask_tensor


