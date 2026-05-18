#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.utils.data import DataLoader
from pathlib import Path
from torch import nn
import sys 
sys.path.append(str(Path(__file__).parent.parent.parent))
from load_dataset import LoadDataset
from src.models.danet import DualAttentionUNet
import torch 
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent.parent

def train_model(model, train_loader, val_loader, num_epochs):
  model = model.to('xpu')  # Move model to GPU if available
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.MSELoss()  # Assuming regression task for precipitation downscaling

  train_losses = []
  val_losses = []

  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, sample in enumerate(train_loader):
        # Unpack the sample: it can be (inputs, targets) or (inputs, targets, station_mask)
        if len(sample) == 7:
            inputs, targets, station_mask_batch, _, _, _, _ = sample
        else:
            inputs, targets, station_mask_batch = sample
            station_mask_batch = None # No station mask available
        inputs, targets, station_mask_batch = inputs.to('xpu'), targets.to('xpu'), station_mask_batch.to('xpu') if station_mask_batch is not None else None  # Move data to GPU if available
        # Zero the parameter gradients
        optimizer.zero_grad()
      # Debugging print to check input tensor shape
# Forward pass
        outputs = model(inputs)

        # Create a mask for valid data points (not equal to no_data_value) & not NaN & station_mask
        mask = (targets != -9999) & (~torch.isnan(targets)) # Initial mask for no-data and NaN

        if station_mask_batch is not None:
            # Combine the initial mask with the station mask
            mask = mask & station_mask_batch

        # Apply the combined mask to both outputs and targets before calculating loss
        # We need to compute loss only on valid pixels
        valid_outputs = outputs[mask]
        valid_targets = targets[mask]

        # Calculate loss only on valid pixels
        # Check if there are any valid pixels before computing loss
        if valid_outputs.numel() > 0:
            loss = criterion(valid_outputs, valid_targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            # If no valid pixels in batch, skip loss calculation and optimization
            print(f"Warning: No valid pixels in batch {i} of epoch {epoch + 1}. Skipping loss calculation.")
            continue

    # Calculate average loss for the epoch
    if len(train_loader) > 0:
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss) # Store training loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    else:
        print(f"Epoch {epoch + 1}/{num_epochs}, No batches processed, Loss: N/A")

    print("Training complete!")

    # Validation loop
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad(): # Disable gradient calculations for validation
          for i, sample in enumerate(val_loader):
            if len(sample) == 7:
                inputs, targets, station_mask_batch, _, _, _, _ = sample
            else:
                inputs, targets, station_mask_batch = sample
                station_mask_batch = None
            inputs, targets, station_mask_batch = inputs.to('xpu'), targets.to('xpu'), station_mask_batch.to('xpu') if station_mask_batch is not None else None  # Move data to GPU if available
            outputs = model(inputs)

            mask = (targets != -9999) & (~torch.isnan(targets))
            if station_mask_batch is not None:
                mask = mask & station_mask_batch

            valid_outputs = outputs[mask]
            valid_targets = targets[mask]

            if valid_outputs.numel() > 0:
                loss = criterion(valid_outputs, valid_targets)
                val_running_loss += loss.item()
            else:
                print(f"Warning: No valid pixels in validation batch {i}. Skipping loss calculation.")
                continue

    # Calculate average validation loss
    if len(val_loader) > 0:
        val_epoch_loss = val_running_loss / len(val_loader)
        val_losses.append(val_epoch_loss) # Store validation loss
        print(f"Validation Loss: {val_epoch_loss:.4f}")
    else:
        print(f"Validation Loss: N/A (No validation batches processed)")
  return train_losses, val_losses


train_data = LoadDataset(
    src_dir=BASE_DIR / "data" / "processed" / 'input',
    csv_path=BASE_DIR / "data" / "processed" / "file_paths.csv",
    start_date="20200101",
    end_date="20201231",
    chip_size=32,
    no_data_value=-9999,
    transform=True,
    apply_normalization=True
)
print("Train dataset loaded with chipping and transformations applied.")

validate_data = LoadDataset(
    src_dir=BASE_DIR / "data" / "processed" / 'input',
    csv_path=BASE_DIR / "data" / "processed" / "file_paths.csv",
    start_date="20210101",
    end_date="20210315",
    chip_size=32,
    no_data_value=-9999,
    transform=False,
    apply_normalization=True
) 
print("Validation dataset loaded with chipping and transformations applied.")

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
validate_loader = DataLoader(validate_data, batch_size=64, shuffle=False)

model = DualAttentionUNet(in_channels=5, out_channels=1, features=[8, 16, 32])

train_losses, val_losses = train_model(model, train_loader, validate_loader, num_epochs=15)

danet_losses = {
    'train': train_losses,
    'val': val_losses
}

df = pd.DataFrame(danet_losses)
plt.plot(df['train'], label='Training Set')
plt.plot(df['val'], label='Validation Set')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss - Unet')
plt.legend()
plt.show();