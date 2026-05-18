# Dual Branch UNet — Design Spec
**Date:** 2026-05-03  
**Branch:** dual-branch-unet

---

## Overview

Implement a Dual Branch UNet model for precipitation downsampling. The model takes 5-channel 32×32 input chips and outputs a 1-channel 32×32 high-resolution precipitation prediction. Two encoder branches process different input modalities independently and merge at the bottleneck before a shared decoder reconstructs the output.

---

## Input / Output

| | Shape | Channels |
|---|---|---|
| Input | `(B, 5, 32, 32)` | Band 1: IMERG precip, Band 2: GOES BCM, Bands 3–5: DEM elevation/slope/aspect |
| Branch 1 input | `(B, 2, 32, 32)` | Bands 0–1 (precip + GOES) |
| Branch 2 input | `(B, 3, 32, 32)` | Bands 2–4 (elevation, slope, aspect) |
| Output | `(B, 1, 32, 32)` | Predicted precipitation |
| Target | `(B, 1, 32, 32)` | Rasterized station precipitation |
| Station mask | `(B, 1, 32, 32)` | Boolean — True where ground-truth measurements exist |

---

## Architecture

### Conv Block
Each Conv Block = `Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → ReLU`

### Branch 1 Encoder (precip + GOES, 2 channels)
```
Input: (B, 2, 32, 32)
ConvBlock(2 → 32)   → (B, 32, 32, 32)
MaxPool2d(2)        → (B, 32, 16, 16)
ConvBlock(32 → 64)  → (B, 64, 16, 16)
MaxPool2d(2)        → (B, 64, 8, 8)
ConvBlock(64 → 128) → (B, 128, 8, 8)
MaxPool2d(2)        → (B, 128, 4, 4)
```

### Branch 2 Encoder (topo, 3 channels)
```
Input: (B, 3, 32, 32)
ConvBlock(3 → 32)   → (B, 32, 32, 32)
MaxPool2d(2)        → (B, 32, 16, 16)
ConvBlock(32 → 64)  → (B, 64, 16, 16)
MaxPool2d(2)        → (B, 64, 8, 8)
ConvBlock(64 → 128) → (B, 128, 8, 8)
MaxPool2d(2)        → (B, 128, 4, 4)
```

### Bottleneck Fusion
```
Concatenate branch outputs: (B, 256, 4, 4)
ConvBlock(256 → 256)              → (B, 256, 4, 4)
```

### Shared Decoder (no skip connections)
```
ConvTranspose2d(256→128, 2×2) → (B, 128, 8, 8)
ConvBlock(128 → 128)
ConvTranspose2d(128→64, 2×2)  → (B, 64, 16, 16)
ConvBlock(64 → 64)
ConvTranspose2d(64→32, 2×2)   → (B, 32, 32, 32)
ConvBlock(32 → 32)
Conv2d(32 → 1, 1×1)           → (B, 1, 32, 32)
```

---

## Loss Function

```
Total Loss = λ₁ · MaskedMSE + λ₂ · TVLoss

MaskedMSE = MSE(pred[mask], target[mask])
          where mask = station_mask (True at valid station pixels)

TVLoss = mean(|pred[:,0,1:,:] - pred[:,0,:-1,:]| 
            + |pred[:,0,:,1:] - pred[:,0,:,:-1]|)
```

**Default weights:** `λ₁ = 1.0`, `λ₂ = 0.1`  
Both are configurable from the notebook.

---

## Files to Create / Modify

| File | Action | Description |
|---|---|---|
| `src/models/unet.py` | Modify | Add `ConvBlock`, `DualBranchUNet`, `DualBranchLoss` classes |
| `notebooks/train_dual_branch_unet.ipynb` | Create | Training notebook |

---

## Notebook Structure (`train_dual_branch_unet.ipynb`)

1. **Imports & config** — paths, hyperparams (lr, batch_size, epochs, chip_size, λ₁, λ₂)
2. **Dataset setup** — `LoadDataset` for train and val date ranges
3. **DataLoaders** — `torch.utils.data.DataLoader` for train/val
4. **Model, loss, optimizer** — instantiate `DualBranchUNet`, `DualBranchLoss`, `Adam`
5. **Training loop** — forward, loss, backward, optimizer step per epoch
6. **Validation loop** — compute val loss after each epoch
7. **Loss curves** — matplotlib plot of train vs. val loss
8. **Save checkpoint** — `torch.save` to `models/dual_branch_unet.pth`
9. **Inference sample** — visualize one batch: input channels, prediction, target

---

## Hyperparameter Defaults

| Param | Value |
|---|---|
| `chip_size` | 32 |
| `batch_size` | 16 |
| `lr` | 1e-3 |
| `epochs` | 50 |
| `λ₁ (masked_mse_weight)` | 1.0 |
| `λ₂ (tv_weight)` | 0.1 |
| `optimizer` | Adam |
