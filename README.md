# Dual-Branch U-Net Precipitation Downscaling

Downscales NASA IMERG daily precipitation from 10 km to 250 m resolution over the Big Island of Hawai'i using a dual-branch convolutional neural network that fuses satellite rainfall data with high-resolution topographic inputs.

**Live demo:** [bikal3.github.io/dual-branch-unet-precip](https://bikal3.github.io/dual-branch-unet-precip)

---

## Overview

NASA's IMERG product provides global daily precipitation at 10 km resolution — too coarse to capture the steep rainfall gradients caused by Hawai'i's volcanic terrain. This project trains a Dual-Branch U-Net to learn the statistical relationship between coarse satellite precipitation and fine-scale ground truth from ~165 rain gauge stations, producing 250 m daily rainfall fields.

## Model Architecture

The model uses two independent encoder branches that each compress a 32×32 input chip to 4×4 feature maps, which are then concatenated and decoded back to 32×32:

- **Branch 1** — IMERG precipitation + GOES cloud brightness temperature (2 channels)
- **Branch 2** — DEM elevation, slope, and aspect (3 channels)
- **Bottleneck** — 256-channel fusion of both branches
- **Decoder** — three transposed convolution upsampling stages → 1-channel output

**Loss function:** station-masked MSE (λ₁ = 1.0) + total variation regularisation (λ₂ = 0.1)

## Data Sources

| Source | Description |
|--------|-------------|
| NASA IMERG | Daily precipitation at 0.1° (~10 km) |
| GOES BCM | Cloud brightness temperature (cloud mask) |
| DEM | 30 m elevation, slope, and aspect stack |
| HCDP | ~165 rain gauge stations for training supervision |

**Splits:** train 2020 · val Jan–Mar 2021 · test Apr–Jun 2021

## Repository Structure

```
├── src/
│   ├── models/
│   │   └── unet.py              # DualBranchUNet + DualBranchLoss
│   └── training/
│       ├── load_dataset.py      # Dataset + chip sampler
│       ├── process_data.py      # Input preprocessing pipeline
│       ├── create_csv.py        # Build file_paths.csv index
│       └── train-dual-branch-unet.py  # Training script
├── configs/
│   └── config-dual-branch-unet.yaml  # Data paths and training config
├── scripts/
│   └── export_samples.py        # Export demo PNGs for the website
├── notebooks/
│   ├── data-preprocessing.ipynb
│   └── dual_branch_unet.ipynb
├── models/
│   └── dual_branch_unet.pth     # Trained checkpoint
└── website/                     # Next.js portfolio site
```

## Training

```bash
python src/training/train-dual-branch-unet.py
```

Key options:

```
--epochs       Number of training epochs       (default: 50)
--batch-size   Batch size                       (default: 64)
--lr           Learning rate                    (default: 1e-3)
--lambda-mse   Masked MSE loss weight           (default: 1.0)
--lambda-tv    Total variation loss weight      (default: 0.1)
--chip-size    Spatial chip size in pixels      (default: 32)
```

The best checkpoint is saved to `models/dual_branch_unet.pth` and a loss curve is written to `docs/loss_curve.png`.

## Exporting Demo Samples

Generates viridis PNGs of IMERG input and model prediction for three sample dates, used by the interactive website demo:

```bash
python scripts/export_samples.py
```

Output: `website/public/samples/YYYY-MM-DD/{imerg,pred}.png`

## Website

A static Next.js 14 portfolio site deployed to GitHub Pages with an interactive Leaflet map for comparing raw IMERG input against the model prediction.

```bash
cd website
npm install
npm run dev        # local development
npm run build      # production build
npm run deploy     # deploy to GitHub Pages
```

## Acknowledgements

Special thanks to [Elisabeth Tappert](https://github.com/ETappert) and [Gabriela de Leon](https://github.com/gabdele) for their support throughout this project.
