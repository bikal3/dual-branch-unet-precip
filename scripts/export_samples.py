#!/usr/bin/env python3
"""
Export IMERG input and model-prediction PNGs for the website demo.

Usage:
    python scripts/export_samples.py

Outputs (per date):
    website/public/samples/YYYY-MM-DD/imerg.png
    website/public/samples/YYYY-MM-DD/pred.png
"""
import sys
from pathlib import Path
import numpy as np
import rasterio
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.models.unet import DualBranchUNet

# ── Config ──────────────────────────────────────────────────────────────────
CHECKPOINT    = BASE_DIR / "models" / "dual_branch_unet.pth"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "input"
OUT_DIR       = BASE_DIR / "website" / "public" / "samples"
CHIP_SIZE     = 32

# Three dates spread across the training year (2020). Edit as needed.
SAMPLE_DATES = ["20200115", "20200310", "20200720"]
# ─────────────────────────────────────────────────────────────────────────────


def load_model(checkpoint: Path, device: torch.device) -> DualBranchUNet:
    model = DualBranchUNet(branch1_in=2, branch2_in=3)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def normalise(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo > 1e-8:
        return (arr - lo) / (hi - lo)
    return np.zeros_like(arr)


def tile_inference(
    model: DualBranchUNet,
    raster: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Tile the full raster into 32×32 chips, run inference, stitch back."""
    C, H, W = raster.shape
    pad_h = (CHIP_SIZE - H % CHIP_SIZE) % CHIP_SIZE
    pad_w = (CHIP_SIZE - W % CHIP_SIZE) % CHIP_SIZE
    padded = np.pad(raster, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    _, PH, PW = padded.shape

    # Normalise IMERG band (band 0) before inference
    padded[0] = normalise(padded[0])

    chips, positions = [], []
    for y in range(0, PH, CHIP_SIZE):
        for x in range(0, PW, CHIP_SIZE):
            chips.append(padded[:, y : y + CHIP_SIZE, x : x + CHIP_SIZE])
            positions.append((y, x))

    output = np.zeros((PH, PW), dtype=np.float32)
    BATCH = 64
    for i in range(0, len(chips), BATCH):
        batch = np.stack(chips[i : i + BATCH])  # (B, 5, 32, 32)
        tensor = torch.from_numpy(batch).float().to(device)
        with torch.no_grad():
            pred = model(tensor).squeeze(1).cpu().numpy()  # (B, 32, 32)
        for j, (y, x) in enumerate(positions[i : i + BATCH]):
            output[y : y + CHIP_SIZE, x : x + CHIP_SIZE] = pred[j]

    return output[:H, :W]


def save_png(arr: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(arr, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def export_date(
    date_str: str, model: DualBranchUNet, device: torch.device
) -> None:
    tif_path = PROCESSED_DIR / f"processed_precip_{date_str}.tif"
    if not tif_path.exists():
        print(f"  SKIP {date_str} — TIF not found at {tif_path}")
        return

    with rasterio.open(tif_path) as src:
        raster = src.read().astype(np.float32)  # (5, H, W)

    imerg_norm = normalise(raster[0])
    pred = tile_inference(model, raster, device)
    pred_norm = normalise(pred)

    date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    out_dir = OUT_DIR / date_fmt
    out_dir.mkdir(parents=True, exist_ok=True)

    save_png(imerg_norm, out_dir / "imerg.png")
    save_png(pred_norm, out_dir / "pred.png")
    print(f"  OK  {date_fmt}  →  {out_dir}")


def main() -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Loading checkpoint: {CHECKPOINT}")
    model = load_model(CHECKPOINT, device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for date_str in SAMPLE_DATES:
        print(f"Exporting {date_str} ...")
        export_date(date_str, model, device)

    print("Done.")


if __name__ == "__main__":
    main()
