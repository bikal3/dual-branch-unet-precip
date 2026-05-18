"""
Dual Branch UNet — Training Script
Usage:
    python src/training/train-dual-branch-unet.py
    python src/training/train-dual-branch-unet.py --train-start 20200101 --train-end 20201231 \
                    --val-start 20210101   --val-end 20210331   \
                    --epochs 50 --batch-size 64 --lr 1e-3
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.models.unet import DualBranchUNet, DualBranchLoss
from src.training.load_dataset import LoadDataset


# ── Device ─────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ── Argument parsing ────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='Train Dual Branch UNet')
    parser.add_argument('--train-start',  default='20200101', help='Training start date YYYYMMDD')
    parser.add_argument('--train-end',    default='20201231', help='Training end date YYYYMMDD')
    parser.add_argument('--val-start',    default='20210101', help='Validation start date YYYYMMDD')
    parser.add_argument('--val-end',      default='20210331', help='Validation end date YYYYMMDD (default: end of Mar 2021)')
    parser.add_argument('--test-start',   default='20210401', help='Test start date YYYYMMDD (default: Apr 2021)')
    parser.add_argument('--test-end',     default='20210630', help='Test end date YYYYMMDD (default: end of Jun 2021)')
    parser.add_argument('--chip-size',    type=int,   default=32,   help='Chip size in pixels')
    parser.add_argument('--batch-size',   type=int,   default=64,   help='Batch size')
    parser.add_argument('--num-workers',  type=int,   default=8,    help='DataLoader worker processes')
    parser.add_argument('--epochs',       type=int,   default=50,   help='Number of training epochs')
    parser.add_argument('--lr',           type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lambda-mse',   type=float, default=1.0,  help='Weight for masked MSE loss')
    parser.add_argument('--lambda-tv',    type=float, default=0.1,  help='Weight for TV loss')
    parser.add_argument('--csv-path',     default=None, help='Path to file_paths.csv (default: data/processed/file_paths.csv)')
    parser.add_argument('--save-path',    default=None, help='Checkpoint save path (default: models/dual_branch_unet.pth)')
    parser.add_argument('--no-plot',      action='store_true', help='Disable saving the loss curve plot')
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    csv_path  = Path(args.csv_path)  if args.csv_path  else BASE_DIR / 'data' / 'processed' / 'file_paths.csv'
    save_path = Path(args.save_path) if args.save_path else BASE_DIR / 'models' / 'dual_branch_unet.pth'

    print(f'Device      : {device}')
    print(f'Batch size  : {args.batch_size}')
    print(f'Workers     : {args.num_workers}')
    print(f'Epochs      : {args.epochs}')
    print(f'LR          : {args.lr}')
    print(f'Train dates : {args.train_start} → {args.train_end}')
    print(f'Val dates   : {args.val_start} → {args.val_end}')
    print(f'Test dates  : {args.test_start} → {args.test_end}')
    print(f'CSV         : {csv_path}')
    print(f'Checkpoint  : {save_path}')
    print()

    # ── Datasets ───────────────────────────────────────────────────────────
    train_ds = LoadDataset(
        src_dir=BASE_DIR,
        csv_path=csv_path,
        start_date=args.train_start,
        end_date=args.train_end,
        chip_size=args.chip_size,
        transform=True,
        apply_normalization=True,
    )
    val_ds = LoadDataset(
        src_dir=BASE_DIR,
        csv_path=csv_path,
        start_date=args.val_start,
        end_date=args.val_end,
        chip_size=args.chip_size,
        transform=False,
        apply_normalization=True,
    )
    test_ds = LoadDataset(
        src_dir=BASE_DIR,
        csv_path=csv_path,
        start_date=args.test_start,
        end_date=args.test_end,
        chip_size=args.chip_size,
        transform=False,
        apply_normalization=True,
    )
    print(f'Train chips : {len(train_ds)} | Val chips: {len(val_ds)} | Test chips: {len(test_ds)}')

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=True,
    )

    # ── Model, loss, optimizer ─────────────────────────────────────────────
    model     = DualBranchUNet(branch1_in=2, branch2_in=3).to(device)
    criterion = DualBranchLoss(masked_mse_weight=args.lambda_mse, tv_weight=args.lambda_tv)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters  : {n_params:,}\n')

    # ── Training loop ──────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_epoch    = 1

    for epoch in range(args.epochs):
        # Train
        model.train()
        epoch_loss = epoch_mse = epoch_tv = 0.0
        for inputs, targets, masks in train_loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            masks   = masks.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss, mse_loss, tv_loss = criterion(preds, targets, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_mse  += mse_loss.item()
            epoch_tv   += tv_loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        # Validate
        model.eval()
        val_loss = val_mse = val_tv = 0.0
        with torch.no_grad():
            for inputs, targets, masks in val_loader:
                inputs  = inputs.to(device)
                targets = targets.to(device)
                masks   = masks.to(device)
                loss, mse_loss, tv_loss = criterion(model(inputs), targets, masks)
                val_loss += loss.item()
                val_mse  += mse_loss.item()
                val_tv   += tv_loss.item()
        val_losses.append(val_loss / len(val_loader))

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_epoch    = epoch + 1
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)

        print(f'Epoch {epoch+1:>3}/{args.epochs}  '
              f'train={train_losses[-1]:.4f} (mse={epoch_mse/len(train_loader):.4f} tv={epoch_tv/len(train_loader):.4f})  '
              f'val={val_losses[-1]:.4f} (mse={val_mse/len(val_loader):.4f} tv={val_tv/len(val_loader):.4f})'
              f'{"  ✓ best" if val_losses[-1] == best_val_loss else ""}')

    # ── Test evaluation ────────────────────────────────────────────────────
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets, masks in test_loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            masks   = masks.to(device)
            loss, _, _ = criterion(model(inputs), targets, masks)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f'\nTest loss: {test_loss:.4f}')

    print(f'\nBest checkpoint (epoch {best_epoch}, val={best_val_loss:.4f}) saved to {save_path}')

    # ── Loss curve plot ────────────────────────────────────────────────────
    if not args.no_plot:
        plot_path = BASE_DIR / 'docs' / 'loss_curve.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses,   label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Dual Branch UNet — Loss Curves')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f'Loss curve saved to {plot_path}')


if __name__ == '__main__':
    main()
