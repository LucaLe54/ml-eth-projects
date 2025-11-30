"""
Training script for a U-Net based binary segmentation model.

The goal is to detect mugs in images. The script:
- builds a U-Net model
- wraps a base dataset with a custom preprocessing pipeline
- trains with BCEWithLogitsLoss
- saves checkpoints and the best model
- optionally saves predictions as PNG masks + RLE CSV

The base dataset class `ETHMugsDataset` is assumed to be provided externally.
"""

import argparse
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from eth_mugs_dataset import ETHMugsDataset  # course-provided dataset
from utils import save_predictions_as_images  # use the cleaned utils I sent you earlier
# from utils import save_predictions_as_imgs   # if you keep the original name


# -----------------------------
# Model definition (U-Net)
# -----------------------------


class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscaling with maxpool then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Up(nn.Module):
    """Upscaling then DoubleConv with skip connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # pad if necessary to match skip connection spatial dims
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Standard U-Net architecture for binary segmentation."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)  # logits (no sigmoid here)
        # use torch.sigmoid outside when needed


# -----------------------------
# Data & preprocessing
# -----------------------------


def custom_preprocess(img: Image.Image) -> torch.Tensor:
    """
    Convert an RGB image into a 4-channel tensor:
    - 3 channels RGB
    - 1 channel gradient magnitude (Sobel filter)
    """
    rgb = TF.to_tensor(img)  # (3, H, W)

    gray = TF.to_grayscale(img, num_output_channels=1)
    gray_t = TF.to_tensor(gray)  # (1, H, W)
    gray_np = np.array(gray_t.squeeze() * 255, dtype=np.uint8)

    # Sobel edge detection
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32,
    ).view(1, 1, 3, 3)

    gray_t = TF.to_tensor(Image.fromarray(gray_np))  # (1, H, W)
    gx = nn.functional.conv2d(gray_t.unsqueeze(0), sobel_x, padding=1)
    gy = nn.functional.conv2d(gray_t.unsqueeze(0), sobel_y, padding=1)
    grad = torch.sqrt(gx ** 2 + gy ** 2).squeeze(0).squeeze(0)  # (H, W)

    # normalize gradient to [0, 1]
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)

    return torch.cat([rgb, grad.unsqueeze(0)], dim=0)  # (4, H, W)


class WrappedDataset(Dataset):
    """
    Wraps a base dataset and applies a custom transform to the image.
    The mask is converted to a float tensor in [0, 1].
    """

    def __init__(self, base_dataset: Dataset, transform) -> None:
        self.base = base_dataset
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, mask = self.base[idx]
        image = self.transform(image)
        mask = self.to_tensor(mask)
        return image, mask


def get_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 8,
) -> Tuple[DataLoader, DataLoader]:
    base_train = ETHMugsDataset(train_dir, mode="train", transform=None)
    train_dataset = WrappedDataset(base_train, transform=custom_preprocess)
    test_dataset = ETHMugsDataset(test_dir, mode="test", transform=custom_preprocess)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader


# -----------------------------
# Metrics
# -----------------------------


def calculate_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute IoU between predicted logits and binary target mask.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets > 0.5).float()

    intersection = (preds * targets).sum().float()
    union = preds.sum() + targets.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


# -----------------------------
# Training loop
# -----------------------------


def train_and_predict(
    train_dir: str,
    test_dir: str,
    output_folder: str = "predictions",
    num_epochs: int = 40,
    batch_size: int = 8,
    lr: float = 3e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    train_loader, test_loader = get_dataloaders(train_dir, test_dir, batch_size=batch_size)

    model = UNet(in_channels=4, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float("inf")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        num_batches = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device).float()

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_iou += calculate_iou(logits.detach(), masks.detach())
            num_batches += 1

        epoch_loss = running_loss / max(1, num_batches)
        epoch_iou = running_iou / max(1, num_batches)

        print(f"[Epoch {epoch + 1:03d}/{num_epochs}] "
              f"Loss: {epoch_loss:.4f}  IoU: {epoch_iou:.4f}")

        # save checkpoint for every epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"unet_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # track best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] Saved new best model to {best_model_path}")

    print("[INFO] Training finished.")
    print("[INFO] Generating predictions on test set...")

    # load best model for prediction
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    save_predictions_as_images(
        model=model,
        dataloader=test_loader,
        folder=output_folder,
        device=device,
        threshold=0.7,
    )

    print(f"[INFO] Done. Results saved in '{output_folder}/'.")


# -----------------------------
# Entry point
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net for mug segmentation.")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training dataset.")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test dataset.")
    parser.add_argument("--output_folder", type=str, default="predictions", help="Folder for predictions.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_predict(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_folder=args.output_folder,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
