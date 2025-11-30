🧪 Project 2 — Image Segmentation (ETH Mugs Challenge)

This project was completed as part of the Machine Learning for Computer Vision course at ETH Zürich (2025).

It implements a U-Net-based image segmentation model to detect ETH mugs in cluttered indoor scenes.

🔍 Features

Custom preprocessing (RGB + Sobel edge gradients)

U-Net architecture with 4-channel input

Training pipeline: preprocessing, augmentation, training, validation

Mask generation + RLE encoding

Automatic leaderboard submission file creation

🧠 Model

Framework: PyTorch

Architecture: Modified U-Net with additional gradient channel

Loss: BCE / Dice / Tversky (configurable)

📁 Files Included

train.py – training pipeline + inference

utils.py – saving predictions, RLE encoding

unet.py (if you add it) – U-Net model

eth_mugs_dataset.py – dataset wrapper

▶️ Usage

Run training + prediction:
