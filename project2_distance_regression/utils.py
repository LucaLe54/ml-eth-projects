import os
import torch
import numpy as np
import pandas as pd
from PIL import Image


def save_predictions_as_images(
    model,
    dataloader,
    folder: str = "predictions",
    device: str = "cuda",
    threshold: float = 0.7
):
    """
    Generate segmentation predictions, save them as PNG masks, and
    create a CSV file with RLE-encoded outputs.

    Args:
        model: PyTorch segmentation model.
        dataloader: DataLoader returning input images.
        folder (str): Output directory.
        device (str): "cuda" or "cpu".
        threshold (float): Threshold for binary mask.
    """
    model.eval()
    os.makedirs(folder, exist_ok=True)

    image_ids = []
    encoded_pixels = []

    with torch.no_grad():
        for idx, x in enumerate(dataloader):
            x = x.to(device)

            preds = torch.sigmoid(model(x))
            preds = (preds > threshold).float()

            mask = preds[0, 0].cpu().numpy()

            # save binary mask as PNG
            img = Image.fromarray((mask * 255).astype(np.uint8))
            img_id = f"{idx:04d}"
            img.save(os.path.join(folder, f"{img_id}_mask.png"))

            # RLE encode
            encoded = rle_encode(mask) or " "

            image_ids.append(img_id)
            encoded_pixels.append(encoded)

    # write submission file
    submission = pd.DataFrame({
        "ImageId": image_ids,
        "EncodedPixels": encoded_pixels
    })
    submission.to_csv(os.path.join(folder, "submission.csv"),
                      index=False, encoding="utf-8")

    print(f"[INFO] Saved predictions and submission to '{folder}/'")


def rle_encode(mask: np.ndarray) -> str:
    """
    Run-length encode a binary mask for Kaggle-style submission.

    Args:
        mask: 2D numpy array of 0s and 1s.

    Returns:
        RLE-encoded string.
    """
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def save_rle_submission(
    image_ids,
    masks,
    save_path: str = "submission.csv"
):
    """
    Save a list of binary masks as an RLE submission file.

    Args:
        image_ids: list of image identifiers.
        masks: list of 2D numpy arrays.
        save_path: output CSV path.
    """
    submission = {"ImageId": [], "EncodedPixels": []}

    for img_id, mask in zip(image_ids, masks):
        encoded = rle_encode(mask) or " "
        submission["ImageId"].append(img_id)
        submission["EncodedPixels"].append(encoded)

    pd.DataFrame(submission).to_csv(save_path, index=False, encoding="utf-8")
    print(f"[INFO] Saved submission to '{save_path}'")
