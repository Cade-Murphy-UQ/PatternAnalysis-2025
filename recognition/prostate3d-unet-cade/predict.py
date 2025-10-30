import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import os
import matplotlib.pyplot as plt

from modules import UNet3D
from dataset import load_from_folders
from train import train_val_test_save

IMG_DIR = "HipMRI_Study_open/semantic_MRs"
MSK_DIR = "HipMRI_Study_open/semantic_labels_only"
CLASSES = 6
CKPT_PATH = "unet3d.pt"
EPOCHS = 20
BATCH_SIZE = 1
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2
AUG_PROB = 0.25

def save_grid(model, loader, device, out_path, mode, rows=3, cols=3):
    model.eval()

    fig, axs = plt.subplots(rows, cols, figsize=(12, 12))
    axs = axs.ravel()

    shown = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            vol = x[0, 0].cpu().numpy()
            lab = y[0].cpu().numpy()

            if mode == "pred":
                overlay = torch.argmax(model(x), dim=1)[0].cpu().numpy()
            else:
                overlay = lab

            d = vol.shape[0] // 2
            ax = axs[shown]
            ax.imshow(vol[d])
            over2d = np.ma.masked_where(overlay[d] == 0, overlay[d])
            ax.imshow(over2d, alpha=0.5)

            shown += 1
            if shown >= rows * cols:
                break

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved to {out_path}")

def main():
    #load data from folders
    X, Y = load_from_folders(IMG_DIR, MSK_DIR)

    #model build
    model = UNet3D(in_channels=1, out_channels=CLASSES)

    #train, validate, test and save
    info = train_val_test_save(
        model=model,
        X_np=X, Y_np=Y,
        classes=CLASSES,
        ckpt_path=CKPT_PATH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        aug_prob=AUG_PROB,
        device=None
    )

    os.makedirs("assets", exist_ok=True)

    #Loss curve (1 - dice) still useful but same info as dice coeffecient curve
    train_losses = info["train_losses"]
    val_losses = info["val_losses"]

    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Dice)")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/loss_curve.png")
    plt.close()
    print("Saved: assets/loss_curve.png")

    #Dice coeffecient Curve
    train_dices = info["train_dices"]
    val_dices = info["val_dices"]

    plt.figure()
    plt.plot(range(1, len(train_dices)+1), train_dices, label="Train Dice")
    plt.plot(range(1, len(val_dices)+1), val_dices, label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.title("Training & Validation Dice Coefficient")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/dice_curve.png")
    plt.close()
    print("Saved: assets/dice_curve.png")

    #Grids for segmantation overlay ground truth and model prediction
    device = info["device"]
    test_loader = info["test_loader"]
    save_grid(model, test_loader, device, "assets/grid_ground_truth.png", mode="gt")
    save_grid(model, test_loader, device, "assets/grid_predicted.png", mode="pred")

if __name__ == "__main__":
    main()