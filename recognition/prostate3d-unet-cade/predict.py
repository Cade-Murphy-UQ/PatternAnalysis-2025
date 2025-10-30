import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split

from modules import UNet3D
from dataset import load_from_folders
from train import train_val_test_save



IMG_DIR = "HipMRI_Study_open/semantic_MRs"
MSK_DIR = "HipMRI_Study_open/semantic_labels_only"
CLASSES = 6
CKPT_PATH = "unet3d.pt"
EPOCHS = 5
BATCH_SIZE = 1
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2
AUG_PROB = 0.25

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

if __name__ == "__main__":
    main()