import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from modules import UNet3D
from dataset import load_from_folders
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

IMG_DIR = "HipMRI_Study_open/semantic_MRs"
MSK_DIR = "HipMRI_Study_open/semantic_labels_only"

X, Y = load_from_folders(IMG_DIR, MSK_DIR)

CKPT_PATH = "unet3d.pt"
best_val = float("inf")

# reshape to PyTorch format
X = torch.from_numpy(X).unsqueeze(1).float()
Y = torch.from_numpy(Y).long()


full_dataset = TensorDataset(X, Y)

train_size = int(0.6 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

# setup
device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

num_classes_found = int(torch.max(Y).item()) + 1
model = UNet3D(in_channels=1, out_channels=num_classes_found).to(device)

def total_loss(logits, target, eps=1e-6):
    # logits: [B,C,D,H,W], target: [B,D,H,W]
    u = F.softmax(logits, dim=1)
    v = torch.zeros_like(u).scatter_(1, target.unsqueeze(1), 1)
    dims = (0, 2, 3, 4)
    inter = (u * v).sum(dims)
    denom = u.sum(dims) + v.sum(dims)
    dice_per_class = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice_per_class.mean()

opt = torch.optim.Adam(model.parameters(), lr=1e-3)


train_losses, val_losses = [], []

#training loop
for epoch in range(5):
    model.train()
    total_train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.long().to(device)
        opt.zero_grad()
        out = model(x)
        loss = total_loss(out, y)
        loss.backward()
        opt.step()
        total_train_loss += loss.item()
    train_loss = total_train_loss / len(train_loader)
    train_losses.append(train_loss)

    #validate
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = total_loss(out, y)
            total_val_loss += loss.item()
    val_loss = total_val_loss / len(val_loader)

    #save chkpt
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"[Saved] {CKPT_PATH}  (val_loss={val_loss:.4f})")

    val_losses.append(val_loss)

    print(f"Epoch {epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}")


#Plotting Curves
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Dice Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.close()


def dice_coefficient(logits, target, eps=1e-6):
    pred = torch.argmax(logits, dim=1)   # [B,D,H,W]
    C = logits.size(1)
    scores = []
    for c in range(1, C):  # foreground only
        pm = (pred == c).float()
        tm = (target == c).float()
        denom = pm.sum() + tm.sum()
        if denom.item() == 0:
            continue
        inter = (pm * tm).sum()
        scores.append((2.0 * inter + eps) / (denom + eps))
    
    if scores {
        return torch.mean(torch.stack(scores))
    } else {
        torch.tensor(1.0, device=logits.device)
    }

# Dice coeffecient Testing
model.eval()
test_loss, test_dice, batches = 0.0, 0.0, 0

print("Testing")
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        test_loss += float(total_loss(logits, y))
        test_dice += float(dice_coefficient(logits, y))
        batches += 1

if batches > 0:
    test_loss /= batches
    test_dice /= batches
    print(f"Loss: {test_loss}, Dice: {test_dice}")
else:
    print("No samples in test split.")