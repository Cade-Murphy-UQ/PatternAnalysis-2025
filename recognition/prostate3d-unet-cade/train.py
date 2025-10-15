import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from modules import UNet3D
from dataset import load_from_folders
import numpy as np
import matplotlib.pyplot as plt

IMG_DIR = "HipMRI_Study_open/semantic_MRs"
MSK_DIR = "HipMRI_Study_open/semantic_labels_only"

X, Y = load_from_folders(IMG_DIR, MSK_DIR)

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

model = UNet3D(in_channels=1, out_channels=6).to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

train_losses, val_losses = [], []

#training loop
for epoch in range(5):
    model.train()
    total_train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.long().to(device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
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
            loss = loss_fn(out, y)
            total_val_loss += loss.item()
    val_loss = total_val_loss / len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}")


#Plotting Curves
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.close()



def dice_coefficient(logits, target, num_classes=6, eps=1e-6):
    pred = torch.argmax(logits, dim=1)
    dice_scores = []

    for c in range(num_classes):
        pred_mask = (pred == c).to(torch.float32)
        true_mask = (target == c).to(torch.float32)

        intersection = torch.sum(pred_mask * true_mask)
        union = torch.sum(pred_mask) + torch.sum(true_mask)

        dice = (2 * intersection + eps) / (union + eps)
        dice_scores.append(dice)

    return torch.mean(torch.stack(dice_scores))

# Dice coeffecient Testing
model.eval()
test_loss, test_dice, batches = 0.0, 0.0, 0

print("Testing")
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        test_loss += float(loss_fn(logits, y))
        test_dice += float(dice_coefficient(logits, y, num_classes=6))
        batches += 1

if batches > 0:
    test_loss /= batches
    test_dice /= batches
    print(f"Loss: {test_loss}, Dice: {test_dice}")
else:
    print("No samples in test split.")