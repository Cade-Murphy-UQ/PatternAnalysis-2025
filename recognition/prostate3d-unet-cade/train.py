import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from modules import UNet3D
from dataset import load_data_3D
import numpy as np
import matplotlib.pyplot as plt

IMG_PATHS = [
    "HipMRI_Study_open/semantic_MRs/B006_Week0_LFOV.nii.gz",
    "HipMRI_Study_open/semantic_MRs/B040_Week0_LFOV.nii.gz"
]
MSK_PATHS = [
    "HipMRI_Study_open/semantic_labels_only/B006_Week0_SEMANTIC.nii.gz",
    "HipMRI_Study_open/semantic_labels_only/B040_Week0_SEMANTIC.nii.gz"
]

# load data
X = load_data_3D(IMG_PATHS, normImage=True, dtype=np.float32)
Y = load_data_3D(MSK_PATHS, normImage=False, dtype=np.uint8)

# reshape to PyTorch format
X = torch.from_numpy(X).unsqueeze(1).float()
Y = torch.from_numpy(Y).long()


full_dataset = TensorDataset(X, Y)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

# setup
device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)

model = UNet3D(in_channels=1, out_channels=6).to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

train_losses, val_losses = [], []

#training loop
for epoch in range(10):
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

print("Saved: loss_curve.png")