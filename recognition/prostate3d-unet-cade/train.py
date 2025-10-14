import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from modules import UNet3D
from dataset import load_data_3D
import numpy as np

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

# simple dataset and dataloader
ds = TensorDataset(X, Y)

# setup
device = "cuda" if torch.cuda.is_available() else "cpu"
loader = DataLoader(ds, batch_size=1, shuffle=True)

model = UNet3D(in_channels=1, out_channels=6).to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

#training loop
for epoch in range(2):
    for x, y in loader:
        x, y = x.to(device), y.long().to(device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        print("loss:", float(loss))