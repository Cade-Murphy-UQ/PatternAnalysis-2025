import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import torch.nn.functional as F
from pyimgaug3d.augmenters import ImageSegmentationAugmenter
from pyimgaug3d.augmentation import GridWarp, Flip, Identity

#Augmentation orientation modifiers
aug = ImageSegmentationAugmenter()
aug.add_augmentation(GridWarp(grid=(4, 5, 5), max_shift=10))
aug.add_augmentation(Flip(0))
aug.add_augmentation(Identity())

def aug_once(x1: torch.Tensor, y1: torch.Tensor, num_classes: int):
    dev = x1.device

    # NumPy conversion
    x_np = x1.detach().cpu().numpy().astype(np.float32)
    x_np = np.transpose(x_np, (1, 2, 3, 0))

    y_np = y1.detach().cpu().numpy().astype(np.int32)
    y_oh = np.eye(num_classes, dtype=np.float32)[y_np]

    # paired augmentation
    aimg, aseg = aug([x_np, y_oh])

    # torch conversion
    x_aug = torch.from_numpy(np.transpose(aimg, (3, 0, 1, 2))).to(dev).type_as(x1)
    y_aug = torch.from_numpy(np.argmax(aseg, axis=-1)).to(dev).long()
    return x_aug, y_aug

def total_loss(logits, target, eps=1e-6):
    u = F.softmax(logits, dim=1)

    #one hot for segments
    classes = logits.size(1)
    v = F.one_hot(target.long(), num_classes=classes)
    v = v.permute(0, 4, 1, 2, 3).float()
    
    dims = (0, 2, 3, 4)
    inter = (u * v).sum(dims)
    denom = u.sum(dims) + v.sum(dims)
    dice_per_class = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice_per_class.mean()

def dice_coefficient(logits, target, eps=1e-6):
    pred = torch.argmax(logits, dim=1)
    C = logits.size(1)
    scores = []
    for c in range(1, C):
        pm = (pred == c).float()
        tm = (target == c).float()
        denom = pm.sum() + tm.sum()
        if denom.item() == 0:
            continue
        inter = (pm * tm).sum()
        scores.append((2.0 * inter + eps) / (denom + eps))
    
    if scores:
        return torch.mean(torch.stack(scores))
    else :
        return torch.tensor(1.0, device=logits.device)


def train_val_test_save(model, X_np, Y_np, classes, ckpt_path, epochs=5, batch_size=1, val_split=0.2, test_split=0.2, aug_prob=0.25, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    best_val = float("inf")

    # reshape to tensors
    X = torch.from_numpy(X_np).unsqueeze(1).float()
    Y = torch.from_numpy(Y_np).long()
    full_dataset = TensorDataset(X, Y)

    test_size = int(test_split * len(full_dataset))
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - test_size - val_size
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)


    train_losses, val_losses = [], []

    #training loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.long().to(device)

            # augment 25% if the time
            if torch.rand(1).item() < aug_prob:
                xa, ya = aug_once(x[0], y[0], classes)
                x = xa.unsqueeze(0)
                y = ya.unsqueeze(0)

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
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved: {ckpt_path}  (val_loss={val_loss})")

        val_losses.append(val_loss)

        print(f"Epoch {epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}")

        

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
    
    return {"train_losses": train_losses,
            "val_losses": val_losses,
            "test_loss": test_loss if batches else None,
            "test_dice": test_dice if batches else None}