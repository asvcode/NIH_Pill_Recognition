import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
import csv


def parse_args():
    parser = argparse.ArgumentParser(description="Train a pill classifier")
    parser.add_argument("data_dir", type=Path, help="Path to dataset root with class subfolders")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Max learning rate for OneCycle")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Directory to save model and metrics")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data used for validation")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")

    # augmentation options
    parser.add_argument("--resize", type=int, default=224, help="Resize shorter side to this size")
    parser.add_argument("--pad", type=int, default=0, help="Reflection padding size")
    parser.add_argument("--rotation", type=float, default=0.0, help="Max rotation degrees")
    parser.add_argument("--zoom", type=float, default=0.0, help="Zoom range as fraction e.g. 0.2")
    parser.add_argument("--brightness", type=float, default=0.0, help="Color jitter brightness")
    parser.add_argument("--contrast", type=float, default=0.0, help="Color jitter contrast")
    parser.add_argument("--dihedral", action="store_true", help="Apply horizontal and vertical flips")
    parser.add_argument("--cutout", type=float, default=0.0, help="Probability of random erasing")
    return parser.parse_args()


def get_dataloaders(data_dir: Path, batch_size: int, num_workers: int, args):
    train_tfms = []
    if args.pad > 0:
        train_tfms.append(transforms.Pad(args.pad, padding_mode="reflect"))
    train_tfms.append(transforms.Resize(args.resize))

    scale = (1.0 - args.zoom, 1.0 + args.zoom) if args.zoom > 0 else (1.0, 1.0)
    if args.rotation != 0.0 or args.zoom != 0.0:
        train_tfms.append(transforms.RandomAffine(degrees=args.rotation, scale=scale))

    if args.dihedral:
        train_tfms.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    else:
        train_tfms.append(transforms.RandomHorizontalFlip())

    if args.brightness > 0 or args.contrast > 0:
        train_tfms.append(transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast))

    train_tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if args.cutout > 0:
        train_tfms.append(transforms.RandomErasing(p=args.cutout))

    train_tfm = transforms.Compose(train_tfms)

    val_tfm = transforms.Compose([
        transforms.Resize(args.resize + 32),
        transforms.CenterCrop(args.resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    base_ds = datasets.ImageFolder(data_dir)
    indices = list(range(len(base_ds)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_split,
        stratify=base_ds.targets,
        random_state=42,
    )

    train_ds = datasets.ImageFolder(data_dir, transform=train_tfm)
    val_ds = datasets.ImageFolder(data_dir, transform=val_tfm)

    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(val_ds, val_idx)

    train_dl = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl, len(base_ds.classes)


def build_model(num_classes: int):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def accuracy(preds, targets):
    _, pred_labels = torch.max(preds, 1)
    return (pred_labels == targets).float().mean().item()


def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        running_acc += accuracy(outputs, labels) * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            running_acc += accuracy(outputs, labels) * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, val_dl, num_classes = get_dataloaders(args.data_dir, args.batch_size, args.num_workers, args)
    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    total_steps = args.epochs * len(train_dl)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)

    metrics = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    csv_file = args.output / "metrics.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_dl, criterion, optimizer, scheduler, device)
        val_loss, val_acc = evaluate(model, val_dl, criterion, device)

        metrics["epoch"].append(epoch)
        metrics["train_loss"].append(tr_loss)
        metrics["train_acc"].append(tr_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, tr_loss, tr_acc, val_loss, val_acc])

        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train_loss: {tr_loss:.4f} train_acc: {tr_acc:.4f} "
            f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.output / "model_best.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(
                    f"Early stopping at epoch {epoch} due to no improvement in validation accuracy"
                )
                break

    torch.save(model.state_dict(), args.output / "model_final.pth")
    with open(args.output / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
