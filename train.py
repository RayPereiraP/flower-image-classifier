import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from torch import nn

from src.dataset import get_loaders
from src.model import build_model

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "outputs"


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if training:
        model.features.eval()  # mantém o BatchNorm do backbone congelado

    total_loss, preds_all, labels_all = 0.0, [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds_all.extend(outputs.argmax(1).cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

    n = len(labels_all)
    acc = sum(p == y for p, y in zip(preds_all, labels_all)) / n
    return total_loss / n, acc, preds_all, labels_all


def main(args):
    OUT_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    train_loader, val_loader, class_names = get_loaders(
        args.data_dir, args.batch_size
    )
    print(f"Classes: {class_names}")

    model = build_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )

    best_acc, best_preds, best_labels, history = 0.0, [], [], []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, device, criterion, optimizer
        )
        val_loss, val_acc, preds, labels = run_epoch(
            model, val_loader, device, criterion
        )
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })
        print(f"Época {epoch}/{args.epochs} | "
              f"treino: loss {train_loss:.3f} acc {train_acc:.3f} | "
              f"validação: loss {val_loss:.3f} acc {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc, best_preds, best_labels = val_acc, preds, labels
            torch.save(
                {"state_dict": model.state_dict(), "class_names": class_names},
                OUT_DIR / "model.pt",
            )

    print(f"\nMelhor acurácia na validação: {best_acc:.3f}\n")
    print(classification_report(best_labels, best_preds, target_names=class_names))

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({"best_val_accuracy": round(best_acc, 4), "history": history}, f, indent=2)

    ConfusionMatrixDisplay.from_predictions(
        best_labels, best_preds, display_labels=class_names,
        xticks_rotation=45, cmap="Blues",
    )
    plt.title("Matriz de confusão (validação)")
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(BASE_DIR / "data" / "flowers"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    main(parser.parse_args())