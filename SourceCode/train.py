import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import config
from dataset import get_dataloaders
from regularization import get_l1_weight, get_l2_weight
from model_baseline import CNNBaseline
from model_cbam import CNN_CBAM

SUPPORTED_MODELS = ["baseline", "cbam", "resnet18"]
PRETRAINED_MODELS = {"resnet18"}


def get_device():
    return torch.device("cuda" if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu")


def get_model(name):
    if name == "cbam":
        return CNN_CBAM()
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
        return model
    return CNNBaseline()


def train_epoch(model, loader, criterion, optimizer, device, l1_factor=0.0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if l1_factor != 0.0:
            l1_penalty = sum(param.abs().sum() for param in model.parameters() if param.requires_grad)
            loss = loss + l1_factor * l1_penalty
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return (total_loss / total) if total else 0.0, (correct / total) if total else 0.0


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return (total_loss / total) if total else 0.0, (correct / total) if total else 0.0


def evaluate_with_metrics(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = (total_loss / total) if total else 0.0
    accuracy = (correct / total) if total else 0.0
    return avg_loss, accuracy, all_labels, all_preds


def confusion_matrix_manual(y_true, y_pred, labels):
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def classification_report_manual(y_true, y_pred, labels):
    cm = confusion_matrix_manual(y_true, y_pred, labels)
    report_lines = []
    header = f"{'label':<12} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>8}"
    report_lines.append(header)
    report_lines.append('-' * len(header))

    total_support = 0
    total_correct = 0
    precision_sum = 0.0
    recall_sum = 0.0
    support_sum = 0

    for idx, label in enumerate(labels):
        tp = cm[idx][idx]
        support = sum(cm[idx])
        pred_positive = sum(row[idx] for row in cm)
        precision = tp / pred_positive if pred_positive > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        report_lines.append(f"{label:<12} {precision:9.4f} {recall:9.4f} {f1:9.4f} {support:8d}")
        total_support += support
        total_correct += tp
        precision_sum += precision * support
        recall_sum += recall * support
        support_sum += support

    accuracy = total_correct / total_support if total_support > 0 else 0.0
    macro_precision = sum(cm[idx][idx] / (sum(row[idx] for row in cm) or 1) if sum(row[idx] for row in cm) > 0 else 0.0 for idx in range(len(labels))) / len(labels)
    macro_recall = sum(cm[idx][idx] / (sum(cm[idx]) or 1) if sum(cm[idx]) > 0 else 0.0 for idx in range(len(labels))) / len(labels)
    macro_f1 = sum(
        (2 * (cm[idx][idx] / (sum(row[idx] for row in cm) or 1)) * (cm[idx][idx] / (sum(cm[idx]) or 1)) /
         ((cm[idx][idx] / (sum(row[idx] for row in cm) or 1)) + (cm[idx][idx] / (sum(cm[idx]) or 1))) if ((cm[idx][idx] / (sum(row[idx] for row in cm) or 1)) + (cm[idx][idx] / (sum(cm[idx]) or 1))) > 0 else 0.0)
        for idx in range(len(labels))
    ) / len(labels)

    report_lines.append('-' * len(header))
    report_lines.append(f"{'accuracy':<12} {accuracy:9.4f} {'':>9} {'':>9} {total_support:8d}")
    report_lines.append(f"{'macro avg':<12} {macro_precision:9.4f} {macro_recall:9.4f} {macro_f1:9.4f} {total_support:8d}")
    return '\n'.join(report_lines), cm


def plot_confusion_matrix(y_true, y_pred, labels, save_path="confusion_matrix.png"):
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
    except ImportError:
        cm = confusion_matrix_manual(y_true, y_pred, labels)
        print("Confusion matrix:")
        for label, row in zip(labels, cm):
            print(f"{label:<12}", ' '.join(str(x) for x in row))
        print("\nInstall matplotlib to save a plotted confusion matrix: pip install matplotlib")
        return

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")


def save_model(model, model_name):
    os.makedirs("saved_models", exist_ok=True)
    path = os.path.join("saved_models", f"{model_name}_best.pth")
    torch.save(model.state_dict(), path)
    return path


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, model_name):
    os.makedirs("saved_models", exist_ok=True)
    path = os.path.join("saved_models", f"{model_name}_latest.pth")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_acc": best_val_acc
    }, path)
    return path


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        epoch = checkpoint.get("epoch", 0)
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
    else:
        model.load_state_dict(checkpoint)
        epoch = 0
        best_val_acc = 0.0
    model.to(device)
    return model, optimizer, scheduler, epoch, best_val_acc


def load_model(model, checkpoint_path, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train and test facial expression models")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="resnet18",
                        help="Chọn model để đào tạo / kiểm tra")
    parser.add_argument("--train", action="store_true", help="Huấn luyện model")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục huấn luyện từ checkpoint hiện có")
    parser.add_argument("--test", action="store_true", help="Kiểm tra model trên tập test")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Đường dẫn model đã lưu để tải và kiểm tra / resume")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help="Số epoch để huấn luyện")
    parser.add_argument("--lr", type=float, default=config.LR,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help="Kích thước batch")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    if not args.train and not args.test:
        raise ValueError("Bạn phải chọn --train hoặc --test (hoặc cả hai).")

    if args.batch_size != config.BATCH_SIZE:
        print(f"Cập nhật batch size: {config.BATCH_SIZE} -> {args.batch_size}")
        config.BATCH_SIZE = args.batch_size

    train_loader, val_loader, test_loader = get_dataloaders(args.model)
    model = get_model(args.model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    best_checkpoint = None
    best_val_acc = 0.0

    if args.train:
        l2_weight = get_l2_weight()
        l1_weight = get_l1_weight()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=l2_weight)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        start_epoch = 1
        if args.resume:
            if args.checkpoint is None:
                raise ValueError("Để resume, hãy cung cấp --checkpoint <path>")
            if not os.path.exists(args.checkpoint):
                raise FileNotFoundError(f"Không tìm thấy checkpoint: {args.checkpoint}")
            model, optimizer, scheduler, loaded_epoch, best_val_acc = load_checkpoint(
                model, optimizer, scheduler, args.checkpoint, device
            )
            start_epoch = loaded_epoch + 1
            print(f"Resume từ epoch {start_epoch} với checkpoint {args.checkpoint}")

        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, l1_factor=l1_weight)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            latest_checkpoint = save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_acc, args.model
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_checkpoint = save_model(model, args.model)

            print(
                f"Epoch {epoch}/{args.epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                f"best_val_acc={best_val_acc:.4f}, "
                f"latest_checkpoint={os.path.basename(latest_checkpoint)}"
            )

        print(f"Best validation model saved: {best_checkpoint}")

    if args.test:
        if args.checkpoint is not None:
            checkpoint_path = args.checkpoint
        elif best_checkpoint is not None:
            checkpoint_path = best_checkpoint
        else:
            raise ValueError("Để kiểm tra mà không huấn luyện, hãy cung cấp --checkpoint <path>")

        model = get_model(args.model)
        model = load_model(model, checkpoint_path, device)
        test_loss, test_acc, y_true, y_pred = evaluate_with_metrics(model, test_loader, criterion, device)
        print(f"Kết quả test: loss={test_loss:.4f}, accuracy={test_acc:.4f}")
        print("\nClassification report:")
        report_text, _ = classification_report_manual(
            y_true, y_pred,
            ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        )
        print(report_text)
        plot_confusion_matrix(y_true, y_pred, [
            "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
        ])


if __name__ == "__main__":
    main()
