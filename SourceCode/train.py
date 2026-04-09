import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import config
from dataset import get_dataloaders
from model_baseline import CNNBaseline
from model_cbam import CNN_CBAM


def get_device():
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(model, loader, criterion, optimizer, device):
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
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


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

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def save_model(model, model_name):
    os.makedirs("saved_models", exist_ok=True)
    path = os.path.join("saved_models", f"{model_name}.pth")
    torch.save(model.state_dict(), path)
    return path


def load_model(model, checkpoint_path, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_model(name):
    if name == "cbam":
        return CNN_CBAM()
    return CNNBaseline()


def parse_args():
    parser = argparse.ArgumentParser(description="Train and test facial expression models")
    parser.add_argument("--model", choices=["baseline", "cbam"], default="baseline",
                        help="Chọn model để đào tạo / kiểm tra")
    parser.add_argument("--train", action="store_true", help="Huấn luyện model")
    parser.add_argument("--test", action="store_true", help="Kiểm tra model trên tập test")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Đường dẫn model đã lưu để tải và kiểm tra")
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

    # Override batch size if do not match config
    if args.batch_size != config.BATCH_SIZE:
        print(f"Cập nhật batch size: {config.BATCH_SIZE} -> {args.batch_size}")
        config.BATCH_SIZE = args.batch_size

    train_loader, test_loader = get_dataloaders()
    model = get_model(args.model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    if args.train:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)
            print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        save_path = save_model(model, args.model)
        print(f"Model đã được lưu: {save_path}")

    if args.test:
        if args.checkpoint is not None:
            model = get_model(args.model)
            model = load_model(model, args.checkpoint, device)
        elif not args.train:
            raise ValueError("Để kiểm tra mà không huấn luyện, hãy cung cấp --checkpoint <path>")

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Kết quả test: loss={test_loss:.4f}, accuracy={test_acc:.4f}")


if __name__ == "__main__":
    main()
