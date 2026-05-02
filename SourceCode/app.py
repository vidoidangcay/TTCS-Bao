import os
import random
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk
try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

import config
from dataset import get_dataloaders
from model_baseline import CNNBaseline
from model_cbam import CNN_CBAM
from regularization import get_l1_weight, get_l2_weight

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SUPPORTED_MODELS = ["cbam", "resnet18"]


def get_device():
    return torch.device("cuda" if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu")


def get_model(model_name):
    if model_name == "cbam":
        return CNN_CBAM()
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
        return model
    raise ValueError(f"Model không hỗ trợ: {model_name}")


def get_transform(model_name="cbam"):
    if model_name == "resnet18":
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(config.GRAYSCALE_MEAN, config.GRAYSCALE_STD),
    ])


def load_image_tensor(image_path, model_name):
    image = Image.open(image_path).convert("RGB")
    transform = get_transform(model_name)
    return transform(image).unsqueeze(0)


def predict_single(image_path, model_name, checkpoint_path):
    device = get_device()
    model = get_model(model_name).to(device)
    load_model_checkpoint(model, checkpoint_path, device)
    model.eval()

    input_tensor = load_image_tensor(image_path, model_name).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = int(probs.argmax())
    return LABELS[pred], float(probs[pred].item())


def get_checkpoint_path(model_name):
    best_path = os.path.join("saved_models", f"{model_name}_best.pth")
    latest_path = os.path.join("saved_models", f"{model_name}_latest.pth")
    if os.path.exists(best_path):
        return best_path
    if os.path.exists(latest_path):
        return latest_path
    return None


def load_model_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    return model


def get_true_label_from_path(image_path):
    normalized = os.path.normpath(image_path)
    parts = normalized.split(os.sep)
    for part in reversed(parts):
        if part in LABELS:
            return part
    return None


class ExpressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Expression Trainer & Tester")
        self.root.geometry("1080x760")
        self.root.configure(bg="#edf2f7")

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Header.TLabel", font=("Segoe UI", 15, "bold"), background="#edf2f7")
        style.configure("Info.TLabel", font=("Segoe UI", 10), foreground="#1a5276", background="#edf2f7")
        style.configure("Card.TLabelframe", background="#ffffff")
        style.configure("Card.TLabelframe.Label", font=("Segoe UI", 11, "bold"))
        style.configure("TButton", padding=(8, 5), font=("Segoe UI", 10))
        style.configure("TLabel", background="#edf2f7")
        style.configure("Status.TLabel", foreground="#2d3436", background="#edf2f7")

        self.model_var = tk.StringVar(value=SUPPORTED_MODELS[0])
        self.epochs_var = tk.StringVar(value=str(config.EPOCHS))
        self.checkpoint_var = tk.StringVar(value="")
        self.model_status_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Sẵn sàng")
        self.image_preview = None
        self.preview_original_image = None
        self.thumbnail_refs = []
        self.cancel_event = threading.Event()

        self._build_ui()
        self.update_model_status()

    def _build_ui(self):
        title_frame = ttk.Frame(self.root, padding=(12, 10, 12, 0), style="Card.TFrame")
        title_frame.pack(fill=tk.X)
        ttk.Label(title_frame, text="Facial Expression Trainer & Tester", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(title_frame, text="Train và dự đoán ảnh biểu cảm khuôn mặt", style="Info.TLabel").pack(side=tk.RIGHT)

        control_frame = ttk.Frame(self.root, padding=(12, 10, 12, 10), style="Card.TFrame")
        control_frame.pack(fill=tk.X)
        for col in range(10):
            control_frame.columnconfigure(col, weight=0)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(3, weight=0)

        ttk.Label(control_frame, text="Chọn model:", style="Status.TLabel").grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
        model_menu = ttk.Combobox(control_frame, textvariable=self.model_var, values=SUPPORTED_MODELS, state="readonly", width=12)
        model_menu.grid(row=0, column=1, sticky=tk.W, padx=(0, 12))
        model_menu.bind("<<ComboboxSelected>>", lambda event: self.update_model_status())

        ttk.Label(control_frame, text="Epochs:", style="Status.TLabel").grid(row=0, column=2, sticky=tk.W, padx=(0, 4))
        ttk.Entry(control_frame, textvariable=self.epochs_var, width=6).grid(row=0, column=3, sticky=tk.W, padx=(0, 12))

        ttk.Button(control_frame, text="Train từ đầu", command=self.on_train_from_scratch).grid(row=0, column=4, padx=4)
        ttk.Button(control_frame, text="Train tiếp", command=self.on_train_resume).grid(row=0, column=5, padx=4)
        ttk.Button(control_frame, text="Làm mới", command=self.on_cancel_task).grid(row=0, column=6, padx=4)
        ttk.Button(control_frame, text="Chọn ảnh test", command=self.on_select_image).grid(row=0, column=7, padx=4)
        ttk.Button(control_frame, text="Random 20 ảnh test", command=self.on_random_test).grid(row=0, column=8, padx=4)
        ttk.Button(control_frame, text="Đánh giá test", command=self.on_evaluate_test).grid(row=0, column=9, padx=4)

        status_frame = ttk.Frame(self.root, padding=(12, 0, 12, 8), style="Card.TFrame")
        status_frame.pack(fill=tk.X)
        ttk.Label(status_frame, text="Trạng thái model:", style="Status.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(status_frame, textvariable=self.model_status_var, style="Info.TLabel").pack(side=tk.LEFT)

        main_frame = ttk.Frame(self.root, padding=(12, 0, 12, 12), style="Card.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        log_box = ttk.LabelFrame(main_frame, text="Log hoạt động", padding=10, style="Card.TLabelframe")
        log_box.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        self.log_text = tk.Text(log_box, wrap=tk.WORD, state=tk.DISABLED, width=60, bg="#ffffff", bd=0, relief=tk.FLAT)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        preview_box = ttk.LabelFrame(main_frame, text="Ảnh và dự đoán", padding=10, style="Card.TLabelframe")
        preview_box.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 8))
        self.preview_label = ttk.Label(preview_box, anchor="center")
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.prediction_label = ttk.Label(preview_box, text="Chưa có dự đoán", font=("Segoe UI", 12, "bold"))
        self.prediction_label.pack(padx=10, pady=6)
        self.evaluation_label = ttk.Label(preview_box, text="Chưa đánh giá test", font=("Segoe UI", 10))
        self.evaluation_label.pack(padx=10, pady=(0, 6))

        self.confusion_image = None
        self.confusion_image_label = ttk.Label(preview_box)
        self.confusion_image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.random_frame = ttk.LabelFrame(self.root, text="Random 20 ảnh test", padding=10, style="Card.TLabelframe")
        self.random_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self.random_canvas = tk.Canvas(self.random_frame, bd=0, highlightthickness=0, background="#f7f9fb")
        self.random_scroll = ttk.Scrollbar(self.random_frame, orient=tk.VERTICAL, command=self.random_canvas.yview)
        self.random_canvas.configure(yscrollcommand=self.random_scroll.set)
        self.random_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.random_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.random_inner = ttk.Frame(self.random_canvas)
        self.random_canvas.create_window((0, 0), window=self.random_inner, anchor="nw")
        self.random_inner.bind("<Configure>", lambda event: self.random_canvas.configure(scrollregion=self.random_canvas.bbox("all")))

    def append_log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def set_status(self, message):
        self.status_var.set(message)

    def on_browse_checkpoint(self):
        path = filedialog.askopenfilename(title="Chọn checkpoint", filetypes=[("PyTorch files", "*.pth"), ("All files", "*")])
        if path:
            self.checkpoint_var.set(path)
            self.append_log(f"Checkpoint đã chọn: {path}")

    def on_train_from_scratch(self):
        self._start_training(resume=False)

    def on_train_resume(self):
        self._start_training(resume=True)

    def _start_training(self, resume):
        try:
            epochs = int(self.epochs_var.get())
            if epochs <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Lỗi", "Số epoch phải là số nguyên dương.")
            return

        model_name = self.model_var.get()
        mode_text = "Train tiếp" if resume else "Train từ đầu"
        self.set_status(f"Bắt đầu {mode_text} {model_name}...")
        self.cancel_event.clear()
        self.append_log(f"[TRAIN] Bắt đầu {mode_text} model {model_name} với {epochs} epochs")
        thread = threading.Thread(target=self.train_thread, args=(model_name, epochs, resume), daemon=True)
        thread.start()

    def train_thread(self, model_name, epochs, resume=False):
        device = get_device()
        try:
            train_loader, val_loader, _ = get_dataloaders(model_name)
            model = get_model(model_name).to(device)
            criterion = nn.CrossEntropyLoss()
            l2_weight = get_l2_weight()
            l1_weight = get_l1_weight()
            optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=l2_weight)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

            start_epoch = 1
            if resume:
                checkpoint_path = self.checkpoint_var.get().strip()
                if checkpoint_path == "":
                    candidate_latest = os.path.join("saved_models", f"{model_name}_latest.pth")
                    candidate_best = os.path.join("saved_models", f"{model_name}_best.pth")
                    checkpoint_path = candidate_latest if os.path.exists(candidate_latest) else candidate_best
                if not checkpoint_path or not os.path.exists(checkpoint_path):
                    raise FileNotFoundError("Không tìm thấy checkpoint để train tiếp. Vui lòng chọn checkpoint hoặc huấn luyện trước.")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    model.load_state_dict(checkpoint["model_state"])
                    if "optimizer_state" in checkpoint and checkpoint["optimizer_state"] is not None:
                        optimizer.load_state_dict(checkpoint["optimizer_state"])
                    if "scheduler_state" in checkpoint and checkpoint["scheduler_state"] is not None:
                        scheduler.load_state_dict(checkpoint["scheduler_state"])
                    start_epoch = checkpoint.get("epoch", 0) + 1
                else:
                    model.load_state_dict(checkpoint)
                    start_epoch = 1
                self.append_log(f"[TRAIN] Resume từ checkpoint: {checkpoint_path}, bắt đầu từ epoch {start_epoch}")

            best_val_acc = 0.0
            best_path = None
            os.makedirs("saved_models", exist_ok=True)
            final_epoch = start_epoch + epochs - 1

            for epoch in range(start_epoch, final_epoch + 1):
                if self.cancel_event.is_set():
                    self.root.after(0, self.on_training_canceled)
                    return
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                for images, labels in train_loader:
                    if self.cancel_event.is_set():
                        self.root.after(0, self.on_training_canceled)
                        return
                    images = images.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    if l1_weight != 0.0:
                        l1_penalty = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
                        loss = loss + l1_weight * l1_penalty
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * images.size(0)
                    train_correct += (outputs.argmax(dim=1) == labels).sum().item()
                    train_total += labels.size(0)
                train_loss /= train_total
                train_acc = train_correct / train_total if train_total else 0.0

                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        if self.cancel_event.is_set():
                            self.root.after(0, self.on_training_canceled)
                            return
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * images.size(0)
                        val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                        val_total += labels.size(0)
                val_loss /= val_total
                val_acc = val_correct / val_total if val_total else 0.0

                scheduler.step(val_loss)
                latest_path = os.path.join("saved_models", f"{model_name}_latest.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                }, latest_path)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_path = os.path.join("saved_models", f"{model_name}_best.pth")
                    torch.save(model.state_dict(), best_path)

                self.root.after(0, lambda e=epoch, tl=train_loss, ta=train_acc, vl=val_loss, va=val_acc, fp=final_epoch, bp=best_path, se=start_epoch, tot=epochs: self.append_log(
                    f"Epoch {e-se+1}/{tot} (global {e}/{fp}): train_loss={tl:.4f}, train_acc={ta:.4f}, val_loss={vl:.4f}, val_acc={va:.4f}, best={os.path.basename(bp) if bp else 'n/a'}"
                ))

            if best_path is None:
                best_path = latest_path
            self.root.after(0, lambda: self.on_training_finished(best_path))
        except Exception as exc:
            self.root.after(0, lambda: self.on_training_error(exc))

    def on_training_finished(self, best_path):
        self.checkpoint_var.set(best_path)
        self.update_model_status()
        self.set_status("Huấn luyện hoàn tất")
        self.append_log(f"[TRAIN] Hoàn tất. Best checkpoint: {best_path}")
        messagebox.showinfo("Hoàn tất", f"Huấn luyện hoàn tất. Checkpoint lưu tại:\n{best_path}")

    def on_training_error(self, exc):
        self.update_model_status()
        self.set_status("Lỗi khi huấn luyện")
        self.append_log(f"[ERROR] {exc}")
        messagebox.showerror("Lỗi", str(exc))

    def update_model_status(self):
        checkpoint_path = get_checkpoint_path(self.model_var.get())
        if checkpoint_path:
            self.model_status_var.set(f"Đã có checkpoint: {os.path.basename(checkpoint_path)}")
        else:
            self.model_status_var.set("Chưa có checkpoint. Hãy train model trước.")

    def on_training_canceled(self):
        self.update_model_status()
        self.set_status("Đã hủy tác vụ")
        self.append_log("[CANCEL] Đã hủy huấn luyện.")
        messagebox.showinfo("Hủy", "Đã hủy tác vụ huấn luyện.")

    def on_cancel_task(self):
        self.cancel_event.set()
        self.append_log("[CANCEL] Yêu cầu hủy tác vụ.")
        self.set_status("Đang hủy tác vụ...")

    def on_select_image(self):
        path = filedialog.askopenfilename(title="Chọn ảnh test", filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*")])
        if not path:
            return
        checkpoint_path = self.checkpoint_var.get().strip()
        if checkpoint_path == "":
            checkpoint_path = get_checkpoint_path(self.model_var.get())
            if checkpoint_path:
                self.checkpoint_var.set(checkpoint_path)
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            messagebox.showwarning("Checkpoint thiếu", "Vui lòng chọn hoặc huấn luyện checkpoint trước khi dự đoán.")
            return
        try:
            label, prob = predict_single(path, self.model_var.get(), checkpoint_path)
            self.display_single_prediction(path, label, prob)
        except Exception as exc:
            self.append_log(f"[ERROR] Dự đoán thất bại: {exc}")
            messagebox.showerror("Lỗi", f"Không thể dự đoán ảnh:\n{exc}")

    def display_single_prediction(self, image_path, label, prob):
        true_label = get_true_label_from_path(image_path)
        image = Image.open(image_path).convert("RGB")
        image = image.resize((320, 320), RESAMPLE)
        self.image_preview = ImageTk.PhotoImage(image)
        self.preview_label.config(image=self.image_preview)
        self.preview_label.image = self.image_preview

        if true_label:
            display_text = f"Thực tế: {true_label} | Dự đoán: {label} ({prob*100:.1f}%)"
            log_text = f"[PREDICT] {image_path} -> true={true_label}, pred={label} ({prob:.4f})"
        else:
            display_text = f"Dự đoán: {label} ({prob*100:.1f}%)"
            log_text = f"[PREDICT] {image_path} -> pred={label} ({prob:.4f})"

        self.prediction_label.config(text=display_text)
        self.set_status(f"Dự đoán xong: {label} ({prob*100:.1f}%)")
        self.append_log(log_text)

    def on_evaluate_test(self):
        checkpoint_path = self.checkpoint_var.get().strip()
        if checkpoint_path == "":
            checkpoint_path = get_checkpoint_path(self.model_var.get())
            if checkpoint_path:
                self.checkpoint_var.set(checkpoint_path)
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            messagebox.showwarning("Checkpoint thiếu", "Vui lòng chọn hoặc huấn luyện checkpoint trước khi đánh giá test.")
            return

        self.set_status("Đang đánh giá test...")
        self.append_log(f"[EVAL] Bắt đầu đánh giá model {self.model_var.get()} trên tập test")
        thread = threading.Thread(target=self.evaluate_test_thread, args=(self.model_var.get(), checkpoint_path), daemon=True)
        thread.start()

    def evaluate_test_thread(self, model_name, checkpoint_path):
        device = get_device()
        try:
            _, _, test_loader = get_dataloaders(model_name)
            model = get_model(model_name).to(device)
            load_model_checkpoint(model, checkpoint_path, device)
            model.eval()

            y_true = []
            y_pred = []
            total = 0
            correct = 0
            criterion = nn.CrossEntropyLoss()
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    y_true.extend(labels.cpu().tolist())
                    y_pred.extend(preds.cpu().tolist())

            accuracy = correct / total if total else 0.0
            report_text, cm = self.classification_report_manual(y_true, y_pred, LABELS)
            self.root.after(0, lambda: self.on_evaluation_finished(accuracy, report_text, cm, y_true, y_pred))
        except Exception as exc:
            self.root.after(0, lambda: self.on_evaluation_error(exc))

    def on_evaluation_finished(self, accuracy, report_text, cm, y_true, y_pred):
        self.set_status(f"Đã đánh giá test: accuracy {accuracy*100:.2f}%")
        self.evaluation_label.config(text=f"Đã đánh giá test: {accuracy*100:.2f}%")
        self.append_log(f"[EVAL] Accuracy trên tập test: {accuracy*100:.2f}%")
        self.append_log("[EVAL] Classification report:")
        for line in report_text.splitlines():
            self.append_log(line)
        save_path = self.plot_confusion_matrix(y_true, y_pred, LABELS)
        if save_path:
            self.open_confusion_chart_window(save_path)

    def on_evaluation_error(self, exc):
        self.update_model_status()
        self.set_status("Lỗi khi đánh giá test")
        self.append_log(f"[ERROR] Đánh giá test thất bại: {exc}")
        messagebox.showerror("Lỗi", f"Không thể đánh giá test:\n{exc}")

    def classification_report_manual(self, y_true, y_pred, labels):
        cm = self.confusion_matrix_manual(y_true, y_pred, labels)
        report_lines = []
        header = f"{'label':<12} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>8}"
        report_lines.append(header)
        report_lines.append('-' * len(header))

        total_support = 0
        total_correct = 0
        for idx, label in enumerate(labels):
            tp = cm[idx][idx]
            support = sum(cm[idx])
            predicted = sum(row[idx] for row in cm)
            precision = tp / predicted if predicted > 0 else 0.0
            recall = tp / support if support > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            report_lines.append(f"{label:<12} {precision:9.4f} {recall:9.4f} {f1:9.4f} {support:8d}")
            total_support += support
            total_correct += tp

        accuracy = total_correct / total_support if total_support > 0 else 0.0
        report_lines.append('-' * len(header))
        report_lines.append(f"{'accuracy':<12} {accuracy:9.4f} {'':>9} {'':>9} {total_support:8d}")
        return '\n'.join(report_lines), cm

    def confusion_matrix_manual(self, y_true, y_pred, labels):
        matrix = [[0 for _ in labels] for _ in labels]
        for t, p in zip(y_true, y_pred):
            matrix[t][p] += 1
        return matrix

    def plot_confusion_matrix(self, y_true, y_pred, labels, save_path="confusion_matrix.png"):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            cm = self.confusion_matrix_manual(y_true, y_pred, labels)
            self.append_log("[EVAL] Confusion matrix:")
            for label, row in zip(labels, cm):
                self.append_log(f"{label:<12} {' '.join(str(x) for x in row)}")
            self.append_log("Cài matplotlib để lưu ảnh confusion matrix: pip install matplotlib")
            return None

        cm = self.confusion_matrix_manual(y_true, y_pred, labels)
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')

        max_val = max(max(row) for row in cm) if cm else 0
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                color = 'white' if cm[i][j] > max_val / 2 else 'black'
                ax.text(j, i, str(cm[i][j]), ha='center', va='center', color=color, fontsize=8)

        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        self.append_log(f"[EVAL] Confusion matrix saved to {save_path}")
        return save_path

    def display_confusion_image(self, image_path):
        try:
            image = Image.open(image_path).resize((320, 320), RESAMPLE)
            self.confusion_image = ImageTk.PhotoImage(image)
            self.confusion_image_label.config(image=self.confusion_image)
        except Exception as exc:
            self.append_log(f"[ERROR] Không thể hiển thị confusion matrix: {exc}")

    def open_confusion_chart_window(self, image_path):
        try:
            chart_window = tk.Toplevel(self.root)
            chart_window.title("Confusion Matrix")
            chart_window.geometry("620x620")
            chart_canvas = tk.Canvas(chart_window, bg="#ffffff")
            chart_canvas.pack(fill=tk.BOTH, expand=True)
            image = Image.open(image_path)
            image = image.resize((580, 580), RESAMPLE)
            chart_image = ImageTk.PhotoImage(image)
            chart_canvas.create_image(0, 0, anchor="nw", image=chart_image)
            chart_canvas.image = chart_image
        except Exception as exc:
            self.append_log(f"[ERROR] Không thể mở cửa sổ confusion matrix: {exc}")

    def on_random_test(self):
        checkpoint_path = self.checkpoint_var.get().strip()
        if checkpoint_path == "":
            checkpoint_path = get_checkpoint_path(self.model_var.get())
            if checkpoint_path:
                self.checkpoint_var.set(checkpoint_path)
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            messagebox.showwarning("Checkpoint thiếu", "Vui lòng chọn hoặc huấn luyện checkpoint trước khi dự đoán random.")
            return
        try:
            test_images = self._select_random_test_images(20)
            self._display_random_predictions(test_images, checkpoint_path)
        except Exception as exc:
            self.append_log(f"[ERROR] Random prediction failed: {exc}")
            messagebox.showerror("Lỗi", str(exc))

    def _select_random_test_images(self, count):
        if not os.path.isdir(config.TEST_DIR):
            raise FileNotFoundError(f"Thư mục test không tồn tại: {config.TEST_DIR}")
        dataset = ImageFolder(config.TEST_DIR)
        if len(dataset) == 0:
            raise ValueError("Tập test rỗng.")
        indices = random.sample(range(len(dataset)), min(count, len(dataset)))
        return [dataset.imgs[i][0] for i in indices]

    def _display_random_predictions(self, image_paths, checkpoint_path):
        self.random_inner.destroy()
        self.random_inner = ttk.Frame(self.random_canvas)
        self.random_canvas.create_window((0, 0), window=self.random_inner, anchor="nw")
        self.thumbnail_refs = []

        device = get_device()
        model = get_model(self.model_var.get()).to(device)
        load_model_checkpoint(model, checkpoint_path, device)
        model.eval()

        row = 0
        col = 0
        for image_path in image_paths:
            try:
                label, prob = predict_single(image_path, self.model_var.get(), checkpoint_path)
                image = Image.open(image_path).convert("RGB")
                thumbnail = image.resize((120, 120), RESAMPLE)
                thumb_tk = ImageTk.PhotoImage(thumbnail)
                self.thumbnail_refs.append(thumb_tk)
                frame = ttk.Frame(self.random_inner, relief=tk.RIDGE, borderwidth=1, padding=4)
                frame.grid(row=row, column=col, padx=4, pady=4, sticky=tk.N)
                label_widget = ttk.Label(frame, image=thumb_tk)
                label_widget.pack()
                name = os.path.basename(image_path)
                true_label = get_true_label_from_path(image_path)
                ttk.Label(frame, text=name, wraplength=120).pack(pady=(4, 0))
                if true_label:
                    ttk.Label(frame, text=f"True: {true_label}\nPred: {label}\n{prob*100:.1f}%", foreground="blue", wraplength=120).pack()
                else:
                    ttk.Label(frame, text=f"Pred: {label}\n{prob*100:.1f}%", foreground="blue", wraplength=120).pack()
            except Exception as exc:
                self.append_log(f"[ERROR] Không thể dự đoán {image_path}: {exc}")
                ttk.Label(self.random_inner, text=f"Lỗi: {os.path.basename(image_path)}").grid(row=row, column=col)
            col += 1
            if col >= 5:
                col = 0
                row += 1

        self.random_canvas.update_idletasks()
        self.random_canvas.configure(scrollregion=self.random_canvas.bbox("all"))
        self.set_status(f"Hiển thị {len(image_paths)} ảnh random")
        self.append_log(f"[RANDOM] Hiển thị {len(image_paths)} ảnh random từ test")


if __name__ == "__main__":
    root = tk.Tk()
    app = ExpressionApp(root)
    root.mainloop()
