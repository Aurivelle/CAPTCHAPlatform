import os
import random
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


# ------------------------------------------------------------
# 模型定義（只留這段在頂層）
# ------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, cls):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 15 * 15, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, cls),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ------------------------------------------------------------
# 主程式訓練流程統統移進 __main__ 區塊
# ------------------------------------------------------------
if __name__ == "__main__":
    # CLI 參數
    parser = argparse.ArgumentParser("CPU-only Char-CNN Trainer")
    parser.add_argument(
        "--dataset_dir", default="dataset", help="資料集根目錄 (需含 clean/<label>/)"
    )
    parser.add_argument("--epochs", type=int, default=60, help="訓練週期數")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="批次大小 (CPU 建議 32~128)"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="初始學習率")
    parser.add_argument(
        "--patience", type=int, default=8, help="Early-stopping patience"
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="DataLoader workers (CPU 可設 0)"
    )
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    parser.add_argument("--save_path", default="char_cnn.pt", help="最佳模型儲存路徑")
    args = parser.parse_args()

    # 隨機種子 & 裝置
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")  # 強制使用 CPU
    print(f"使用裝置：{device}\n")

    # 資料檢查：若資料不存在，自動生成 (僅限單字元)
    if not (Path(args.dataset_dir) / "clean").exists():
        print("⚠️ 偵測到缺少 dataset/clean，開始自動生成 5000 張圖像 …")
        from data import generate_dataset, get_font_paths

        cfg = {
            "length": 1,
            "charset": "0123456789abcdefghijklmnopqrstuvwxyz",
            "font_paths": get_font_paths("fonts"),
            "font_size": 42,
            "image_size": (60, 60),
            "bg_color": "white",
            "char_color": "black",
            "char_spacing": 4,
            "seed": args.seed,
            "x_jitter": 5,
            "y_jitter": 5,
            "wave_amplitude": 2.0,
            "background_blur": True,
        }
        generate_dataset(args.dataset_dir, 5000, cfg, noise_config=None)

    # 資料增強
    transform_train = transforms.Compose(
        [
            transforms.Resize((60, 60)),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=5),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    transform_valid = transforms.Compose(
        [
            transforms.Resize((60, 60)),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # 載入資料集
    dataset = ImageFolder(root=f"{args.dataset_dir}/clean", transform=transform_train)
    num_classes = len(dataset.classes)
    if num_classes == 0:
        raise RuntimeError("dataset/clean 必須包含至少一個子資料夾 (label)")

    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_ds.dataset.transform = transform_train
    val_ds.dataset.transform = transform_valid

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    model = SimpleCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_acc = 0.0
    pat_cnt = 0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        train_correct = train_total = 0
        for imgs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False
        ):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
        train_acc = train_correct / train_total

        # ---- valid ----
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:2d}/{args.epochs} | train_acc={train_acc:.4f} | valid_acc={val_acc:.4f}"
        )

        # ---- checkpoint & early stop ----
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            pat_cnt = 0
        else:
            pat_cnt += 1
            if pat_cnt >= args.patience:
                print("Early stopping triggered.")
                break

    print(f"\n訓練結束，最佳驗證準確率 {best_acc:.4f}，模型已存至 {args.save_path}")
    print(f"總耗時：{(time.time() - start_time)/60:.1f} 分鐘")
