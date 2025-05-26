# train_char_cnn.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from char import CharDataset  # 請確保已有 char.py 並在裡面定義了 CharDataset

# ========= 超參數設定 =========
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 15
VALID_RATIO = 0.2
# 根據你在 generate_dataset 時用的字符集調整
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"


# ======== 輕量級 CNN 定義 ========
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 3→32 channels
            nn.ReLU(),
            nn.MaxPool2d(2),  # →14×14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # →7×7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    # 1) 設備檢測
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用裝置：", device)

    # 2) 資料預處理與 Dataset 切分
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),  # 縮小加速
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    full_dataset = CharDataset(root_dir="dataset", transform=transform, charset=CHARSET)
    n_valid = int(len(full_dataset) * VALID_RATIO)
    n_train = len(full_dataset) - n_valid
    train_ds, valid_ds = random_split(full_dataset, [n_train, n_valid])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 3) 模型、損失函數、優化器
    model = SimpleCNN(num_classes=len(CHARSET)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    best_path = "char_cnn.pt"

    # 4) 訓練 & 驗證迴圈
    for epoch in range(1, EPOCHS + 1):
        # --- 訓練階段 ---
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        # --- 驗證階段 ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in valid_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"Valid acc={val_acc:.4f}"
        )

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

    print(f"\n最佳驗證準確率：{best_acc:.4f}，模型已保存至 {best_path}")


if __name__ == "__main__":
    main()
