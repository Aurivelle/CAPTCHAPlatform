import os
from pathlib import Path
from PIL import Image
import pytesseract
import torch
from torchvision import transforms

from char import CharDataset  # 如果你想重用 Dataset
from torch_char_cnn import SimpleCNN  # 或直接複製定義
from metrics import compute_accuracy, compute_CER
import pytesseract

# （如果 tesseract.exe 沒在預設路徑，可這樣指定）
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# 1) 基本設定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"
MODEL_PATH = "char_cnn.pt"
CAPTCHA_DIR = Path("dataset")
LABEL_FILE = CAPTCHA_DIR / "labels.txt"

# 2) 載入 CNN 模型
model = SimpleCNN(num_classes=len(CHARSET)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 3) 圖像前處理 (同訓練時)
transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def cnn_predict_captcha(img: Image.Image, length: int):
    """把一張 CAPTCHA 等寬切成 length 張，CNN 預測後串回文字"""
    w, h = img.size
    char_w = w // length
    preds = []
    for i in range(length):
        crop = img.crop((i * char_w, 0, (i + 1) * char_w, h))
        x = transform(crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(x)
        preds.append(CHARSET[out.argmax(1).item()])
    return "".join(preds)


def ocr_predict_captcha(img: Image.Image):
    """用 Tesseract OCR 辨識整張 CAPTCHA"""
    config = f"--psm 8 -c tessedit_char_whitelist={CHARSET}"
    text = pytesseract.image_to_string(img, config=config)
    return text.strip()


# 4) 讀 labels.txt，對每張圖做兩種攻擊，蒐集 preds & gts
gts, preds_cnn, preds_ocr = [], [], []
with open(LABEL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        fn, gt = line.strip().split()
        gts.append(gt)
        img = Image.open(CAPTCHA_DIR / fn).convert("RGB")

        # CNN
        pred1 = cnn_predict_captcha(img, length=len(gt))
        preds_cnn.append(pred1)

        # OCR
        pred2 = ocr_predict_captcha(img)
        preds_ocr.append(pred2)

# 5) 計算指標
acc_cnn = compute_accuracy(preds_cnn, gts)
cer_cnn = compute_CER(preds_cnn, gts)
acc_ocr = compute_accuracy(preds_ocr, gts)
cer_ocr = compute_CER(preds_ocr, gts)

print("=== 攻擊模型 vs OCR 對照 ===")
print(f"CNN attack    → Accuracy: {acc_cnn:.4f}, CER: {cer_cnn:.4f}")
print(f"Tesseract OCR → Accuracy: {acc_ocr:.4f}, CER: {cer_ocr:.4f}")
