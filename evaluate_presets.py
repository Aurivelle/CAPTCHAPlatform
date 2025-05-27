# evaluate_presets.py
"""
批次評估不同防禦 Preset 對 Char-CNN / OCR 的影響
$ python evaluate_presets.py --samples 600 --out_csv preset_eval.csv
"""

import argparse, tempfile, shutil, string, json
from pathlib import Path
import torch, pandas as pd
from PIL import Image

from presets import PRESETS
from data import generate_dataset
from metrics import evaluate_folder
from torch_char_cnn import SimpleCNN

# ---------- 參數 ----------
CHARSET = string.ascii_lowercase + string.digits
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 載入字元模型 ----------
model = SimpleCNN(len(CHARSET)).to(device)
ckpt = torch.load("char_cnn_wt.pt", map_location=device)
model.load_state_dict(ckpt)
model.eval()

tfm = torch.nn.Sequential(
    torch.nn.Identity()  # 佔位，用 torchvision 會更長；這裡用 metrics 評估時另外給函式
)


def char_predict(img: Image.Image) -> str:
    from torchvision import transforms

    x = (
        transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.Grayscale(1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )(img)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        out = model(x)
    return CHARSET[out.argmax(1).item()]


# ---------- Tesseract ----------
import pytesseract, os, sys

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def ocr_predict(img: Image.Image) -> str:
    txt = pytesseract.image_to_string(
        img, config="--psm 10 -c tessedit_char_whitelist=" + CHARSET
    )
    return txt.strip()


# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument("--samples", type=int, default=500)
ap.add_argument("--out_csv", default="preset_evaluation.csv")
args = ap.parse_args()

results = {}
for name, cfg in {
    "Baseline": dict(font_size=48, x_jitter=0, y_jitter=0, wave_amplitude=0, noise={}),
    **PRESETS,
}.items():
    with tempfile.TemporaryDirectory(prefix=f"{name}_") as tmp:
        generate_dataset(
            output_dir=tmp,
            n_samples=args.samples,
            dataset_config=dict(
                length=1,
                charset=CHARSET,
                font_paths=["fonts/arial.ttf"],
                font_size=cfg["font_size"],
                image_size=(28, 28),
                x_jitter=cfg["x_jitter"],
                y_jitter=cfg["y_jitter"],
                wave_amplitude=cfg["wave_amplitude"],
                seed=42,
            ),
            noise_config=cfg["noise"],
        )
        lbl = Path(tmp) / "labels.txt"
        res_cnn = evaluate_folder(char_predict, tmp, lbl)
        res_ocr = evaluate_folder(ocr_predict, tmp, lbl)
        results[name] = {
            "CNN_acc": res_cnn["accuracy"],
            "CNN_CER": res_cnn["cer"],
            "OCR_acc": res_ocr["accuracy"],
            "OCR_CER": res_ocr["cer"],
        }

df = pd.DataFrame(results).T
print(df.round(3))
df.to_csv(args.out_csv)
print(f"✅ 結果寫入 {args.out_csv}")
