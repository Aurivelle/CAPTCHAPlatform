import sys, os
import argparse, tempfile
from pathlib import Path
import torch
import pandas as pd
import string
from tqdm import tqdm
from data import generate_dataset, get_font_paths
from presets import PRESETS
from metrics import evaluate_folder

parser = argparse.ArgumentParser(
    description="Evaluate all models on the same dataset under each preset."
)
parser.add_argument("--samples", type=int, default=500, help="每組樣本數量")
parser.add_argument(
    "--cnn_ckpt", type=str, default="char_cnn.pt", help="SimpleCNN 權重檔"
)
parser.add_argument(
    "--vgg_ckpt", type=str, default="vgg16_char_best.pt", help="VGG16 權重檔"
)
parser.add_argument("--fonts_dir", type=str, default="fonts", help="字型資料夾路徑")
parser.add_argument(
    "--tess_cmd", type=str, default=None, help="Tesseract-OCR 執行檔路徑"
)
parser.add_argument(
    "--out_csv", type=str, default="preset_multi_eval.csv", help="輸出 CSV 檔名"
)
args = parser.parse_args()

# ==== 統一 label 空間（依據最大涵蓋，建議與 VGG 一致）====
CHARSET = string.digits + string.ascii_lowercase
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置：{device}")

# ==== 載入 SimpleCNN ====
from torch_char_cnn import SimpleCNN
from torchvision import models, transforms

cnn_model = SimpleCNN(len(CHARSET)).to(device)
cnn_ckpt = torch.load(args.cnn_ckpt, map_location=device)
cnn_model.load_state_dict(cnn_ckpt)
cnn_model.eval()
cnn_tf = transforms.Compose(
    [
        transforms.Resize((60, 60)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def cnn_predict(img):
    x = cnn_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = cnn_model(x)
    return CHARSET[out.argmax(1).item()]


# ==== 載入 VGG16 ====
vgg_model = models.vgg16(pretrained=False)
vgg_model.classifier[6] = torch.nn.Linear(4096, len(CHARSET))
vgg_model = vgg_model.to(device)
vgg_ckpt = torch.load(args.vgg_ckpt, map_location=device)
vgg_model.load_state_dict(vgg_ckpt)
vgg_model.eval()
vgg_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)


def vgg_predict(img):
    x = vgg_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = vgg_model(x)
    return CHARSET[out.argmax(1).item()]


# ==== Tesseract OCR ====
import pytesseract

if args.tess_cmd:
    pytesseract.pytesseract.tesseract_cmd = args.tess_cmd


def ocr_predict(img):
    txt = pytesseract.image_to_string(
        img, config="--psm 10 -c tessedit_char_whitelist=" + "".join(CHARSET)
    )
    return txt.strip()


# ==== 每組 preset：產生資料（一次），三模型共用，同步測試 ====
results = {}
for name, cfg in tqdm(PRESETS.items(), desc="Presets"):
    with tempfile.TemporaryDirectory(prefix=f"{name}_") as tmp:
        ds_cfg = {
            "length": 1,
            "charset": CHARSET,
            "font_paths": get_font_paths(args.fonts_dir),
            "font_size": cfg["font_size"],
            "image_size": cfg["image_size"],
            "bg_color": "white",
            "char_color": "black",
            "char_spacing": 4,
            "seed": 42,
            "x_jitter": cfg["x_jitter"],
            "y_jitter": cfg["y_jitter"],
            "wave_amplitude": cfg["wave_amplitude"],
            "background_blur": False,
        }
        generate_dataset(
            output_dir=tmp,
            n_samples=args.samples,
            dataset_config=ds_cfg,
            noise_config=cfg.get("noise", {}),
        )
        lbl = Path(tmp) / "labels.txt"
        rec_cnn = evaluate_folder(cnn_predict, tmp, lbl)
        rec_vgg = evaluate_folder(vgg_predict, tmp, lbl)
        rec_ocr = evaluate_folder(ocr_predict, tmp, lbl)
        results[name] = {
            "CNN_acc": rec_cnn["accuracy"],
            "CNN_CER": rec_cnn["cer"],
            "VGG_acc": rec_vgg["accuracy"],
            "VGG_CER": rec_vgg["cer"],
            "OCR_acc": rec_ocr["accuracy"],
            "OCR_CER": rec_ocr["cer"],
        }

df = pd.DataFrame(results).T
print(df.round(4))
df.to_csv(args.out_csv)
print(f"✅ 結果寫入 {args.out_csv}")
