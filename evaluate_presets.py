import sys
import os


sys.argv = [sys.argv[0]]
import argparse
import tempfile
from pathlib import Path
import torch
import pandas as pd
import string

from tqdm import tqdm
from data import generate_dataset, get_font_paths
from presets import PRESETS
from metrics import evaluate_folder
from torch_char_cnn import SimpleCNN


parser = argparse.ArgumentParser(
    description="Evaluate model under different CAPTCHA defense presets."
)
parser.add_argument("--samples", type=int, default=500, help="每組樣本數量")
parser.add_argument(
    "--ckpt", type=str, default="char_cnn.pt", help="Char-CNN 模型權重檔案"
)
parser.add_argument("--fonts_dir", type=str, default="fonts", help="字型資料夾路徑")
parser.add_argument(
    "--tess_cmd", type=str, default=None, help="Tesseract-OCR 執行檔路徑"
)
parser.add_argument(
    "--only",
    choices=["all", "cnn", "ocr"],
    default="all",
    help="只做 CNN、只做 OCR，或兩者都做",
)
parser.add_argument(
    "--out_csv", type=str, default="preset_eval.csv", help="輸出 CSV 檔名"
)
args = parser.parse_args()


CHARSET = string.ascii_lowercase + string.digits
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置：{device}")


if args.tess_cmd:
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = args.tess_cmd


model = SimpleCNN(len(CHARSET)).to(device)
ckpt = torch.load(args.ckpt, map_location=device)
model.load_state_dict(ckpt)
model.eval()


from torchvision import transforms

from torchvision.datasets import ImageFolder


DATASET_ROOT = "dataset/clean"
if not os.path.exists(DATASET_ROOT):

    DATASET_ROOT = "./dataset/clean"
dataset = ImageFolder(root=DATASET_ROOT)
CHARSET = dataset.classes

print("模型 label list:", CHARSET)


def char_predict(img):
    x = (
        transforms.Compose(
            [
                transforms.Resize((60, 60)),
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


if args.only in ("all", "ocr"):
    import pytesseract

    def ocr_predict(img):
        txt = pytesseract.image_to_string(
            img, config="--psm 10 -c tessedit_char_whitelist=" + "".join(CHARSET)
        )
        return txt.strip()


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

        record = {}

        if args.only in ("all", "cnn"):
            res_cnn = evaluate_folder(char_predict, tmp, lbl)
            record.update(
                {
                    "CNN_acc": res_cnn["accuracy"],
                    "CNN_CER": res_cnn["cer"],
                }
            )

        if args.only in ("all", "ocr"):
            res_ocr = evaluate_folder(ocr_predict, tmp, lbl)
            record.update(
                {
                    "OCR_acc": res_ocr["accuracy"],
                    "OCR_CER": res_ocr["cer"],
                }
            )
        results[name] = record


df = pd.DataFrame(results).T
print(df.round(4))
df.to_csv(args.out_csv)
print(f"✅ 結果寫入 {args.out_csv}")
