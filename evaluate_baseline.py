# evaluate_baseline.py
import string, shutil
from pathlib import Path
from data import generate_dataset
from metrics import evaluate_folder
from torch_char_cnn import SimpleCNN
import torch
from torchvision import transforms
from PIL import Image

# 1) 生成 Baseline 資料集（無任何 noise）
workdir = Path("tmp_baseline")
if workdir.exists():
    shutil.rmtree(workdir)
generate_dataset(
    output_dir=str(workdir),
    n_samples=2000,
    dataset_config={
        "length": 1,
        "charset": string.ascii_lowercase + string.digits,
        "font_paths": ["fonts/arial.ttf"],
        "font_size": 48,
        "image_size": (28, 28),
        "x_jitter": 0,
        "y_jitter": 0,
        "wave_amplitude": 0,
        "seed": 42,
    },
    noise_config=None,  # ← 這裡不加任何擾動
)

# 2) 載入模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=36).to(device)
model.load_state_dict(torch.load("char_cnn.pt", map_location=device))
model.eval()

# 3) 定義預測函式
CHARSET = string.ascii_lowercase + string.digits
transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),  # 不要轉灰階，保留 RGB
    ]
)


def char_predict(img: Image.Image) -> str:
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    return CHARSET[out.argmax(1).item()]


# 4) 評估
metrics = evaluate_folder(char_predict, str(workdir), str(workdir / "labels.txt"))
print("Baseline 無擾動 → Accuracy:", metrics["accuracy"], " CER:", metrics["cer"])
