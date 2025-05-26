import random
import string
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from pathlib import Path

# 自動搜尋 fonts 目錄下所有 .ttf 字型
FONT_DIR = Path("fonts")
FONT_FILES = [str(p) for p in FONT_DIR.glob("*.ttf")]
if not FONT_FILES:
    FONT_FILES = ["fonts/arial.ttf"]

# --- CAPTCHA 生成與擾動 ---
from data import generate_text_image
from perturber import ImagePerturber

# --- 模型推論所需 ---
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import CIFAR100


# ========== Streamlit App 設定 ==========
st.set_page_config(page_title="Custom CAPTCHA Demo", layout="centered")
st.title("🛡️ Customizable CAPTCHA Generator & Model Inference Demo")


# ---- 載入並緩存模型 ----
@st.cache_resource
def load_model(model_path: str = "vgg16_cifar100_best.pt"):
    device = torch.device("cpu")
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 100)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model


model = load_model()

# ---- 定義推論 transform ----
inference_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ---- 載入 CIFAR-100 類別名稱 ----
@st.cache_data
def load_label_names():
    return CIFAR100(root="./data", train=False, download=True).classes


label_names = load_label_names()


# ========== 側欄：CAPTCHA 參數 ==========
with st.sidebar:
    st.header("CAPTCHA Settings")
    text_input = st.text_input("Input text (leave blank → random)")
    length = st.slider("Text length", 4, 8, 5)
    charset = string.ascii_lowercase + string.digits
    font_size = st.slider("Font size", 24, 60, 42)
    bg_color = st.color_picker("Background color", "#FFFFFF")
    char_color = st.color_picker("Text color", "#000000")
    st.markdown("---")
    st.subheader("文字 & 佈局設定")

    # 1) 字型選擇
    font_choice = st.selectbox("Font", ["隨機"] + FONT_FILES)

    # 2) 排版抖動參數
    x_jitter = st.slider("水平抖動 (像素)", min_value=0, max_value=20, value=0)
    y_jitter = st.slider("垂直抖動 (像素)", min_value=0, max_value=20, value=0)
    wave_amplitude = st.slider("波形振幅", min_value=0.0, max_value=10.0, value=0.0)

    st.markdown("---")
    st.subheader("Perturbation Settings")
    apply_gauss = st.checkbox("Gaussian Noise")
    gauss_std = st.slider("Noise σ", 5, 40, 15)
    apply_rot = st.checkbox("Rotation")
    rot_deg = st.slider("Rotation ±deg", 5, 45, 15)
    apply_cut = st.checkbox("Cutout")
    apply_brightness = st.checkbox("Brightness")
    if apply_brightness:
        bright_min, bright_max = st.slider(
            "Brightness factor 範圍", 0.5, 2.0, (0.8, 1.2)
        )

    apply_contrast = st.checkbox("Contrast")
    if apply_contrast:
        cont_min, cont_max = st.slider("Contrast factor 範圍", 0.5, 2.0, (0.8, 1.2))
    cut_num = st.slider("Num patches", 1, 5, 2)
    cut_size = st.slider("Max patch %", 5, 40, 20) / 100


# ========== 生成原始 CAPTCHA ==========
text = (
    text_input[:length]
    if text_input
    else "".join(random.choice(charset) for _ in range(length))
)

# 準備 font_paths list
if font_choice == "隨機":
    font_paths = FONT_FILES
else:
    font_paths = [font_choice]

img = generate_text_image(
    text=text,
    font_paths=font_paths,  # ← 傳入 list
    font_size=font_size,
    image_size=(160, 60),
    bg_color=bg_color,
    char_color=char_color,
    char_spacing=4,
    x_jitter=x_jitter,  # ← 新增
    y_jitter=y_jitter,  # ← 新增
    wave_amplitude=wave_amplitude,  # ← 新增
)

# ========== 應用擾動 ==========
noise_cfg = {}
if apply_gauss:
    noise_cfg["gaussian_noise"] = {"std": gauss_std}
if apply_rot:
    noise_cfg["rotate"] = {"angle_range": (-rot_deg, rot_deg)}
if apply_cut:
    noise_cfg["cutout"] = {"num_patches": cut_num, "max_size": cut_size}
if apply_brightness:
    noise_cfg["brightness"] = {"factor_range": (bright_min, bright_max)}
if apply_contrast:
    noise_cfg["contrast"] = {"factor_range": (cont_min, cont_max)}

perturber = ImagePerturber(seed=42)
noisy_img = perturber.apply(img, noise_cfg) if noise_cfg else img

# ========== 顯示 CAPTCHA ==========
col1, col2 = st.columns(2)
col1.subheader("Original CAPTCHA")
col1.image(img, use_container_width=True)
col2.subheader("Perturbed CAPTCHA")
col2.image(noisy_img, use_container_width=True)


# ========== 模型攻擊 / 推論結果 ==========
st.markdown("---")
st.header("Model Inference Result")

try:
    # 圖像預處理並推論
    x = inference_transform(noisy_img).unsqueeze(0)  # shape (1,3,224,224)
    with torch.no_grad():
        out = model(x)
    pred_idx = out.argmax(1).item()
    pred_label = label_names[pred_idx]

    st.success(f"🖥️ Model predicts: **{pred_label}**")
except Exception as e:
    st.error(f"⚠️ Inference failed: {e}")


# ========== 下載圖片 ==========
st.markdown("---")
with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir) / "captcha.png"
    noisy_img.save(tmp_path)
    st.download_button(
        label="💾 Download CAPTCHA",
        data=open(tmp_path, "rb").read(),
        file_name="captcha.png",
        mime="image/png",
    )
