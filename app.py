import random
import string
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

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
    st.subheader("Perturbation Settings")
    apply_gauss = st.checkbox("Gaussian Noise")
    gauss_std = st.slider("Noise σ", 5, 40, 15)
    apply_rot = st.checkbox("Rotation")
    rot_deg = st.slider("Rotation ±deg", 5, 45, 15)
    apply_cut = st.checkbox("Cutout")
    cut_num = st.slider("Num patches", 1, 5, 2)
    cut_size = st.slider("Max patch %", 5, 40, 20) / 100


# ========== 生成原始 CAPTCHA ==========
text = (
    text_input[:length]
    if text_input
    else "".join(random.choice(charset) for _ in range(length))
)

img = generate_text_image(
    text=text,
    font_path="fonts/arial.ttf",  # 確保字型檔存在
    font_size=font_size,
    image_size=(160, 60),
    bg_color=bg_color,
    char_color=char_color,
    char_spacing=4,
)

# ========== 應用擾動 ==========
noise_cfg = {}
if apply_gauss:
    noise_cfg["gaussian_noise"] = {"std": gauss_std}
if apply_rot:
    noise_cfg["rotate"] = {"angle_range": (-rot_deg, rot_deg)}
if apply_cut:
    noise_cfg["cutout"] = {"num_patches": cut_num, "max_size": cut_size}

perturber = ImagePerturber(seed=42)
noisy_img = perturber.apply(img, noise_cfg) if noise_cfg else img

# ========== 顯示 CAPTCHA ==========
col1, col2 = st.columns(2)
col1.subheader("Original CAPTCHA")
col1.image(img, use_column_width=True)
col2.subheader("Perturbed CAPTCHA")
col2.image(noisy_img, use_column_width=True)


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
