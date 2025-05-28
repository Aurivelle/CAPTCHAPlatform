import os
import random
import string
import tempfile
from pathlib import Path
from metrics import compute_accuracy, compute_CER

import streamlit as st
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from data import generate_text_image
from perturber import ImagePerturber

import torch
import torch.nn as nn
from torchvision import transforms
from torch_char_cnn import SimpleCNN

# ========== Streamlit App Configuration ==========
st.set_page_config(page_title="CAPTCHA Demo with Char-CNN & OCR", layout="centered")
st.title("🛡️ CAPTCHA Generator & Inference Demo")

# ========== Sidebar: CAPTCHA Settings ==========
with st.sidebar:
    st.header("CAPTCHA Settings")
    text_input = st.text_input("Input text (leave blank → random)")
    length = st.slider("Text length", 1, 5, 1)
    charset = string.ascii_lowercase + string.digits
    font_dir = Path("fonts")
    font_files = [str(p) for p in font_dir.glob("*.ttf")] or ["fonts/arial.ttf"]
    font_choice = st.selectbox("Font", ["Random"] + font_files)
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
    apply_brightness = st.checkbox("Brightness")
    if apply_brightness:
        bright_min, bright_max = st.slider("Brightness range", 0.5, 2.0, (0.8, 1.2))
    apply_contrast = st.checkbox("Contrast")
    if apply_contrast:
        cont_min, cont_max = st.slider("Contrast range", 0.5, 2.0, (0.8, 1.2))

# ========== Generate CAPTCHA Image ==========
text = (
    text_input[:length]
    if text_input
    else "".join(random.choice(charset) for _ in range(length))
)
font_paths = font_files if font_choice == "Random" else [font_choice]

orig_img = generate_text_image(
    text=text,
    font_paths=font_paths,
    font_size=font_size,
    image_size=(160, 60),
    bg_color=bg_color,
    char_color=char_color,
    char_spacing=4,
)

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
noisy_img = perturber.apply(orig_img, noise_cfg) if noise_cfg else orig_img

# ========== Display CAPTCHA ==========
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original CAPTCHA")
    st.image(orig_img, use_container_width=True)
with col2:
    st.subheader("Perturbed CAPTCHA")
    st.image(noisy_img, use_container_width=True)

import numpy as np


import cv2  # <-- 一定要加

try:
    # 將 PIL.Image 轉換成 numpy BGR 格式
    arr = np.array(noisy_img)
    # 如果是灰階，加一個通道
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # 這兩個 yml 參數檔在 OpenCV 資料夾內，若沒有可以從 OpenCV repo 下載（下方補充）
    model_path = cv2.data.haarcascades + "brisque_model_live.yml"
    range_path = cv2.data.haarcascades + "brisque_range_live.yml"

    # 建立評分器
    brisque = cv2.quality.QualityBRISQUE_create(model_path, range_path)
    score = brisque.compute(arr)[0][0]
    st.info(f"BRISQUE 畫質指標 (OpenCV, 越低越好): **{score:.2f}**")
except Exception as e:
    st.warning(f"BRISQUE 計算失敗: {e}")


# ========== Load Char-CNN Model ==========
@st.cache_resource
def load_charcnn(model_path: str = "char_cnn.pt", num_classes: int = 36):
    device = torch.device("cpu")
    model = SimpleCNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model


CHARSET = list(string.ascii_lowercase + string.digits)
char_cnn = load_charcnn(num_classes=len(CHARSET))

# ========= Transforms & Predict Functions ==========
cnn_transform = transforms.Compose(
    [
        transforms.Resize((60, 60)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def charcnn_predict(img: Image.Image) -> str:
    x = cnn_transform(img).unsqueeze(0)
    with torch.no_grad():
        out = char_cnn(x)
    idx = out.argmax(1).item()
    return CHARSET[idx]


def ocr_predict(img: Image.Image) -> str:
    whitelist = "".join(CHARSET)
    txt = pytesseract.image_to_string(
        img, config="--psm 10 -c tessedit_char_whitelist=" + whitelist
    )
    return txt.strip()


# ========== Inference Results ==========
st.markdown("---")
st.header("Model Inference Results")

d1, d2 = st.columns(2)
with d1:
    try:
        pred_char = charcnn_predict(noisy_img)
        st.success(f"🖥️ Char-CNN predicts: **{pred_char}**")
    except Exception as e:
        st.error(f"Char-CNN inference failed: {e}")
with d2:
    try:
        pred_ocr = ocr_predict(noisy_img)
        st.info(f"📝 OCR predicts: **{pred_ocr}**")
    except Exception as e:
        st.error(f"OCR inference failed: {e}")

# Ground truth and success indicator
st.markdown("---")
st.write(f"**Ground truth:** {text}")
if "pred_char" in locals() and pred_char == text:
    st.balloons()
# ========== CAPTCHA 即時評量 ==========

if "pred_char" in locals() and "pred_ocr" in locals():
    st.subheader("CAPTCHA 評分標準（即時比較）")
    # 封裝成 list，方便調用 metrics
    cnn_acc = compute_accuracy([pred_char], [text])
    cnn_cer = compute_CER([pred_char], [text])
    ocr_acc = compute_accuracy([pred_ocr], [text])
    ocr_cer = compute_CER([pred_ocr], [text])

    st.write(
        f"Char-CNN  | 完全正確率(Accuracy): **{cnn_acc:.2f}** | 字元錯誤率(CER): **{cnn_cer:.3f}**"
    )
    st.write(
        f"Tesseract | 完全正確率(Accuracy): **{ocr_acc:.2f}** | 字元錯誤率(CER): **{ocr_cer:.3f}**"
    )


# ========== Download CAPTCHA ==========
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
